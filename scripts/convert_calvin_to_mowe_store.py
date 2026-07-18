#!/usr/bin/env python3
"""Convert official CALVIN ABC language segments to mowe_feature_store_v1."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.backbones import (
    OpenVLAContextAdapter,
    VisualTargetEncoder,
    resolve_original_openvla_identity,
    teacher_transform_metadata,
)
from mowe_wam.benchmarks.calvin import CalvinActionAdapter, CalvinLanguageSegmentDataset
from mowe_wam.data import (
    MoWECanonicalArchiveWriter,
    MoWEFeatureStoreWriter,
    canonical_conversion_environment,
    episode_partition,
)
from mowe_wam.training.flow_runtime import deep_update, resolve_device
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_nominal_flow_wam.yaml")
    parser.add_argument("--benchmark-config", default="configs/mowe_wam/calvin_abc_d.yaml")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-revision", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--canonical-output",
        help="Optional per-camera Parquet+MP4 archive written in the same source pass.",
    )
    parser.add_argument(
        "--canonical-fps",
        type=float,
        help="Required source frame rate when --canonical-output is used.",
    )
    parser.add_argument("--canonical-episodes-per-chunk", type=int, default=32)
    parser.add_argument("--ffmpeg", help="Explicit ffmpeg executable for canonical MP4 encoding.")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", default="medium")
    parser.add_argument("--audit-output")
    parser.add_argument("--skill-config-output")
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--episodes-per-shard", type=int, default=96)
    parser.add_argument("--limit-segments", type=int)
    parser.add_argument("--allow-partial-audit", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["bf16", "fp16", "float32"], default="bf16")
    return parser.parse_args()


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _class_source_contract(value):
    cls = value.__class__
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:
        path = None
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "source_file": str(Path(path).resolve()) if path is not None else None,
        "source_sha256": (
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            if path is not None and Path(path).is_file()
            else None
        ),
    }


def _pooled_language(backbone, language: str):
    tokens, mask = backbone.encode_language_tokens([language])
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return ((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))[0]


def _encode_segment(torch, backbone, teacher, adapter, segment, batch_size: int):
    from PIL import Image

    transform = backbone.processor.image_processor.apply_transform
    views = []
    dino = []
    length = int(segment["length"])
    for start in range(0, length, batch_size):
        end = min(length, start + batch_size)
        primary = torch.stack(
            [transform(Image.fromarray(frame).convert("RGB")) for frame in segment["rgb_static"][start:end]]
        )
        wrist = torch.stack(
            [
                transform(Image.fromarray(frame).convert("RGB"))
                for frame in segment["rgb_gripper"][start:end]
            ]
        )
        raw = torch.from_numpy(segment["rgb_static"][start:end].copy()).permute(0, 3, 1, 2)
        with torch.no_grad():
            views.append(backbone.encode_pooled_views(primary, wrist).float().cpu())
            dino.append(teacher.encode(raw).float().cpu())
    raw_actions = torch.from_numpy(segment["rel_actions"].copy()).float()
    shared_actions = adapter.to_shared_action(raw_actions).float().cpu()
    with torch.no_grad():
        language = _pooled_language(backbone, segment["language"]).float().cpu()
    return {
        "openvla_views": torch.cat(views).numpy(),
        "dino_tokens": torch.cat(dino).numpy(),
        "actions": shared_actions.numpy(),
        "skills": segment["skill_ids"],
        "language_feature": language.numpy(),
    }


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1 or args.episodes_per_shard < 1:
        raise SystemExit("Batch and shard sizes must be positive.")
    if args.limit_segments is not None and args.limit_segments < 1:
        raise SystemExit("--limit-segments must be positive.")
    if args.allow_partial_audit and args.limit_segments is None:
        raise SystemExit("--allow-partial-audit is permitted only with --limit-segments smoke runs.")
    if args.canonical_output and (
        args.canonical_fps is None
        or args.canonical_fps <= 0
        or args.canonical_episodes_per_chunk < 1
    ):
        raise SystemExit(
            "--canonical-output requires positive --canonical-fps and canonical chunk size."
        )
    canonical_environment = (
        canonical_conversion_environment(args.ffmpeg) if args.canonical_output else None
    )
    torch = require_torch()
    cfg = load_config(args.config)
    benchmark_cfg = load_config(args.benchmark_config)
    official_commit = benchmark_cfg["benchmark"]["official_repo_commit"]
    dataset = CalvinLanguageSegmentDataset(
        args.dataset_root,
        min_segment_length=max(
            int(cfg["data"].get("action_chunk_size", 16)),
            max(
                int(value)
                for value in cfg["data"].get("future_horizons", [1, 4, 8, 16])
            )
            + 1,
        ),
        official_repo_commit=official_commit,
    )
    audit = dataset.audit(limit_segments=args.limit_segments)
    formal_ready = bool(audit["passed"] and args.limit_segments is None)
    if not audit["passed"] and not args.allow_partial_audit:
        raise SystemExit(
            "CALVIN ABC data/taxonomy audit failed. Inspect the report with "
            "scripts/audit_calvin_training_data.py; use --allow-partial-audit only for a limited smoke."
        )
    output_root = Path(args.output)
    audit_path = Path(args.audit_output) if args.audit_output else output_root / "calvin_data_audit.json"
    skill_path = (
        Path(args.skill_config_output)
        if args.skill_config_output
        else output_root / "calvin_skill_experts.json"
    )
    _atomic_json(audit_path, audit)
    _atomic_json(skill_path, dataset.skill_config(audit, audit_path=str(audit_path.resolve())))

    action_stats = audit["action_statistics"]
    action_adapter = CalvinActionAdapter.from_config(action_stats)
    deep_update(
        cfg,
        {
            "backbone": {
                "checkpoint": args.checkpoint,
                "revision": args.backbone_revision,
            },
            "teacher": {"checkpoint": args.teacher_checkpoint},
            "training": {"device": args.device, "precision": args.precision},
        },
    )
    backbone_identity = resolve_original_openvla_identity(
        args.checkpoint,
        revision=cfg["backbone"].get("revision"),
        repo_id=cfg["backbone"].get("repo_id", "openvla/openvla-7b"),
    )
    cfg["backbone"].update(
        {
            "identity": backbone_identity,
            "repo_id": backbone_identity["repo_id"],
            "revision": backbone_identity["revision"],
        }
    )
    device = resolve_device(cfg)
    backbone_cfg = dict(cfg["backbone"])
    backbone_cfg.pop("mode", None)
    backbone_cfg.pop("hidden_dim", None)
    backbone_cfg.update(
        {
            "checkpoint": args.checkpoint,
            "device": device,
            "dtype": args.precision if device.startswith("cuda") else "float32",
            "freeze_backbone": True,
            "feature_source": "pre_action_context",
            "num_images_in_input": 2,
        }
    )
    backbone = OpenVLAContextAdapter(**backbone_cfg)
    teacher = VisualTargetEncoder(
        checkpoint=args.teacher_checkpoint,
        spatial_grid=int(cfg["teacher"].get("spatial_grid", 4)),
        target_dim=int(cfg["teacher"].get("target_dim", 384)),
        num_spatial_tokens=int(cfg["teacher"].get("spatial_tokens", 16)),
        device=device,
        dtype=args.precision if device.startswith("cuda") else "float32",
    )
    joint_action_statistics = {
        "q01": list(action_stats["motion_q01"]) + [0.0],
        "q99": list(action_stats["motion_q99"]) + [1.0],
        "mask": [True] * 6 + [False],
        "motion_dim": 6,
        "gripper_contract": "canonical_absolute_binary_no_normalization",
        "method": "calvin_abc_language_frames_motion_q01_q99",
        "raw_calvin_contract": action_adapter.contract(),
    }
    skill_metadata = {
        "format": "calvin_language_segment_skills_v1",
        "fingerprint_sha256": dataset.annotation_fingerprint,
        "label_version": audit["label_version"],
        "alignment_verified": False,
        "join_key": "auto_lang_ann.info.indx inclusive segment",
    }
    transform = teacher_transform_metadata(
        args.teacher_checkpoint,
        list(backbone.resize_resolution),
        int(cfg["teacher"].get("spatial_grid", 4)),
    )
    source_contract = {
        "dataset_contract": audit["dataset_contract"],
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "dataset_names": ["calvin_abc_language_segments"],
        "official_repo_commit": official_commit,
        "train_eval_isolation": audit["train_eval_isolation"],
        "annotation_fingerprint": dataset.annotation_fingerprint,
        "skill_sidecar_metadata": skill_metadata,
        "skill_sidecar_fingerprint": dataset.annotation_fingerprint,
        "joint_action_statistics": joint_action_statistics,
        "formal_training_ready": formal_ready,
        "partial_limit_segments": args.limit_segments,
        "expected_counts": {
            "episode_count": int(audit["segments"]),
            "frame_count": int(audit["transitions"]),
            "window_count": int(audit["valid_windows_h8"]),
        },
        "openvla_checkpoint": str(args.checkpoint),
        "openvla_identity": backbone_identity,
        "openvla_feature_source": "pre_action_context",
        "openvla_num_images_in_input": 2,
        "openvla_view_order": ["primary", "wrist"],
        "openvla_dtype": args.precision,
        "openvla_hidden_dim": int(backbone.hidden_dim),
        "openvla_resize_resolution": list(backbone.resize_resolution),
        "openvla_processor": _class_source_contract(backbone.processor),
        "openvla_image_processor": _class_source_contract(backbone.processor.image_processor),
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_transform_metadata": transform,
        "conversion_script": "scripts/convert_calvin_to_mowe_store.py",
        "conversion_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "image_aug": False,
    }
    writer = MoWEFeatureStoreWriter(
        output_root,
        source_contract=source_contract,
        history_length=int(cfg["data"].get("history_length", 8)),
        long_memory_slots=int(cfg["data"].get("long_memory_slots", 4)),
        future_horizons=cfg["data"].get("future_horizons", [1, 4, 8, 16]),
        action_chunk_size=int(cfg["data"].get("action_chunk_size", 16)),
        episodes_per_shard=args.episodes_per_shard,
    )
    canonical_writer = (
        MoWECanonicalArchiveWriter(
            args.canonical_output,
            source_contract=source_contract,
            fps=args.canonical_fps,
            episodes_per_chunk=args.canonical_episodes_per_chunk,
            ffmpeg=args.ffmpeg,
            video_crf=args.video_crf,
            video_preset=args.video_preset,
        )
        if args.canonical_output
        else None
    )
    converted = 0
    canonical_converted = 0
    skipped = 0
    validation_fraction = float(cfg["data"].get("validation_fraction", 0.05))
    split_seed = int(cfg["data"].get("split_seed", 17))
    for record in dataset.iter_segment_records(limit=args.limit_segments):
        episode_id = dataset.segment_episode_id(record)
        needs_feature = not writer.has_episode(episode_id)
        needs_canonical = canonical_writer is not None and not canonical_writer.has_episode(
            episode_id
        )
        if not needs_feature and not needs_canonical:
            skipped += 1
            continue
        segment = dataset.load_segment(record)
        partition = episode_partition(
            episode_id,
            validation_fraction=validation_fraction,
            split_seed=split_seed,
        )
        source_file_key = (
            f"{dataset.frame_path(segment['start_frame'])}:"
            f"{dataset.frame_path(segment['end_frame']).name}"
        )
        encoded = None
        if needs_feature:
            encoded = _encode_segment(
                torch, backbone, teacher, action_adapter, segment, args.encode_batch_size
            )
            writer.add_episode(
                episode_id=episode_id,
                dataset_name="calvin_abc_language_segments",
                partition=partition,
                language=segment["language"],
                language_feature=encoded["language_feature"],
                openvla_views=encoded["openvla_views"],
                dino_tokens=encoded["dino_tokens"],
                actions=encoded["actions"],
                skills=encoded["skills"],
                source_traj_index=int(segment["segment_index"]),
                source_file_key=source_file_key,
            )
            converted += 1
        if needs_canonical:
            if encoded is None:
                raw_actions = torch.from_numpy(segment["rel_actions"].copy()).float()
                shared_actions = action_adapter.to_shared_action(raw_actions).cpu().numpy()
                skills = segment["skill_ids"]
            else:
                shared_actions = encoded["actions"]
                skills = encoded["skills"]
            canonical_writer.add_episode(
                episode_id=episode_id,
                dataset_name="calvin_abc_language_segments",
                partition=partition,
                language=segment["language"],
                actions=shared_actions,
                skills=skills,
                primary_frames=segment["rgb_static"],
                wrist_frames=segment["rgb_gripper"],
                proprio=segment["robot_obs"],
                source_traj_index=int(segment["segment_index"]),
                source_file_key=source_file_key,
            )
            canonical_converted += 1
        if (converted + canonical_converted) % 10 == 0:
            print(
                f"feature_segments={converted} canonical_segments={canonical_converted} "
                f"skipped_segments={skipped}",
                flush=True,
            )
    manifest = writer.finalize()
    canonical_manifest = canonical_writer.finalize() if canonical_writer is not None else None
    print(
        json.dumps(
            {
                "output": str(output_root.resolve()),
                "converted_segments": converted,
                "canonical_converted_segments": canonical_converted,
                "skipped_segments": skipped,
                "episode_count": manifest["episode_count"],
                "frame_count": manifest["frame_count"],
                "window_count": manifest["window_count"],
                "formal_training_ready": manifest["formal_training_ready"],
                "completion_contract": manifest.get("completion_contract"),
                "canonical_output": (
                    str(Path(args.canonical_output).resolve()) if args.canonical_output else None
                ),
                "canonical_chunk_count": (
                    int(canonical_manifest["chunk_count"])
                    if canonical_manifest is not None
                    else 0
                ),
                "canonical_environment": canonical_environment,
                "audit": str(audit_path.resolve()),
                "skill_config": str(skill_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.limit_segments is None and not manifest.get(
        "formal_training_ready", False
    ):
        raise RuntimeError(
            "Converted CALVIN feature store is incomplete; inspect completion_contract."
        )


if __name__ == "__main__":
    main()
