#!/usr/bin/env python3
"""Convert deterministic LIBERO RLDS episodes into mowe_feature_store_v1."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.backbones import (
    OpenVLAContextAdapter,
    VisualTargetEncoder,
    resolve_original_openvla_identity,
    teacher_transform_metadata,
)
from mowe_wam.data import (
    LiberoSequenceDataset,
    MoWECanonicalArchiveWriter,
    MoWEFeatureStoreWriter,
    canonical_conversion_environment,
    episode_partition,
    rlds_manifest_fingerprint,
    source_episode_key,
)
from mowe_wam.training.flow_runtime import deep_update, resolve_device
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_nominal_flow_wam.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint", help="Local snapshot of original openvla/openvla-7b.")
    parser.add_argument(
        "--backbone-revision",
        required=True,
        help="Immutable 40-character Hugging Face commit.",
    )
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument(
        "--skill-sidecar",
        help="Explicit CoT skill sidecar; overrides data.skill_sidecar_path in the config.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--canonical-output",
        help="Optional LeRobot-v3-style Parquet+MP4 archive written in the same RLDS pass.",
    )
    parser.add_argument(
        "--canonical-fps",
        type=float,
        help="Required source control/frame rate when --canonical-output is used.",
    )
    parser.add_argument("--canonical-episodes-per-chunk", type=int, default=32)
    parser.add_argument("--ffmpeg", help="Explicit ffmpeg executable for canonical MP4 encoding.")
    parser.add_argument("--video-crf", type=int, default=18)
    parser.add_argument("--video-preset", default="medium")
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--episodes-per-shard", type=int, default=96)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["bf16", "fp16", "float32"], default="bf16")
    return parser.parse_args()


def _pooled_language(backbone, language: str):
    tokens, mask = backbone.encode_language_tokens([language])
    weights = mask.to(tokens.dtype).unsqueeze(-1)
    return ((tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0))[0]


def _class_source_contract(value) -> dict[str, str | None]:
    cls = value.__class__
    try:
        path = inspect.getsourcefile(cls)
    except TypeError:
        path = None
    source_hash = None
    if path is not None and Path(path).is_file():
        source_hash = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "source_file": str(Path(path).resolve()) if path is not None else None,
        "source_sha256": source_hash,
    }


def _encode_episode(torch, backbone, teacher, episode, batch_size: int):
    views = []
    dino = []
    for start in range(0, len(episode), batch_size):
        chunk = episode[start : start + batch_size]
        primary = torch.stack([step["policy_pixel_values_primary"] for step in chunk])
        wrist = torch.stack([step["policy_pixel_values_wrist"] for step in chunk])
        raw = torch.stack([step["raw_pixel_values"] for step in chunk])
        with torch.no_grad():
            views.append(backbone.encode_pooled_views(primary, wrist).float().cpu())
            dino.append(teacher.encode(raw).float().cpu())
    actions = torch.stack([step["actions"][0].float() for step in episode]).cpu()
    skills = torch.tensor(
        [int(step.get("expert_skill_label", -1)) for step in episode], dtype=torch.int8
    )
    with torch.no_grad():
        language_feature = _pooled_language(backbone, episode[0]["language"]).float().cpu()
    return {
        "openvla_views": torch.cat(views).numpy(),
        "dino_tokens": torch.cat(dino).numpy(),
        "actions": actions.numpy(),
        "skills": skills.numpy(),
        "language_feature": language_feature.numpy(),
    }


def main() -> None:
    args = parse_args()
    if args.encode_batch_size < 1 or args.episodes_per_shard < 1:
        raise SystemExit("Batch and shard sizes must be positive.")
    if args.limit_episodes is not None and args.limit_episodes < 1:
        raise SystemExit("--limit-episodes must be positive when provided.")
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
    deep_update(
        cfg,
        {
            "data": {
                "data_root": args.data_root,
                "skill_sidecar_path": args.skill_sidecar,
            },
            "backbone": {
                "checkpoint": args.checkpoint,
                "revision": args.backbone_revision,
            },
            "teacher": {"checkpoint": args.teacher_checkpoint},
            "training": {"device": args.device, "precision": args.precision},
        },
    )
    if cfg["data"].get("data_root") in {None, "TBD"}:
        raise SystemExit("A concrete --data-root is required.")
    checkpoint = cfg["backbone"].get("checkpoint") or cfg["backbone"].get("vla_path")
    if checkpoint in {None, "TBD"}:
        raise SystemExit("A concrete --checkpoint is required.")
    backbone_identity = resolve_original_openvla_identity(
        checkpoint,
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
            "checkpoint": checkpoint,
            "device": device,
            "dtype": args.precision if device.startswith("cuda") else "float32",
            "freeze_backbone": True,
            "feature_source": "pre_action_context",
            "num_images_in_input": 2,
        }
    )
    backbone = OpenVLAContextAdapter(**backbone_cfg)
    teacher = VisualTargetEncoder(
        checkpoint=cfg["teacher"]["checkpoint"],
        spatial_grid=int(cfg["teacher"].get("spatial_grid", 4)),
        target_dim=int(cfg["teacher"].get("target_dim", 384)),
        num_spatial_tokens=int(cfg["teacher"].get("spatial_tokens", 16)),
        device=device,
        dtype=args.precision if device.startswith("cuda") else "float32",
    )
    dataset = LiberoSequenceDataset(
        dataset_root=cfg["data"]["data_root"],
        processor=backbone.processor,
        dataset_names=cfg["data"]["dataset_names"],
        history_length=int(cfg["data"].get("history_length", 8)),
        long_memory_slots=int(cfg["data"].get("long_memory_slots", 4)),
        future_horizons=cfg["data"].get("future_horizons", [1, 4, 8, 16]),
        split=cfg["data"].get("split", "train"),
        resize_resolution=tuple(backbone.resize_resolution),
        image_aug=False,
        use_proprio=bool(args.canonical_output),
        openvla_root=cfg["backbone"].get("openvla_root", "external/openvla-oft"),
        limit=None,
        joint_action_normalization=bool(cfg["data"].get("joint_action_normalization", True)),
        skill_sidecar_path=cfg["data"].get("skill_sidecar_path"),
        assume_sidecar_timestep_aligned=bool(
            cfg["data"].get("assume_sidecar_timestep_aligned", True)
        ),
        window_shuffle_buffer_size=0,
        episode_partition_name="all",
        validation_fraction=float(cfg["data"].get("validation_fraction", 0.05)),
        split_seed=int(cfg["data"].get("split_seed", 17)),
        action_chunk_size=int(cfg["data"].get("action_chunk_size", 16)),
        distributed_rank=0,
        distributed_world_size=1,
        tf_frame_parallel_calls=1,
    )
    transform = teacher_transform_metadata(
        cfg["teacher"]["checkpoint"],
        list(backbone.resize_resolution),
        int(cfg["teacher"].get("spatial_grid", 4)),
    )
    expected_episode_count = sum(int(item.dataset_length) for item in dataset.datasets)
    expected_frame_count = sum(
        int(dataset.action_statistics[name]["num_transitions"])
        for name in dataset.dataset_names
    )
    maximum_offset = max(
        max(int(value) for value in cfg["data"].get("future_horizons", [1, 4, 8, 16])),
        int(cfg["data"].get("action_chunk_size", 16)) - 1,
    )
    expected_counts = {
        "episode_count": expected_episode_count,
        "frame_count": expected_frame_count,
        "window_count": expected_frame_count
        - maximum_offset * expected_episode_count,
    }
    source_contract = {
        "rlds_manifest_fingerprint": rlds_manifest_fingerprint(
            cfg["data"]["data_root"], cfg["data"]["dataset_names"]
        ),
        "dataset_names": list(cfg["data"]["dataset_names"]),
        "skill_sidecar_metadata": dataset.skill_sidecar_metadata,
        "skill_sidecar_fingerprint": (dataset.skill_sidecar_metadata or {}).get(
            "fingerprint_sha256"
        ),
        "joint_action_statistics": dataset.joint_action_statistics,
        "openvla_checkpoint": str(checkpoint),
        "openvla_identity": backbone_identity,
        "openvla_feature_source": "pre_action_context",
        "openvla_num_images_in_input": 2,
        "openvla_view_order": ["primary", "wrist"],
        "openvla_dtype": args.precision,
        "openvla_hidden_dim": int(backbone.hidden_dim),
        "openvla_resize_resolution": list(backbone.resize_resolution),
        "openvla_processor": _class_source_contract(backbone.processor),
        "openvla_image_processor": _class_source_contract(
            backbone.processor.image_processor
        ),
        "teacher_checkpoint": str(cfg["teacher"]["checkpoint"]),
        "teacher_transform_metadata": transform,
        "conversion_script": "scripts/convert_rlds_to_mowe_store.py",
        "conversion_script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "image_aug": False,
        "formal_training_ready": args.limit_episodes is None,
        "conversion_limit_episodes": args.limit_episodes,
        "expected_counts": expected_counts,
    }
    writer = MoWEFeatureStoreWriter(
        args.output,
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
    completed_source_identities = writer.source_episode_identities()
    if canonical_writer is not None:
        # An episode may be filtered before decoding only when every requested
        # output already owns a durable committed/staged copy.
        completed_source_identities &= canonical_writer.source_episode_identities()
    completed_source_keys = {
        source_episode_key(file_key, trajectory_index)
        for file_key, trajectory_index in completed_source_identities
    }
    dataset.exclude_source_episodes_before_frame_transform(completed_source_keys)
    converted = 0
    canonical_converted = 0
    skipped = 0
    seen = 0
    for episode_id, episode in dataset.iter_transformed_episodes():
        if args.limit_episodes is not None and seen >= args.limit_episodes:
            break
        seen += 1
        needs_feature = not writer.has_episode(episode_id)
        needs_canonical = canonical_writer is not None and not canonical_writer.has_episode(
            episode_id
        )
        if not needs_feature and not needs_canonical:
            skipped += 1
            continue
        if not episode:
            continue
        partition = episode_partition(
            episode_id,
            validation_fraction=float(cfg["data"].get("validation_fraction", 0.05)),
            split_seed=int(cfg["data"].get("split_seed", 17)),
        )
        features = None
        if needs_feature:
            # Formal features are computed directly from the original RLDS
            # tensors before any lossy MP4 archival encoding.
            features = _encode_episode(torch, backbone, teacher, episode, args.encode_batch_size)
            writer.add_episode(
                episode_id=episode_id,
                dataset_name=episode[0]["dataset_name"],
                partition=partition,
                language=episode[0]["language"],
                language_feature=features["language_feature"],
                openvla_views=features["openvla_views"],
                dino_tokens=features["dino_tokens"],
                actions=features["actions"],
                skills=features["skills"],
                source_traj_index=episode[0].get("source_traj_index"),
                source_file_key=episode[0].get("source_file_key"),
            )
            converted += 1
        if needs_canonical:
            actions = (
                features["actions"]
                if features is not None
                else torch.stack([step["actions"][0].float() for step in episode]).numpy()
            )
            skills = (
                features["skills"]
                if features is not None
                else torch.tensor(
                    [int(step.get("expert_skill_label", -1)) for step in episode],
                    dtype=torch.int8,
                ).numpy()
            )
            proprio = (
                torch.stack([step["proprio"].float() for step in episode]).numpy()
                if all("proprio" in step for step in episode)
                else None
            )
            canonical_writer.add_episode(
                episode_id=episode_id,
                dataset_name=episode[0]["dataset_name"],
                partition=partition,
                language=episode[0]["language"],
                actions=actions,
                skills=skills,
                primary_frames=[step["raw_pixel_values"] for step in episode],
                wrist_frames=[step["raw_wrist_pixel_values"] for step in episode],
                proprio=proprio,
                source_traj_index=episode[0].get("source_traj_index"),
                source_file_key=episode[0].get("source_file_key"),
            )
            canonical_converted += 1
        if (converted + canonical_converted) % 10 == 0:
            print(
                "feature_episodes="
                f"{converted} canonical_episodes={canonical_converted} "
                f"skipped_episodes={skipped}",
                flush=True,
            )
    manifest = writer.finalize()
    canonical_manifest = canonical_writer.finalize() if canonical_writer is not None else None
    print(
        json.dumps(
            {
                "output": str(Path(args.output).resolve()),
                "converted_episodes": converted,
                "canonical_converted_episodes": canonical_converted,
                "skipped_episodes": skipped,
                "episode_count": manifest["episode_count"],
                "frame_count": manifest["frame_count"],
                "window_count": manifest["window_count"],
                "shard_count": len(manifest["shards"]),
                "canonical_output": (
                    str(Path(args.canonical_output).resolve()) if args.canonical_output else None
                ),
                "canonical_chunk_count": (
                    int(canonical_manifest["chunk_count"])
                    if canonical_manifest is not None
                    else 0
                ),
                "canonical_environment": canonical_environment,
                "pre_frame_filtered_episodes": len(completed_source_keys),
                "formal_training_ready": args.limit_episodes is None,
                "completion_contract": manifest.get("completion_contract"),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if args.limit_episodes is None and not manifest.get("formal_training_ready", False):
        raise RuntimeError(
            "Converted feature store is not formal-training ready; inspect completion_contract."
        )


if __name__ == "__main__":
    main()
