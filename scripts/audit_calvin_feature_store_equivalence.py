#!/usr/bin/env python3
"""Compare official CALVIN ABC frames with their frozen-feature training store."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.backbones import (
    OpenVLAContextAdapter,
    VisualTargetEncoder,
    openvla_identities_match,
    resolve_original_openvla_identity,
)
from mowe_wam.benchmarks.calvin import CalvinActionAdapter, resolve_calvin_training_dataset
from mowe_wam.data import LatentWAMCollator, MoWEFeatureWindowDataset
from mowe_wam.training.flow_runtime import (
    build_flow_policy,
    deep_update,
    resolve_device,
    resolve_feature_store_contract,
)
from mowe_wam.training.latent_losses import flow_wam_skill_losses
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch
from scripts.convert_calvin_to_mowe_store import _encode_segment


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/mowe_wam/ddp8_calvin_nominal_flow_feature_store.yaml",
    )
    parser.add_argument(
        "--benchmark-config", default="configs/mowe_wam/calvin_abc_d.yaml"
    )
    parser.add_argument("--store", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "official_npz", "rlds"],
        default="auto",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-revision", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument(
        "--stage",
        choices=["nominal_flow_pretrain", "expert_warmstart", "joint"],
        default="nominal_flow_pretrain",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--precision", choices=["bf16", "fp16", "float32"], default="bf16"
    )
    parser.add_argument("--feature-atol", type=float, default=0.03)
    parser.add_argument("--output-atol", type=float, default=0.10)
    parser.add_argument("--loss-atol", type=float, default=0.05)
    parser.add_argument("--output")
    return parser.parse_args()


def _sparse_prefix_indices(prefix_end: int, slots: int) -> list[int]:
    if prefix_end <= 0 or slots <= 0:
        return []
    if prefix_end <= slots:
        return list(range(prefix_end))
    if slots == 1:
        return [0]
    return [(slot * (prefix_end - 1)) // (slots - 1) for slot in range(slots)]


def _window_from_encoded_segment(
    torch, encoded, segment, step_id, window_contract, *, dataset_name=None
):
    """Independently rebuild the map-style window from online encoded arrays."""

    import numpy as np

    step_id = int(step_id)
    history_slots = int(window_contract["history_length"]) - 1
    long_slots = int(window_contract["long_memory_slots"])
    horizons = tuple(int(value) for value in window_contract["future_horizons"])
    chunk_size = int(window_contract["action_chunk_size"])
    length = int(segment["length"])
    maximum = max(max(horizons), chunk_size - 1)
    if not 0 <= step_id < length - maximum:
        raise IndexError("CALVIN window start is outside the encoded segment contract.")

    views = np.asarray(encoded["openvla_views"])
    targets = np.asarray(encoded["dino_tokens"])
    actions = np.asarray(encoded["actions"], dtype=np.float32)
    skills = np.asarray(encoded["skills"], dtype=np.int64)
    view_shape = tuple(views.shape[1:])

    history_indices = list(range(max(0, step_id - history_slots), step_id))
    history_pad = history_slots - len(history_indices)
    history_views = np.zeros((history_slots, *view_shape), dtype=views.dtype)
    history_actions = np.zeros((history_slots, 7), dtype=np.float32)
    if history_indices:
        history_views[history_pad:] = views[history_indices]
        history_actions[history_pad:] = actions[history_indices]

    short_start = max(0, step_id - history_slots)
    long_indices = _sparse_prefix_indices(short_start, long_slots)
    long_pad = long_slots - len(long_indices)
    long_views = np.zeros((long_slots, *view_shape), dtype=views.dtype)
    long_actions = np.zeros((long_slots, 7), dtype=np.float32)
    if long_indices:
        long_views[long_pad:] = views[long_indices]
        long_actions[long_pad:] = actions[long_indices]

    target_actions = actions[step_id : step_id + chunk_size].copy()
    skill_labels = skills[step_id : step_id + chunk_size].copy()
    future_indices = [step_id + horizon for horizon in horizons]
    return {
        "episode_id": str(segment["episode_id"]),
        "step_id": step_id,
        "dataset_name": dataset_name or "calvin_abc_language_segments",
        "language": str(segment["language"]),
        "current_visual_views": torch.from_numpy(views[step_id].copy()),
        "history_visual_views": torch.from_numpy(history_views.copy()),
        "long_history_visual_views": torch.from_numpy(long_views.copy()),
        "precomputed_language": torch.from_numpy(
            np.asarray(encoded["language_feature"]).copy()
        ),
        "history_actions": torch.from_numpy(history_actions),
        "history_mask": torch.tensor(
            [False] * history_pad + [True] * len(history_indices) + [True],
            dtype=torch.bool,
        ),
        "long_history_actions": torch.from_numpy(long_actions),
        "long_history_mask": torch.tensor(
            [False] * long_pad + [True] * len(long_indices), dtype=torch.bool
        ),
        "current_latent_target": torch.from_numpy(targets[step_id].copy()),
        "future_latent_targets": torch.from_numpy(targets[future_indices].copy()),
        "future_horizons": torch.tensor(horizons, dtype=torch.long),
        "future_mask": torch.ones(len(horizons), dtype=torch.bool),
        "target_actions": torch.from_numpy(target_actions),
        "target_motion": torch.from_numpy(target_actions[:, :6].copy()),
        "target_gripper": torch.from_numpy(target_actions[:, 6:7].copy()),
        "expert_skill_labels": torch.from_numpy(skill_labels),
        "expert_skill_mask": torch.from_numpy(skill_labels != -1),
        "expert_label_source": [
            "sidecar" if int(value) != -1 else "unknown" for value in skill_labels
        ],
    }


def _tensor_error(left, right):
    difference = (left.detach().float().cpu() - right.detach().float().cpu()).abs()
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
    }


def _set_rng(torch, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.encode_batch_size < 1:
        raise SystemExit("--samples and --encode-batch-size must be positive.")
    if min(args.feature_atol, args.output_atol, args.loss_atol) < 0:
        raise SystemExit("Equivalence tolerances must be non-negative.")

    torch = require_torch()
    cfg = load_config(args.config)
    deep_update(
        cfg,
        {
            "backbone": {
                "mode": "precomputed_features",
                "feature_source": "pre_action_context_cache",
                "checkpoint": args.checkpoint,
                "dtype": args.precision,
            },
            "teacher": {
                "checkpoint": args.teacher_checkpoint,
                "cache_path": None,
                "inference_enabled": False,
                "dtype": args.precision,
            },
            "data": {
                "backend": "mowe_feature_store_v1",
                "feature_store_path": args.store,
                "image_aug": False,
                "num_workers": 0,
                "pin_memory": False,
            },
            "training": {
                "device": args.device,
                "precision": args.precision,
                "batch_size": 1,
            },
            "validation": {"enabled": False},
        },
    )
    manifest = resolve_feature_store_contract(cfg)
    source = manifest.get("source_contract", {})
    requested_identity = resolve_original_openvla_identity(
        args.checkpoint,
        revision=args.backbone_revision,
        repo_id=source.get("openvla_identity", {}).get("repo_id", "openvla/openvla-7b"),
    )
    if not openvla_identities_match(source.get("openvla_identity"), requested_identity):
        raise RuntimeError("CALVIN equivalence checkpoint differs from the store OpenVLA identity.")
    benchmark_cfg = load_config(args.benchmark_config)
    dataset = resolve_calvin_training_dataset(
        args.dataset_root,
        dataset_format=args.dataset_format,
        min_segment_length=max(
            int(cfg["data"]["action_chunk_size"]),
            max(int(value) for value in cfg["data"]["future_horizons"]) + 1,
        ),
        official_repo_commit=benchmark_cfg["benchmark"]["official_repo_commit"],
    )
    if source.get("dataset_fingerprint") != dataset.dataset_fingerprint:
        raise RuntimeError("CALVIN feature store dataset fingerprint differs from raw ABC data.")
    if source.get("annotation_fingerprint") != dataset.annotation_fingerprint:
        raise RuntimeError("CALVIN feature store annotation fingerprint differs from raw ABC data.")

    device = resolve_device(cfg)
    encoder_cfg = dict(cfg["backbone"])
    encoder_cfg.pop("mode", None)
    encoder_cfg.pop("hidden_dim", None)
    encoder_cfg.update(
        {
            "checkpoint": args.checkpoint,
            "device": device,
            "dtype": args.precision if device.startswith("cuda") else "float32",
            "freeze_backbone": True,
            "feature_source": "pre_action_context",
            "num_images_in_input": 2,
        }
    )
    backbone = OpenVLAContextAdapter(**encoder_cfg)
    teacher = VisualTargetEncoder(
        checkpoint=args.teacher_checkpoint,
        spatial_grid=int(cfg["teacher"].get("spatial_grid", 4)),
        target_dim=int(cfg["teacher"].get("target_dim", 384)),
        num_spatial_tokens=int(cfg["teacher"].get("spatial_tokens", 16)),
        device=device,
        dtype=args.precision if device.startswith("cuda") else "float32",
    )
    raw_contract = source.get("joint_action_statistics", {}).get(
        "raw_calvin_contract"
    )
    if not isinstance(raw_contract, dict):
        raise RuntimeError("CALVIN feature store is missing raw action adapter contract.")
    action_adapter = CalvinActionAdapter.from_config(raw_contract)
    model = build_flow_policy(cfg, include_teacher=False)
    model.eval()

    cached_dataset = MoWEFeatureWindowDataset(args.store, partition="train")
    sample_count = min(args.samples, len(cached_dataset))
    selected = random.Random(args.seed).sample(range(len(cached_dataset)), sample_count)
    cached_samples = {index: cached_dataset[index] for index in selected}
    requested_pairs = {
        (str(sample["episode_id"]), int(sample["step_id"])): sample
        for sample in cached_samples.values()
    }
    records_by_episode: dict[str, list[dict]] = {}
    for record in dataset.iter_segment_records():
        episode_id = dataset.segment_episode_id(record)
        if any(pair[0] == episode_id for pair in requested_pairs):
            records_by_episode.setdefault(episode_id, []).append(record)

    collator = LatentWAMCollator()
    records = []
    found = set()
    feature_keys = (
        "current_visual_views",
        "history_visual_views",
        "long_history_visual_views",
        "precomputed_language",
        "current_latent_target",
        "future_latent_targets",
        "target_actions",
        "expert_skill_labels",
    )
    output_keys = (
        "nominal_actions",
        "future_latents",
        "route_world_tokens",
        "router_logits",
        "motion_actions",
        "actions",
    )
    with torch.no_grad():
        for episode_id, segment_records in records_by_episode.items():
            if len(segment_records) != 1:
                raise RuntimeError(f"CALVIN episode ID is not unique: {episode_id}")
            segment = dataset.load_segment(segment_records[0])
            encoded = _encode_segment(
                torch,
                backbone,
                teacher,
                action_adapter,
                segment,
                args.encode_batch_size,
            )
            for pair, cached_sample in requested_pairs.items():
                if pair[0] != episode_id:
                    continue
                raw_sample = _window_from_encoded_segment(
                    torch,
                    encoded,
                    segment,
                    pair[1],
                    manifest["window_contract"],
                    dataset_name=dataset.dataset_name,
                )
                feature_errors = {
                    name: _tensor_error(raw_sample[name], cached_sample[name])
                    for name in feature_keys
                }
                raw_batch = collator([raw_sample])
                cached_batch = collator([cached_sample])
                flow_seed = args.seed + len(records)
                _set_rng(torch, flow_seed, device)
                raw_output = model(
                    raw_batch,
                    action_condition_mode="ground_truth",
                    route_mode="oracle",
                    flow_seed=flow_seed,
                    compute_teacher_targets=True,
                    compute_residual=args.stage != "nominal_flow_pretrain",
                )
                _set_rng(torch, flow_seed, device)
                cached_output = model(
                    cached_batch,
                    action_condition_mode="ground_truth",
                    route_mode="oracle",
                    flow_seed=flow_seed,
                    compute_teacher_targets=True,
                    compute_residual=args.stage != "nominal_flow_pretrain",
                )
                output_errors = {
                    name: _tensor_error(raw_output[name], cached_output[name])
                    for name in output_keys
                }
                raw_losses = flow_wam_skill_losses(
                    raw_output, raw_batch, cfg["loss_weights"], stage=args.stage
                )
                cached_losses = flow_wam_skill_losses(
                    cached_output,
                    cached_batch,
                    cfg["loss_weights"],
                    stage=args.stage,
                )
                loss_errors = {
                    name: abs(float(raw_losses[name]) - float(cached_losses[name]))
                    for name in raw_losses
                    if name in cached_losses
                    and getattr(raw_losses[name], "numel", lambda: 0)() == 1
                }
                records.append(
                    {
                        "episode_id": pair[0],
                        "step_id": pair[1],
                        "feature_errors": feature_errors,
                        "output_errors": output_errors,
                        "loss_abs_errors": loss_errors,
                    }
                )
                found.add(pair)

    missing = sorted(set(requested_pairs) - found)
    max_feature = max(
        (
            value["max_abs"]
            for record in records
            for value in record["feature_errors"].values()
        ),
        default=float("inf"),
    )
    max_output = max(
        (
            value["max_abs"]
            for record in records
            for value in record["output_errors"].values()
        ),
        default=float("inf"),
    )
    max_loss = max(
        (
            value
            for record in records
            for value in record["loss_abs_errors"].values()
        ),
        default=float("inf"),
    )
    report = {
        "format": "mowe_feature_store_equivalence_v1",
        "benchmark": "calvin_abc_d",
        "store": str(Path(args.store).resolve()),
        "openvla_identity_sha256": requested_identity["identity_sha256"],
        "dataset_root": str(dataset.root),
        "dataset_fingerprint": dataset.dataset_fingerprint,
        "annotation_fingerprint": dataset.annotation_fingerprint,
        "requested_samples": args.samples,
        "compared_samples": len(records),
        "missing_pairs": [list(value) for value in missing],
        "tolerances": {
            "feature_atol": args.feature_atol,
            "output_atol": args.output_atol,
            "loss_atol": args.loss_atol,
        },
        "max_feature_abs_error": max_feature,
        "max_output_abs_error": max_output,
        "max_loss_abs_error": max_loss,
        "records": records,
    }
    report["passed"] = (
        len(records) == sample_count
        and not missing
        and max_feature <= args.feature_atol
        and max_output <= args.output_atol
        and max_loss <= args.loss_atol
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
