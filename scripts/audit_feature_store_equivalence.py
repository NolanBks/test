#!/usr/bin/env python3
"""Compare raw RLDS/OpenVLA/DINO windows with mowe_feature_store_v1."""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.backbones import openvla_identities_match, resolve_original_openvla_identity
from mowe_wam.data import LatentWAMCollator, MoWEFeatureWindowDataset
from mowe_wam.training.flow_runtime import (
    FLOW_COMPONENTS,
    _build_flow_dataset,
    build_flow_policy,
    deep_update,
    resolve_feature_store_contract,
)
from mowe_wam.training.latent_losses import flow_wam_skill_losses
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_nominal_flow_wam.yaml")
    parser.add_argument("--store", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-revision", required=True)
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument(
        "--skill-sidecar",
        help="Path to the RLDS CoT skill sidecar; overrides data.skill_sidecar_path in the config.",
    )
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--stage", choices=["nominal_flow_pretrain", "expert_warmstart", "joint"], default="nominal_flow_pretrain")
    parser.add_argument("--feature-atol", type=float, default=0.03)
    parser.add_argument("--output-atol", type=float, default=0.10)
    parser.add_argument("--loss-atol", type=float, default=0.05)
    parser.add_argument("--output")
    return parser.parse_args()


def _tensor_error(left, right, *, mask=None):
    left = left.detach().float().cpu()
    right = right.detach().float().cpu()
    if left.shape != right.shape:
        raise ValueError(
            f"Cannot compare feature tensors with different shapes: {tuple(left.shape)} != "
            f"{tuple(right.shape)}."
        )
    if left.ndim < 1:
        raise ValueError("Feature tensors must have at least one dimension.")

    vector_shape = left.shape[:-1]
    if mask is None:
        vector_mask = left.new_ones(vector_shape, dtype=require_torch().bool)
    else:
        vector_mask = mask.detach().bool().cpu()
        while vector_mask.ndim < len(vector_shape):
            vector_mask = vector_mask.unsqueeze(-1)
        try:
            vector_mask = vector_mask.expand(vector_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Feature mask shape {tuple(mask.shape)} cannot cover {tuple(vector_shape)}."
            ) from exc

    left_vectors = left.reshape(-1, left.shape[-1])[vector_mask.reshape(-1)]
    right_vectors = right.reshape(-1, right.shape[-1])[vector_mask.reshape(-1)]
    compared_vectors = int(left_vectors.shape[0])
    total_vectors = int(vector_mask.numel())
    if not compared_vectors:
        return {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "rmse": 0.0,
            "smooth_l1": 0.0,
            "max_cosine_distance": 0.0,
            "mean_cosine_distance": 0.0,
            "compared_vectors": 0,
            "ignored_vectors": total_vectors,
        }

    signed_difference = left_vectors - right_vectors
    difference = signed_difference.abs()
    smooth_l1 = require_torch().where(
        difference < 1.0,
        0.5 * difference.square(),
        difference - 0.5,
    )
    left_norm = left_vectors.norm(dim=-1)
    right_norm = right_vectors.norm(dim=-1)
    denominator = left_norm * right_norm
    cosine = (left_vectors * right_vectors).sum(dim=-1) / denominator.clamp_min(1e-12)
    both_zero = left_norm.le(1e-12) & right_norm.le(1e-12)
    cosine = require_torch().where(both_zero, require_torch().ones_like(cosine), cosine)
    cosine_distance = (1.0 - cosine.clamp(-1.0, 1.0)).clamp_min(0.0)
    return {
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        "rmse": float(signed_difference.square().mean().sqrt()),
        "smooth_l1": float(smooth_l1.mean()),
        "max_cosine_distance": float(cosine_distance.max()),
        "mean_cosine_distance": float(cosine_distance.mean()),
        "compared_vectors": compared_vectors,
        "ignored_vectors": total_vectors - compared_vectors,
    }


def _feature_gate_error(feature_errors: dict[str, dict[str, float]]) -> float:
    """Return the largest training-relevant feature discrepancy.

    OpenVLA/language features are consumed as context, so their mean absolute
    and cosine errors are checked. DINO features are regression targets trained
    with cosine and Smooth-L1 components, so the audit uses those same metrics.
    Absolute maxima remain in the report as diagnostics but are not stable
    across BF16 batch shapes and FP16 cache serialization.
    """

    context_names = {
        "current_visual_views",
        "history_visual_views",
        "long_history_visual_views",
        "language",
    }
    target_names = {"current_dino", "future_dino"}
    values = []
    for name in context_names:
        error = feature_errors[name]
        values.extend((error["mean_abs"], error["mean_cosine_distance"]))
    for name in target_names:
        error = feature_errors[name]
        values.extend((error["smooth_l1"], error["mean_cosine_distance"]))
    return max(values, default=float("inf"))


def _output_gate_error(output_errors: dict[str, dict[str, float]]) -> float:
    """Gate continuous outputs and gripper logits; keep binary actions diagnostic."""

    continuous_names = {
        "nominal_actions",
        "future_latents",
        "route_world_tokens",
        "router_logits",
        "motion_actions",
        "gripper_logits",
    }
    return max(
        (output_errors[name]["max_abs"] for name in continuous_names),
        default=float("inf"),
    )


def _set_rng(torch, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)


def _source_episode_identity(sample):
    """Return the exact TFDS source key recorded by formal conversion, if present."""

    file_key = sample.get("source_file_key")
    trajectory_index = sample.get("source_traj_index")
    if file_key is None or trajectory_index is None:
        return None
    return (str(file_key), int(trajectory_index))


def main() -> None:
    args = parse_args()
    if args.samples < 1:
        raise SystemExit("--samples must be positive.")
    torch = require_torch()
    online_cfg = load_config(args.config)
    deep_update(
        online_cfg,
        {
            "backbone": {
                "mode": "online_openvla",
                "feature_source": "pre_action_context",
                "checkpoint": args.checkpoint,
                "freeze_backbone": True,
            },
            "teacher": {"checkpoint": args.teacher_checkpoint, "cache_path": None},
            "data": {
                "backend": "rlds",
                "data_root": args.data_root,
                **({"skill_sidecar_path": args.skill_sidecar} if args.skill_sidecar else {}),
                "image_aug": False,
                "window_shuffle_buffer_size": 0,
                "num_workers": 0,
                "pin_memory": False,
            },
            "training": {"batch_size": 1},
            "validation": {"enabled": False},
        },
    )
    cached_cfg = copy.deepcopy(online_cfg)
    deep_update(
        cached_cfg,
        {
            "backbone": {
                "mode": "precomputed_features",
                "feature_source": "pre_action_context_cache",
            },
            "data": {
                "backend": "mowe_feature_store_v1",
                "feature_store_path": args.store,
                "max_open_feature_shards": 2,
            },
        },
    )
    manifest = resolve_feature_store_contract(cached_cfg)
    source = manifest.get("source_contract", {})
    requested_identity = resolve_original_openvla_identity(
        args.checkpoint,
        revision=args.backbone_revision,
        repo_id=source.get("openvla_identity", {}).get("repo_id", "openvla/openvla-7b"),
    )
    if not openvla_identities_match(source.get("openvla_identity"), requested_identity):
        raise RuntimeError("Raw equivalence checkpoint differs from the feature-store OpenVLA identity.")
    online_cfg["backbone"].update(
        {
            "identity": requested_identity,
            "repo_id": requested_identity["repo_id"],
            "revision": requested_identity["revision"],
        }
    )
    if source.get("image_aug") is not False:
        raise RuntimeError("Equivalence requires a store built with image_aug=false.")

    online_model = build_flow_policy(online_cfg, include_teacher=True)
    cached_model = build_flow_policy(cached_cfg, include_teacher=False)
    if int(online_model.backbone.hidden_dim) != int(cached_model.backbone.hidden_dim):
        raise RuntimeError("Online and cached OpenVLA hidden dimensions differ.")
    for name in FLOW_COMPONENTS:
        getattr(cached_model, name).load_state_dict(getattr(online_model, name).state_dict())
    online_model.eval()
    cached_model.eval()

    cached_dataset = MoWEFeatureWindowDataset(args.store, partition="train")
    sample_count = min(args.samples, len(cached_dataset))
    selected_indices = random.Random(args.seed).sample(range(len(cached_dataset)), sample_count)
    cached_samples = {index: cached_dataset[index] for index in selected_indices}
    source_to_indices = defaultdict(list)
    pair_to_indices = defaultdict(list)
    for index, sample in cached_samples.items():
        source_identity = _source_episode_identity(sample)
        if source_identity is not None:
            source_to_indices[(source_identity, int(sample["step_id"]))].append(index)
        pair_to_indices[(sample["episode_id"], int(sample["step_id"]))].append(index)
    raw_dataset = _build_flow_dataset(
        online_cfg,
        online_model,
        episode_partition_name="train",
        limit=None,
        window_shuffle_buffer_size=0,
        windows_per_episode=None,
        distributed_rank=0,
        distributed_world_size=1,
    )
    collator = LatentWAMCollator()
    records = []
    found = set()
    source_matches = 0
    episode_id_matches = 0
    device = str(online_cfg["training"].get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    context_keys = {
        "pixel_values_primary",
        "pixel_values_wrist",
        "history_pixel_values_primary",
        "history_pixel_values_wrist",
        "long_history_pixel_values_primary",
        "long_history_pixel_values_wrist",
        "language",
    }
    cached_context_keys = {
        "current_visual_views",
        "history_visual_views",
        "long_history_visual_views",
        "precomputed_language",
    }
    with torch.no_grad():
        for raw_sample in raw_dataset:
            pair = (raw_sample["episode_id"], int(raw_sample["step_id"]))
            source_identity = _source_episode_identity(raw_sample)
            candidates = source_to_indices.get(
                (source_identity, int(raw_sample["step_id"])), []
            )
            matching_mode = "source_identity"
            if len(candidates) != 1:
                candidates = pair_to_indices.get(pair, [])
                matching_mode = "episode_id"
            if len(candidates) != 1:
                continue
            cached_index = candidates[0]
            if cached_index in found:
                continue
            cached_sample = cached_samples[cached_index]
            raw_batch = collator([raw_sample])
            cached_batch = collator([cached_sample])
            raw_context = online_model.backbone.extract_context_features(
                {key: raw_batch[key] for key in context_keys}
            )
            cached_context = cached_model.backbone.extract_context_features(
                {key: cached_batch[key] for key in cached_context_keys}
            )
            raw_history_mask = raw_batch["history_mask"][..., :-1].bool()
            cached_history_mask = cached_batch["history_mask"][..., :-1].bool()
            raw_long_mask = raw_batch["long_history_mask"].bool()
            cached_long_mask = cached_batch["long_history_mask"].bool()
            mask_checks = {
                "history": {
                    "matches": bool(torch.equal(raw_history_mask, cached_history_mask)),
                    "valid_positions": int((raw_history_mask & cached_history_mask).sum()),
                    "total_positions": int(raw_history_mask.numel()),
                },
                "long_history": {
                    "matches": bool(torch.equal(raw_long_mask, cached_long_mask)),
                    "valid_positions": int((raw_long_mask & cached_long_mask).sum()),
                    "total_positions": int(raw_long_mask.numel()),
                },
            }
            history_mask = raw_history_mask & cached_history_mask
            long_mask = raw_long_mask & cached_long_mask
            feature_errors = {
                "current_visual_views": _tensor_error(
                    raw_context["current_visual_views"],
                    cached_context["current_visual_views"],
                ),
                "history_visual_views": _tensor_error(
                    raw_context["history_visual_views"],
                    cached_context["history_visual_views"],
                    mask=history_mask,
                ),
                "long_history_visual_views": _tensor_error(
                    raw_context["long_history_visual_views"],
                    cached_context["long_history_visual_views"],
                    mask=long_mask,
                ),
                "language": _tensor_error(
                    raw_context["language"], cached_context["language"]
                ),
            }
            raw_current_target, raw_future_targets = online_model._teacher_targets(
                raw_batch, torch.device(device)
            )
            feature_errors["current_dino"] = _tensor_error(
                raw_current_target, cached_batch["current_latent_target"]
            )
            feature_errors["future_dino"] = _tensor_error(
                raw_future_targets, cached_batch["future_latent_targets"]
            )
            flow_seed = args.seed + len(records)
            _set_rng(torch, flow_seed, device)
            raw_output = online_model(
                raw_batch,
                action_condition_mode="ground_truth",
                route_mode="oracle",
                flow_seed=flow_seed,
                compute_teacher_targets=True,
                compute_residual=args.stage != "nominal_flow_pretrain",
            )
            _set_rng(torch, flow_seed, device)
            cached_output = cached_model(
                cached_batch,
                action_condition_mode="ground_truth",
                route_mode="oracle",
                flow_seed=flow_seed,
                compute_teacher_targets=True,
                compute_residual=args.stage != "nominal_flow_pretrain",
            )
            output_errors = {
                name: _tensor_error(raw_output[name], cached_output[name])
                for name in (
                    "nominal_actions",
                    "future_latents",
                    "route_world_tokens",
                    "router_logits",
                    "motion_actions",
                    "actions",
                )
            }
            output_errors["gripper_logits"] = _tensor_error(
                raw_output["gripper_logits"], cached_output["gripper_logits"]
            )
            gripper_action_disagreements = int(
                raw_output["actions"][..., 6:7]
                .ne(cached_output["actions"][..., 6:7])
                .sum()
            )
            raw_losses = flow_wam_skill_losses(
                raw_output, raw_batch, online_cfg["loss_weights"], stage=args.stage
            )
            cached_losses = flow_wam_skill_losses(
                cached_output, cached_batch, cached_cfg["loss_weights"], stage=args.stage
            )
            loss_errors = {
                name: abs(float(raw_losses[name]) - float(cached_losses[name]))
                for name in raw_losses
                if name in cached_losses and getattr(raw_losses[name], "numel", lambda: 0)() == 1
            }
            records.append(
                {
                    "episode_id": pair[0],
                    "step_id": pair[1],
                    "mask_checks": mask_checks,
                    "feature_errors": feature_errors,
                    "feature_gate_error": _feature_gate_error(feature_errors),
                    "output_errors": output_errors,
                    "output_gate_error": _output_gate_error(output_errors),
                    "gripper_action_disagreements": gripper_action_disagreements,
                    "loss_abs_errors": loss_errors,
                    "loss_gate_error": max(
                        (
                            value
                            for name, value in loss_errors.items()
                            if name != "gripper_accuracy"
                        ),
                        default=float("inf"),
                    ),
                }
            )
            found.add(cached_index)
            if matching_mode == "source_identity":
                source_matches += 1
            else:
                episode_id_matches += 1
            if len(found) == sample_count:
                break
    missing_indices = sorted(set(cached_samples) - found)
    missing = [
        (
            cached_samples[index]["episode_id"],
            int(cached_samples[index]["step_id"]),
        )
        for index in missing_indices
    ]
    missing_source_identities = [
        _source_episode_identity(cached_samples[index]) for index in missing_indices
    ]
    max_feature = max(
        (value["max_abs"] for record in records for value in record["feature_errors"].values()),
        default=float("inf"),
    )
    max_feature_gate = max(
        (record["feature_gate_error"] for record in records),
        default=float("inf"),
    )
    masks_match = all(
        check["matches"]
        for record in records
        for check in record["mask_checks"].values()
    )
    max_output = max(
        (value["max_abs"] for record in records for value in record["output_errors"].values()),
        default=float("inf"),
    )
    max_output_gate = max(
        (record["output_gate_error"] for record in records),
        default=float("inf"),
    )
    max_loss = max(
        (value for record in records for value in record["loss_abs_errors"].values()),
        default=float("inf"),
    )
    max_loss_gate = max(
        (record["loss_gate_error"] for record in records),
        default=float("inf"),
    )
    total_gripper_action_disagreements = sum(
        record["gripper_action_disagreements"] for record in records
    )
    report = {
        "format": "mowe_feature_store_equivalence_v1",
        "benchmark": "libero",
        "store": str(Path(args.store).resolve()),
        "openvla_identity_sha256": requested_identity["identity_sha256"],
        "requested_samples": args.samples,
        "compared_samples": len(records),
        "missing_pairs": [list(value) for value in missing],
        "missing_source_episode_identities": [
            list(value) if value is not None else None for value in missing_source_identities
        ],
        "matching": {
            "source_identity_matches": source_matches,
            "episode_id_fallback_matches": episode_id_matches,
            "source_identity_available_for_selected": sum(
                _source_episode_identity(sample) is not None
                for sample in cached_samples.values()
            ),
        },
        "tolerances": {
            "feature_atol": args.feature_atol,
            "output_atol": args.output_atol,
            "loss_atol": args.loss_atol,
        },
        "comparison_contract": {
            "name": "mask_aware_training_metric_v1",
            "history_padding": "excluded_using_matching_raw_and_cached_masks",
            "context_gate_metrics": ["mean_abs", "mean_cosine_distance"],
            "dino_gate_metrics": ["smooth_l1", "mean_cosine_distance"],
            "max_abs_role": "diagnostic_only",
            "output_gate": "continuous_outputs_and_gripper_logits",
            "loss_gate": "all_scalar_losses_except_gripper_accuracy",
            "binary_action_role": "diagnostic_only",
        },
        "masks_match": masks_match,
        "max_feature_abs_error": max_feature,
        "max_feature_gate_error": max_feature_gate,
        "max_output_abs_error": max_output,
        "max_output_gate_error": max_output_gate,
        "max_loss_abs_error": max_loss,
        "max_loss_gate_error": max_loss_gate,
        "gripper_action_disagreements": total_gripper_action_disagreements,
        "records": records,
    }
    report["passed"] = (
        len(records) == sample_count
        and not missing
        and masks_match
        and max_feature_gate <= args.feature_atol
        and max_output_gate <= args.output_atol
        and max_loss_gate <= args.loss_atol
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
