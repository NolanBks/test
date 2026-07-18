#!/usr/bin/env python3
"""Check latent-WAM training invariants; this is not benchmark evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.latent_runtime import (
    build_latent_policy,
    build_sequence_dataloader,
    configure_stage,
    deep_update,
)
from mowe_wam.training.latent_losses import latent_wam_training_losses
from mowe_wam.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_predictive_residual_moe.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-cache")
    parser.add_argument("--stage", choices=("pretrain", "joint"), default="joint")
    parser.add_argument("--backward", action="store_true")
    return parser.parse_args()


def static_checks(cfg):
    checks = []
    root = Path(cfg["data"]["data_root"])
    checks.append({"risk": "R12", "check": "dataset_root", "ok": root.exists(), "value": str(root)})
    for name in cfg["data"]["dataset_names"]:
        checks.append(
            {"risk": "R12", "check": f"suite:{name}", "ok": (root / name).exists(), "value": str(root / name)}
        )
    horizons = list(cfg["data"]["future_horizons"])
    checks.append({"risk": "R3", "check": "future_horizons", "ok": horizons == [1, 4, 8], "value": horizons})
    serialized = json.dumps(cfg, sort_keys=True)
    checks.append(
        {
            "risk": "R15",
            "check": "no_transition_or_predicate_labels",
            "ok": "transition_label_path" not in serialized and "predicate_label" not in serialized,
            "value": cfg.get("model", {}).get("variant"),
        }
    )
    checks.append(
        {
            "risk": "R10",
            "check": "spatial_checkpoint_claim_boundary",
            "ok": True,
            "value": "pilot only; do not claim a multi-suite backbone result",
        }
    )
    checks.append(
        {
            "risk": "R9",
            "check": "joint_action_statistics",
            "ok": bool(cfg["data"].get("joint_action_normalization", False)),
            "value": "runtime reconstructs physical actions and applies four-suite q01/q99 envelope",
        }
    )
    return checks


def backward_checks(cfg, stage):
    import torch

    model = build_latent_policy(cfg, include_teacher=not bool(cfg["teacher"].get("cache_path")))
    configure_stage(model, stage)
    loader = build_sequence_dataloader(cfg, model)
    batch = next(iter(loader))
    if "labels" in batch or "input_ids" in batch:
        raise RuntimeError("R4: latent batch contains action-token labels/input_ids.")
    model.residual_gate_threshold = 0.0
    outputs = model(
        batch,
        action_condition_mode="ground_truth",
        router_hard_topk=False,
        compute_teacher_targets=True,
    )
    losses = latent_wam_training_losses(outputs, batch, cfg["loss_weights"], stage=stage)
    losses["total_loss"].backward()
    report = [
        {"risk": "R3", "check": "future_shape", "ok": list(outputs["predicted_future_latents"].shape[1:]) == [3, 16, 384], "value": list(outputs["predicted_future_latents"].shape)},
        {"risk": "R1", "check": "action_shape", "ok": list(outputs["final_actions"].shape[1:]) == [8, 7], "value": list(outputs["final_actions"].shape)},
        {"risk": "R7", "check": "finite_loss", "ok": bool(torch.isfinite(losses["total_loss"])), "value": float(losses["total_loss"].detach())},
        {"risk": "R8", "check": "teacher_frozen", "ok": all(not parameter.requires_grad for parameter in model.visual_teacher.parameters()) if model.visual_teacher is not None else True, "value": "online" if model.visual_teacher is not None else "cache"},
    ]
    if stage == "joint":
        coverage = [
            any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in expert.parameters())
            for expert in model.residual_experts.experts
        ]
        report.append({"risk": "R17", "check": "soft_expert_gradient_coverage", "ok": all(coverage), "value": coverage})
    if torch.cuda.is_available():
        report.append({"risk": "R11", "check": "peak_cuda_bytes", "ok": True, "value": int(torch.cuda.max_memory_allocated())})
    return report


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    deep_update(
        cfg,
        {
            "data": {"data_root": args.data_root, "limit_batches": 1},
            "backbone": {"checkpoint": args.checkpoint},
            "teacher": {"checkpoint": args.teacher_checkpoint, "cache_path": args.teacher_cache},
            "training": {"batch_size": 1, "grad_accumulation_steps": 1},
        },
    )
    checks = static_checks(cfg)
    if args.backward:
        checks.extend(backward_checks(cfg, args.stage))
    print(json.dumps({"kind": "training_preflight_not_benchmark", "checks": checks}, indent=2))
    failures = [item for item in checks if not item["ok"]]
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
