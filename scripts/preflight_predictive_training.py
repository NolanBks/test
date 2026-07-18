#!/usr/bin/env python3
"""Run one real predictive RLDS batch through the MoWE training contract.

This is intentionally a smoke/preflight command, not a benchmark.  It checks
the exact data loader, OpenVLA feature extraction, future-label cache, router,
and loss interface before a long stage-1 or stage-2 job is launched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.losses import weighted_training_losses
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def _shape(value):
    return [int(dim) for dim in getattr(value, "shape", [])]


def _require_finite(torch, outputs: dict) -> None:
    for key, value in outputs.items():
        if not getattr(value, "is_floating_point", lambda: False)():
            continue
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Non-finite value in model output: {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mowe_wam/train_predictive_memory_router.yaml")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--transition-label-path", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--backward", action="store_true", help="Also run one optimizer step on local trainable heads.")
    args = parser.parse_args()

    torch = require_torch()
    from scripts.train_mowe_wam import (
        _autocast_context,
        _deep_update,
        _device_from_config,
        _move_training_targets,
        build_dataloader,
        build_model,
    )

    cfg = load_config(args.config)
    _deep_update(
        cfg,
        {
            "data": {
                "data_root": args.data_root,
                "dataset_name": args.dataset_name,
                "transition_label_path": args.transition_label_path,
                "limit_batches": 1,
            },
            "backbone": {"checkpoint": args.checkpoint},
            "training": {"batch_size": args.batch_size},
        },
    )
    if cfg.get("model", {}).get("variant") != "predictive_memory":
        raise SystemExit("This preflight requires model.variant='predictive_memory'.")

    device = _device_from_config(torch, cfg["training"])
    cfg["backbone"]["device"] = device
    model = build_model(cfg, torch)
    dataloader = build_dataloader(cfg, model, torch)
    try:
        raw_batch = next(iter(dataloader))
    except StopIteration as exc:
        raise SystemExit("No valid predictive windows were produced by the RLDS dataset.") from exc
    batch = _move_training_targets(raw_batch, device, torch)

    required = {
        "actions",
        "history_actions",
        "history_predicates",
        "future_predicates",
        "progress_delta",
        "future_risk",
        "future_recovery",
        "memory_state",
        "event_target",
        "phase_target",
        "previous_expert",
    }
    missing = sorted(required - set(batch))
    if missing:
        raise RuntimeError(f"Predictive batch is missing required fields: {', '.join(missing)}")

    precision = str(cfg["training"].get("precision", "bf16"))
    model.train(args.backward)
    with _autocast_context(torch, device, precision):
        outputs = model(batch, use_oracle_future=False)
        _require_finite(torch, outputs)
        losses = weighted_training_losses(outputs, batch, cfg.get("loss_weights", {}))
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError("Non-finite total loss.")

    if args.backward:
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=float(cfg["training"].get("learning_rate", 1e-4)))
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        optimizer.step()

    summary = {
        "device": device,
        "checkpoint": args.checkpoint,
        "dataset_name": args.dataset_name,
        "episode_id": raw_batch.get("episode_id", [None])[0],
        "step_id": int(raw_batch["step_id"][0]) if "step_id" in raw_batch else None,
        "batch_shapes": {key: _shape(batch[key]) for key in sorted(required)},
        "output_shapes": {
            key: _shape(outputs[key])
            for key in ("actions", "future_predicates", "progress_delta", "future_risk", "future_recovery", "router_probs")
        },
        "total_loss": float(losses["total_loss"].detach().float().cpu()),
        "backward_step": bool(args.backward),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("Predictive real-batch preflight OK")


if __name__ == "__main__":
    main()
