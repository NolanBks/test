#!/usr/bin/env python3
"""Stage-1 synthetic trainer for WorldTransitionHead and EventMemoryEncoder."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.memory import EventMemoryEncoder, EventMemoryState
from mowe_wam.models import WorldTransitionHead
from mowe_wam.predicates.schema import predicate_dim
from mowe_wam.training.losses import progress_delta_loss, risk_loss
from mowe_wam.training.train_utils import make_synthetic_predictive_batch, set_seed
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def _real_training(args, torch, cfg):
    """Train only transition/memory modules once real labels and data are ready."""

    from scripts.train_mowe_wam import (
        _autocast_context,
        _deep_update,
        _device_from_config,
        _make_grad_scaler,
        _move_training_targets,
        build_dataloader,
        build_model,
    )

    _deep_update(
        cfg,
        {
            "data": {
                "data_root": args.data_root,
                "dataset_name": args.dataset_name,
                "transition_label_path": args.transition_label_path,
                "limit_batches": args.limit_batches,
            },
            "backbone": {"checkpoint": args.checkpoint},
            "output_dir": args.output_dir,
            "resume": args.resume,
            "training": {"grad_accumulation_steps": args.grad_accumulation_steps},
        },
    )
    if cfg["data"].get("data_root") in {None, "TBD"} or cfg["data"].get("transition_label_path") in {None, "TBD"}:
        raise SystemExit("Real transition training requires --data-root and --transition-label-path.")
    if cfg["model"].get("variant") != "predictive_memory":
        raise SystemExit("WorldTransitionHead training requires model.variant='predictive_memory'.")
    device = _device_from_config(torch, cfg["training"])
    cfg["backbone"]["device"] = device
    set_seed(int(cfg.get("seed", 7)))
    model = build_model(cfg, torch)
    dataloader = build_dataloader(cfg, model, torch)
    params = list(model.world_head.parameters()) + list(model.memory_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=float(cfg["training"].get("learning_rate", 1e-4)), weight_decay=float(cfg["training"].get("weight_decay", 0.01)))
    output_dir = Path(cfg.get("output_dir", "outputs/train/world_transition"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    log_path = output_dir / "world_transition_log.jsonl"
    max_steps = int(args.max_steps or cfg["training"].get("max_steps", 1000))
    save_freq = int(args.save_freq or cfg["training"].get("save_freq", 100))
    precision = str(cfg["training"].get("precision", "bf16"))
    grad_accumulation_steps = int(cfg["training"].get("grad_accumulation_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    scaler = _make_grad_scaler(torch, device, precision)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    micro_step = 0
    if cfg.get("resume"):
        state = torch.load(Path(cfg["resume"]), map_location="cpu")
        try:
            model.world_head.load_state_dict(state["world_head"])
            model.memory_encoder.load_state_dict(state["memory_encoder"])
            optimizer.load_state_dict(state["optimizer"])
        except KeyError as exc:
            raise ValueError(f"{cfg['resume']} is not a WorldTransitionHead checkpoint.") from exc
        if state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        global_step = int(state.get("step", 0))
    with log_path.open("a", encoding="utf-8") as log_file:
        while global_step < max_steps:
            made_progress = False
            for raw_batch in dataloader:
                made_progress = True
                batch = _move_training_targets(raw_batch, device, torch)
                with _autocast_context(torch, device, precision):
                    features = model.extract_features(batch)
                    memory_context = model.memory_encoder(batch["memory_state"].to(dtype=features.dtype))
                    outputs = model.world_head(features, batch["history_actions"].to(dtype=features.dtype), batch["history_predicates"].to(dtype=features.dtype), memory_context)
                    losses = {
                        "future_predicate_loss": risk_loss(outputs["future_predicate_logits"], batch["future_predicates"]),
                        "progress_delta_loss": progress_delta_loss(outputs["progress_delta"], batch["progress_delta"]),
                        "future_risk_loss": risk_loss(outputs["future_risk_logits"], batch["future_risk"]),
                        "future_recovery_loss": risk_loss(outputs["future_recovery_logits"], batch["future_recovery"]),
                    }
                    total = sum(losses.values())
                    loss = total / grad_accumulation_steps
                scaler.scale(loss).backward()
                micro_step += 1
                if micro_step % grad_accumulation_steps != 0:
                    continue
                scaler.unscale_(optimizer)
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                row = {"step": global_step, **{key: float(value.detach().float().cpu()) for key, value in losses.items()}, "total_loss": float(total.detach().float().cpu())}
                log_file.write(json.dumps(row, sort_keys=True) + "\n")
                log_file.flush()
                print(row)
                if global_step % save_freq == 0 or global_step >= max_steps:
                    torch.save(
                        {
                            "world_head": model.world_head.state_dict(),
                            "memory_encoder": model.memory_encoder.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "scaler": scaler.state_dict(),
                            "step": global_step,
                            "config": cfg,
                        },
                        output_dir / "world_transition_latest.pt",
                    )
                if global_step >= max_steps:
                    break
            if not made_progress:
                raise SystemExit("Transition dataloader produced no batches.")
    torch.save(
        {
            "world_head": model.world_head.state_dict(),
            "memory_encoder": model.memory_encoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": global_step,
            "config": cfg,
        },
        output_dir / "world_transition_latest.pt",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mowe_wam/train_predictive_memory_router.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--transition-label-path", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    args = parser.parse_args()

    try:
        torch = require_torch()
        cfg = load_config(args.config)
        if not args.mock:
            _real_training(args, torch, cfg)
            return
        if not args.dry_run:
            raise SystemExit("Mock mode requires --dry-run.")
        model_cfg = cfg["model"]
        hidden_dim = int(model_cfg.get("hidden_dim") or 1024)
        action_dim = int(model_cfg.get("action_dim", 7))
        chunk_size = int(model_cfg.get("chunk_size", 8))
        num_experts = int(model_cfg.get("num_experts", 5))
        history_steps = int(model_cfg.get("history_steps", 4))
        memory_state = EventMemoryState(num_experts=num_experts)
        memory_encoder = EventMemoryEncoder(memory_state.vector_dim, int(model_cfg.get("memory_context_dim", 128)))
        head = WorldTransitionHead(
            hidden_dim,
            action_dim,
            predicate_dim(),
            memory_context_dim=memory_encoder.context_dim,
            temporal_dim=int(model_cfg.get("temporal_dim", 512)),
            temporal_layers=int(model_cfg.get("temporal_layers", 2)),
            temporal_heads=int(model_cfg.get("temporal_heads", 8)),
            temporal_ff_dim=int(model_cfg.get("temporal_ff_dim", 1024)),
            max_history_steps=history_steps,
        )
        set_seed(int(cfg.get("seed", 7)))
        optimizer = torch.optim.AdamW(
            list(head.parameters()) + list(memory_encoder.parameters()),
            lr=float(cfg["training"].get("learning_rate", 1e-4)),
        )
        for step in range(args.max_steps or 2):
            batch = make_synthetic_predictive_batch(
                2,
                hidden_dim,
                action_dim,
                chunk_size,
                history_steps=history_steps,
                num_experts=num_experts,
                memory_dim=memory_state.vector_dim,
            )
            memory_context = memory_encoder(batch["memory_state"])
            outputs = head(batch["features"], batch["history_actions"], batch["history_predicates"], memory_context)
            losses = {
                "future_predicate_loss": risk_loss(outputs["future_predicate_logits"], batch["future_predicates"]),
                "progress_delta_loss": progress_delta_loss(outputs["progress_delta"], batch["progress_delta"]),
                "future_risk_loss": risk_loss(outputs["future_risk_logits"], batch["future_risk"]),
                "future_recovery_loss": risk_loss(outputs["future_recovery_logits"], batch["future_recovery"]),
            }
            total = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            print({"step": step, "total_loss": float(total.detach()), **{key: float(value.detach()) for key, value in losses.items()}})
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
