#!/usr/bin/env python3
"""Single-node MoWE-WAM training entrypoint for OpenVLA-OFT + LIBERO RLDS."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.backbones import OpenVLAOFTAdapter
from mowe_wam.data import LiberoPredicateDataset, MoWEPaddedCollator
from mowe_wam.memory import EVENT_TYPES, EventMemoryEncoder, EventMemoryState
from mowe_wam.models import (
    ExpertRouter,
    MoEActionExperts,
    MoWEPolicyWrapper,
    PredictiveExpertRouter,
    WorldPredicateHead,
    WorldTransitionHead,
)
from mowe_wam.predicates.schema import predicate_dim
from mowe_wam.training.losses import weighted_training_losses
from mowe_wam.training.train_utils import set_seed
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        elif value is not None:
            base[key] = value
    return base


def _device_from_config(torch, cfg: dict[str, Any]):
    requested = cfg.get("device", "auto")
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _autocast_context(torch, device: str, precision: str):
    if not str(device).startswith("cuda") or precision == "float32":
        return nullcontext()
    dtype = torch.bfloat16 if precision in {"bf16", "bfloat16"} else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def _make_grad_scaler(torch, device: str, precision: str):
    """Use dynamic loss scaling only for CUDA fp16 training."""

    enabled = str(device).startswith("cuda") and str(precision).lower() in {"fp16", "float16", "half"}
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # Compatibility with older OpenVLA environments.
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _move_training_targets(batch: dict[str, Any], device: str, torch) -> dict[str, Any]:
    out = dict(batch)
    float_keys = (
        "actions",
        "predicates",
        "progress",
        "risk",
        "history_actions",
        "history_predicates",
        "current_predicates",
        "future_predicates",
        "progress_delta",
        "future_risk",
        "future_recovery",
        "memory_state",
    )
    for key in float_keys:
        if key in batch:
            out[key] = batch[key].to(device=device, dtype=torch.float32)
    for key in ("event_target", "phase_target", "previous_expert"):
        if key in batch:
            out[key] = batch[key].to(device=device, dtype=torch.long)
    return out


def _save_checkpoint(path: Path, model, optimizer, scheduler, scaler, step: int, cfg: dict[str, Any], torch) -> None:
    state = {
        "step": step,
        "config": cfg,
        "world_head": model.world_head.state_dict(),
        "router": model.router.state_dict(),
        "experts": model.experts.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }
    if getattr(model, "memory_encoder", None) is not None:
        state["memory_encoder"] = model.memory_encoder.state_dict()
    state["predictive"] = bool(getattr(model, "predictive", False))
    trainable_backbone = hasattr(model.backbone, "model") and any(param.requires_grad for param in model.backbone.model.parameters())
    if trainable_backbone:
        state["backbone"] = model.backbone.model.state_dict()
    proprio_projector = getattr(model.backbone, "proprio_projector", None)
    if proprio_projector is not None and any(param.requires_grad for param in proprio_projector.parameters()):
        state["proprio_projector"] = proprio_projector.state_dict()
    torch.save(state, path)


def _load_checkpoint(path: Path, model, optimizer, scheduler, scaler, torch) -> int:
    state = torch.load(path, map_location="cpu")
    saved_predictive = bool(state.get("predictive", "memory_encoder" in state))
    if saved_predictive != bool(getattr(model, "predictive", False)):
        raise ValueError(
            f"Checkpoint predictive={saved_predictive} does not match current "
            f"model.predictive={bool(getattr(model, 'predictive', False))}."
        )
    model.world_head.load_state_dict(state["world_head"])
    model.router.load_state_dict(state["router"])
    model.experts.load_state_dict(state["experts"])
    if state.get("memory_encoder") is not None and getattr(model, "memory_encoder", None) is not None:
        model.memory_encoder.load_state_dict(state["memory_encoder"])
    if "backbone" in state and hasattr(model.backbone, "model"):
        model.backbone.model.load_state_dict(state["backbone"])
    if state.get("proprio_projector") is not None and getattr(model.backbone, "proprio_projector", None) is not None:
        model.backbone.proprio_projector.load_state_dict(state["proprio_projector"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return int(state.get("step", 0))


def _load_transition_initialization(path: Path, model, torch) -> None:
    """Initialize predictive modules from a stage-1 transition checkpoint."""

    if not getattr(model, "predictive", False) or getattr(model, "memory_encoder", None) is None:
        raise ValueError("--init-transition requires model.variant='predictive_memory'.")
    state = torch.load(path, map_location="cpu")
    model.world_head.load_state_dict(state["world_head"])
    model.memory_encoder.load_state_dict(state["memory_encoder"])


def build_model(cfg: dict[str, Any], torch):
    backbone_cfg = cfg["backbone"]
    model_cfg = cfg["model"]
    backbone = OpenVLAOFTAdapter(**backbone_cfg)
    hidden_dim = int(model_cfg.get("hidden_dim") or backbone.hidden_dim)
    action_dim = int(model_cfg.get("action_dim", 7))
    chunk_size = int(model_cfg.get("chunk_size", 8))
    num_experts = int(model_cfg.get("num_experts", 5))
    top_k = int(model_cfg.get("top_k", 2))
    variant = str(model_cfg.get("variant", "baseline"))
    experts = MoEActionExperts(hidden_dim, action_dim, chunk_size, num_experts)
    if variant == "predictive_memory":
        memory_state = EventMemoryState(num_experts=num_experts)
        memory_dim = int(model_cfg.get("memory_dim") or memory_state.vector_dim)
        if memory_dim != memory_state.vector_dim:
            raise ValueError(
                f"memory_dim={memory_dim} does not match EventMemoryState vector dim "
                f"{memory_state.vector_dim} for num_experts={num_experts}."
            )
        memory_context_dim = int(model_cfg.get("memory_context_dim", 128))
        memory_encoder = EventMemoryEncoder(memory_dim, memory_context_dim)
        transition_head = WorldTransitionHead(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            predicate_dim=predicate_dim(),
            memory_context_dim=memory_context_dim,
            temporal_dim=int(model_cfg.get("temporal_dim", 512)),
            temporal_layers=int(model_cfg.get("temporal_layers", 2)),
            temporal_heads=int(model_cfg.get("temporal_heads", 8)),
            temporal_ff_dim=int(model_cfg.get("temporal_ff_dim", 1024)),
            max_history_steps=int(model_cfg.get("history_steps", 4)),
        )
        router = PredictiveExpertRouter(
            hidden_dim=hidden_dim,
            predicate_dim=predicate_dim(),
            memory_context_dim=memory_context_dim,
            num_experts=num_experts,
            top_k=top_k,
            state_dim=int(model_cfg.get("router_state_dim", 256)),
            transition_dim=int(model_cfg.get("router_transition_dim", 128)),
            previous_expert_dim=int(model_cfg.get("previous_expert_dim", 32)),
            switch_penalty=float(model_cfg.get("switch_penalty", 0.05)),
        )
        model = MoWEPolicyWrapper(backbone, transition_head, router, experts, memory_encoder=memory_encoder, predictive=True)
    else:
        model = MoWEPolicyWrapper(
            backbone,
            WorldPredicateHead(hidden_dim, predicate_dim(), model_cfg.get("hidden_layers", [512])),
            ExpertRouter(hidden_dim, predicate_dim(), num_experts, top_k),
            experts,
        )
    return model.to(backbone.device)


def build_dataloader(cfg: dict[str, Any], model, torch):
    from torch.utils.data import DataLoader

    data_cfg = cfg["data"]
    num_workers = int(data_cfg.get("num_workers", 0))
    if num_workers != 0:
        raise ValueError(
            "MoWE's RLDS IterableDataset is not worker-sharded. Keep data.num_workers=0 "
            "to avoid duplicated trajectories and inconsistent temporal memory windows."
        )
    dataset = LiberoPredicateDataset(
        dataset_root=data_cfg["data_root"],
        split=data_cfg.get("split", "train"),
        limit=data_cfg.get("limit_batches"),
        cfg={
            "dataset_name": data_cfg["dataset_name"],
            "num_images_in_input": cfg["backbone"].get("num_images_in_input", 1),
            "use_proprio": cfg["backbone"].get("use_proprio", False),
            "predictive": cfg["model"].get("variant") == "predictive_memory",
            "history_steps": cfg["model"].get("history_steps", 4),
            "prediction_horizon": cfg["model"].get("prediction_horizon", cfg["model"].get("chunk_size", 8)),
            "action_dim": cfg["model"].get("action_dim", 7),
            "chunk_size": cfg["model"].get("chunk_size", 8),
            "transition_label_path": data_cfg.get("transition_label_path"),
            "strict_episode_id": bool(data_cfg.get("strict_episode_id", True)),
        },
        processor=model.backbone.processor,
        resize_resolution=model.backbone.resize_resolution,
        openvla_root=cfg["backbone"].get("openvla_root", "external/openvla-oft"),
        image_aug=bool(data_cfg.get("image_aug", True)),
        shuffle_buffer_size=int(data_cfg.get("shuffle_buffer_size", 100_000)),
    )
    return DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        collate_fn=MoWEPaddedCollator(model.backbone.processor),
        num_workers=0,
        pin_memory=bool(data_cfg.get("pin_memory", True)) and str(model.backbone.device).startswith("cuda"),
    )


def _routing_trace(outputs: dict[str, Any], batch: dict[str, Any]) -> dict[str, Any]:
    """Serialize a small routing/memory diagnostic without checkpoint-sized tensors."""

    topk = outputs["topk_experts"].detach().cpu()
    logits = outputs["router_logits"].detach().float().cpu()
    probs = outputs["router_probs"].detach().float().cpu()
    trace: dict[str, Any] = {
        "router_top1": int(topk[0, 0]),
        "topk_experts": [int(value) for value in topk[0].tolist()],
        "router_top1_batch": [int(value) for value in topk[:, 0].tolist()],
        "router_logits": logits[0].tolist(),
        "router_probs": probs[0].tolist(),
    }
    if "event_target" in batch:
        events = batch["event_target"].detach().cpu().view(-1).tolist()
        names = [EVENT_TYPES[int(value)] if 0 <= int(value) < len(EVENT_TYPES) else "invalid" for value in events]
        trace["memory_event"] = names[0]
        trace["memory_events_batch"] = names
    if "previous_expert" in batch:
        trace["previous_expert"] = int(batch["previous_expert"].detach().cpu().view(-1)[0])
    return trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mowe_wam/train_mowe_wam_libero.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--log-freq", type=int, default=None)
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-transition", default=None)
    parser.add_argument("--transition-label-path", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(
        "WARNING: scripts/train_mowe_wam.py is the legacy predicate/transition-label trainer. "
        "Use scripts/train_predictive_residual_moe.py for the current latent-WAM main path.",
        file=sys.stderr,
    )
    torch = require_torch()
    cfg = load_config(args.config)
    overrides = {
        "data": {
            "data_root": args.data_root,
            "dataset_name": args.dataset_name,
            "limit_batches": args.limit_batches,
            "transition_label_path": args.transition_label_path,
        },
        "backbone": {"checkpoint": args.checkpoint},
        "training": {
            "max_steps": args.max_steps,
            "grad_accumulation_steps": args.grad_accumulation_steps,
            "save_freq": args.save_freq,
            "log_freq": args.log_freq,
        },
        "output_dir": args.output_dir,
        "resume": args.resume,
        "init_transition": args.init_transition,
    }
    _deep_update(cfg, overrides)

    if cfg["data"].get("data_root") in {None, "TBD"}:
        raise SystemExit("Real training requires --data-root pointing to modified_libero_rlds.")
    if cfg.get("model", {}).get("variant") == "predictive_memory" and cfg["data"].get("transition_label_path") in {
        None,
        "TBD",
    }:
        raise SystemExit(
            "Predictive training requires --transition-label-path with a trajectory-level future/event label cache."
        )

    output_dir = Path(cfg.get("output_dir", "outputs/train/mowe_wam_libero"))
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(cfg.get("seed", 7)))
    device = _device_from_config(torch, cfg["training"])
    cfg["backbone"]["device"] = device
    (output_dir / "config_resolved.json").write_text(json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8")
    model = build_model(cfg, torch)
    dataloader = build_dataloader(cfg, model, torch)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise SystemExit("No trainable parameters found. Check freeze_backbone and head configuration.")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["training"].get("learning_rate", 1e-4)),
        weight_decay=float(cfg["training"].get("weight_decay", 0.0)),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(cfg["training"].get("lr_decay_steps", [])),
        gamma=float(cfg["training"].get("lr_decay_gamma", 0.1)),
    )
    precision = str(cfg["training"].get("precision", "bf16"))
    scaler = _make_grad_scaler(torch, device, precision)

    start_step = 0
    if cfg.get("init_transition"):
        _load_transition_initialization(Path(cfg["init_transition"]), model, torch)
    if cfg.get("resume"):
        if cfg.get("init_transition"):
            raise SystemExit("Use either --resume or --init-transition, not both.")
        start_step = _load_checkpoint(Path(cfg["resume"]), model, optimizer, scheduler, scaler, torch)

    max_steps = int(cfg["training"].get("max_steps", 1))
    grad_accumulation_steps = int(cfg["training"].get("grad_accumulation_steps", 1))
    save_freq = int(cfg["training"].get("save_freq", 1000))
    log_freq = int(cfg["training"].get("log_freq", 10))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    use_oracle_predicates = bool(cfg["model"].get("use_oracle_predicates", False))
    use_oracle_future = bool(cfg["model"].get("use_oracle_future", False))
    oracle_future_warmup_steps = int(cfg["model"].get("oracle_future_warmup_steps", 0))
    loss_weights = cfg.get("loss_weights", {})
    log_path = output_dir / "train_log.jsonl"

    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = start_step
    micro_step = 0
    with log_path.open("a", encoding="utf-8") as log_file:
        while global_step < max_steps:
            made_progress = False
            for batch_idx, raw_batch in enumerate(dataloader):
                limit_batches = cfg["data"].get("limit_batches")
                if limit_batches is not None and batch_idx >= int(limit_batches):
                    break
                made_progress = True
                batch = _move_training_targets(raw_batch, device, torch)
                with _autocast_context(torch, device, precision):
                    outputs = model(
                        batch,
                        use_oracle_predicates=use_oracle_predicates,
                        use_oracle_future=use_oracle_future or global_step < oracle_future_warmup_steps,
                    )
                    losses = weighted_training_losses(outputs, batch, loss_weights)
                    loss = losses["total_loss"] / grad_accumulation_steps
                scaler.scale(loss).backward()
                micro_step += 1
                if micro_step % grad_accumulation_steps != 0:
                    continue

                scaler.unscale_(optimizer)
                if max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % log_freq == 0 or global_step == 1:
                    row = {
                        "step": global_step,
                        "lr": optimizer.param_groups[0]["lr"],
                        **{name: float(value.detach().float().cpu()) for name, value in losses.items()},
                        **_routing_trace(outputs, batch),
                    }
                    log_file.write(json.dumps(row, sort_keys=True) + "\n")
                    log_file.flush()
                    print(row)

                if global_step % save_freq == 0 or global_step >= max_steps:
                    _save_checkpoint(
                        output_dir / "checkpoint_latest.pt", model, optimizer, scheduler, scaler, global_step, cfg, torch
                    )

                if global_step >= max_steps:
                    break
            if not made_progress:
                raise SystemExit("Dataloader produced no batches; check data_root, dataset_name, and limit_batches.")


if __name__ == "__main__":
    main()
