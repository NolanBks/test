"""Shared single-GPU runtime for latent-WAM pretraining and joint training."""

from __future__ import annotations

import json
import random
import hashlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from mowe_wam.backbones import OpenVLAOFTAdapter, VisualTargetEncoder
from mowe_wam.data import LatentWAMCollator, LiberoSequenceDataset
from mowe_wam.memory import MultiScaleMemoryEncoder
from mowe_wam.models import (
    LegacyFutureGroundedRouter,
    LatentWAMPolicy,
    LegacyLatentWorldModel,
    RegressionNominalActionHead,
    ResidualActionExperts,
)
from mowe_wam.training.latent_losses import latent_wam_training_losses
from mowe_wam.training.schedules import router_schedule, teacher_forcing_probability
from mowe_wam.utils.optional import require_torch


TRAINABLE_COMPONENTS = (
    "memory_encoder",
    "nominal_action_head",
    "world_model",
    "router",
    "residual_experts",
    "expert_context",
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_seed(seed: int) -> None:
    torch = require_torch()
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: dict[str, Any]) -> str:
    torch = require_torch()
    requested = str(cfg.get("training", {}).get("device", "auto"))
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def autocast_context(device: str, precision: str):
    torch = require_torch()
    normalized = precision.lower()
    if not device.startswith("cuda") or normalized in {"fp32", "float32"}:
        return nullcontext()
    dtype = torch.bfloat16 if normalized in {"bf16", "bfloat16"} else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(device: str, precision: str):
    torch = require_torch()
    enabled = device.startswith("cuda") and precision.lower() in {"fp16", "float16"}
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def build_latent_policy(cfg: dict[str, Any], include_teacher: bool = True):
    require_torch()
    device = resolve_device(cfg)
    backbone_cfg = dict(cfg["backbone"])
    backbone_cfg["device"] = device
    backbone = OpenVLAOFTAdapter(**backbone_cfg)
    context_dim = int(backbone.hidden_dim)
    data_cfg = cfg["data"]
    memory_cfg = cfg["memory"]
    world_cfg = cfg["world_model"]
    router_cfg = cfg["router"]
    expert_cfg = cfg["experts"]
    action_dim = int(data_cfg.get("action_dim", 7))
    chunk_size = int(data_cfg.get("action_chunk_size", 8))
    hidden_dim = int(world_cfg.get("hidden_dim", 512))

    teacher = None
    if include_teacher:
        teacher_cfg = cfg["teacher"]
        teacher = VisualTargetEncoder(
            checkpoint=teacher_cfg.get("checkpoint", "facebook/dinov2-small"),
            spatial_grid=int(teacher_cfg.get("spatial_grid", 4)),
            target_dim=int(teacher_cfg.get("target_dim", 384)),
            num_spatial_tokens=int(teacher_cfg.get("spatial_tokens", 16)),
            device=device,
            dtype=teacher_cfg.get("dtype", cfg["training"].get("precision", "bf16")),
        )
    target_tokens = int(cfg["teacher"].get("spatial_tokens", 16))
    target_dim = int(cfg["teacher"].get("target_dim", 384))
    if teacher is not None and (teacher.spatial_tokens != target_tokens or teacher.target_dim != target_dim):
        raise ValueError(
            "Teacher output does not match config: "
            f"got [{teacher.spatial_tokens}, {teacher.target_dim}], expected [{target_tokens}, {target_dim}]."
        )

    memory = MultiScaleMemoryEncoder(
        visual_dim=context_dim,
        language_dim=context_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        max_short_tokens=int(data_cfg.get("history_length", 8)),
        max_long_tokens=int(data_cfg.get("long_memory_slots", 4)),
        heads=int(memory_cfg.get("heads", 8)),
        dropout=float(memory_cfg.get("dropout", 0.0)),
    )
    nominal = RegressionNominalActionHead(
        context_dim=context_dim,
        memory_dim=hidden_dim,
        hidden_dim=int(cfg["nominal_action"].get("hidden_dim", 512)),
        action_dim=action_dim,
        chunk_size=chunk_size,
    )
    world = LegacyLatentWorldModel(
        context_dim=context_dim,
        action_dim=action_dim,
        action_chunk_size=chunk_size,
        future_horizons=data_cfg.get("future_horizons", [1, 4, 8]),
        hidden_dim=hidden_dim,
        layers=int(world_cfg.get("layers", 6)),
        heads=int(world_cfg.get("heads", 8)),
        mlp_ratio=int(world_cfg.get("mlp_ratio", 4)),
        target_tokens=target_tokens,
        target_dim=target_dim,
        dropout=float(world_cfg.get("dropout", 0.0)),
        predict_uncertainty=bool(world_cfg.get("predict_uncertainty", False)),
    )
    router = LegacyFutureGroundedRouter(
        world_dim=hidden_dim,
        memory_dim=hidden_dim,
        latent_dim=target_dim,
        hidden_dim=int(router_cfg.get("hidden_dim", 512)),
        num_experts=int(expert_cfg.get("num_experts", 5)),
        top_k=int(expert_cfg.get("top_k", 2)),
        use_uncertainty=bool(world_cfg.get("predict_uncertainty", False)),
    )
    experts = ResidualActionExperts(
        hidden_dim=int(expert_cfg.get("hidden_dim", 512)),
        action_dim=action_dim,
        chunk_size=chunk_size,
        num_experts=int(expert_cfg.get("num_experts", 5)),
    )
    model = LatentWAMPolicy(
        backbone=backbone,
        memory_encoder=memory,
        nominal_action_head=nominal,
        world_model=world,
        router=router,
        residual_experts=experts,
        visual_teacher=teacher,
        context_dim=context_dim,
        memory_dim=hidden_dim,
        world_dim=hidden_dim,
        expert_hidden_dim=int(expert_cfg.get("hidden_dim", 512)),
        residual_gate_threshold=float(expert_cfg.get("residual_gate_threshold", 0.05)),
        ablation=cfg.get("ablation"),
    )
    return model.to(device)


def build_sequence_dataloader(cfg: dict[str, Any], model):
    torch = require_torch()
    data_cfg = cfg["data"]
    dataset = LiberoSequenceDataset(
        dataset_root=data_cfg["data_root"],
        processor=model.backbone.processor,
        dataset_names=data_cfg["dataset_names"],
        history_length=int(data_cfg.get("history_length", 8)),
        long_memory_slots=int(data_cfg.get("long_memory_slots", 4)),
        future_horizons=data_cfg.get("future_horizons", [1, 4, 8]),
        split=data_cfg.get("split", "train"),
        resize_resolution=tuple(model.backbone.resize_resolution),
        image_aug=bool(data_cfg.get("image_aug", False)),
        use_proprio=bool(data_cfg.get("use_proprio", True)),
        openvla_root=cfg["backbone"].get("openvla_root", "external/openvla-oft"),
        limit=data_cfg.get("limit_batches"),
        joint_action_normalization=bool(data_cfg.get("joint_action_normalization", True)),
    )
    cfg["data"]["joint_action_statistics"] = _jsonable(dataset.joint_action_statistics)
    cache_path = cfg.get("teacher", {}).get("cache_path")
    if cache_path:
        cache_state = torch.load(Path(cache_path), map_location="cpu")
        if cache_state.get("format") != "latent_teacher_cache_v1":
            raise ValueError(f"Unsupported teacher cache format: {cache_path}")
        metadata = cache_state.get("metadata", {})
        expected = {
            "teacher_checkpoint": cfg["teacher"].get("checkpoint"),
            "spatial_tokens": int(cfg["teacher"].get("spatial_tokens", 16)),
            "target_dim": int(cfg["teacher"].get("target_dim", 384)),
            "future_horizons": list(data_cfg.get("future_horizons", [1, 4, 8])),
            "dataset_names": list(data_cfg["dataset_names"]),
            "dataset_fingerprint": hashlib.sha256(
                "|".join(
                    [str(Path(data_cfg["data_root"]).resolve()), *data_cfg["dataset_names"]]
                ).encode("utf-8")
            ).hexdigest(),
            "transform_hash": hashlib.sha256(
                "upstream_rlds_resize_then_frozen_dinov2_processor".encode("utf-8")
            ).hexdigest(),
        }
        mismatches = {
            key: {"cache": metadata.get(key), "config": value}
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Teacher cache metadata mismatch: {mismatches}")
        targets = cache_state.get("targets", {})
        base_dataset = dataset

        class CachedTargetDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                for sample in base_dataset:
                    key = f"{sample['episode_id']}:{sample['step_id']}"
                    if key not in targets:
                        raise KeyError(f"Teacher cache miss for {key} in {cache_path}.")
                    record = targets[key]
                    output = dict(sample)
                    output["current_latent_target"] = record["current"].float()
                    output["future_latent_targets"] = record["future"].float()
                    yield output

        dataset = CachedTargetDataset()
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        collate_fn=LatentWAMCollator(),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)) and resolve_device(cfg).startswith("cuda"),
    )


def configure_stage(model, stage: str) -> None:
    if stage not in {"pretrain", "joint"}:
        raise ValueError("stage must be pretrain or joint.")
    train_components = {"memory_encoder", "nominal_action_head", "world_model"}
    if stage == "joint":
        train_components.update({"router", "residual_experts", "expert_context"})
    for name in TRAINABLE_COMPONENTS:
        component = getattr(model, name)
        enabled = name in train_components
        component.train(enabled)
        for parameter in component.parameters():
            parameter.requires_grad_(enabled)
    model.backbone.freeze()
    if model.visual_teacher is not None:
        model.visual_teacher.freeze()


def build_optimizer(cfg: dict[str, Any], model):
    torch = require_torch()
    training_cfg = cfg["training"]
    base_lr = float(training_cfg.get("learning_rate", 1e-4))
    world_lr = float(training_cfg.get("world_learning_rate", base_lr))
    world_ids = {id(parameter) for parameter in model.world_model.parameters() if parameter.requires_grad}
    world_params = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) in world_ids]
    other_params = [parameter for parameter in model.parameters() if parameter.requires_grad and id(parameter) not in world_ids]
    groups = []
    if world_params:
        groups.append({"params": world_params, "lr": world_lr, "name": "world_model"})
    if other_params:
        groups.append({"params": other_params, "lr": base_lr, "name": "policy_heads"})
    if not groups:
        raise RuntimeError("No trainable parameters after stage configuration.")
    return torch.optim.AdamW(
        groups,
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
    )


def checkpoint_state(model, optimizer, scheduler, scaler, step: int, cfg: dict[str, Any], stage: str):
    return {
        "format": "latent_wam_components_v1",
        "stage": stage,
        "step": int(step),
        "config": cfg,
        "components": {name: getattr(model, name).state_dict() for name in TRAINABLE_COMPONENTS},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
    }


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, step: int, cfg: dict[str, Any], stage: str):
    torch = require_torch()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_state(model, optimizer, scheduler, scaler, step, cfg, stage), path)


def load_checkpoint(path: str | Path, model, optimizer=None, scheduler=None, scaler=None, resume: bool = False) -> int:
    torch = require_torch()
    state = torch.load(Path(path), map_location="cpu")
    if state.get("format") != "latent_wam_components_v1":
        raise ValueError(f"Unsupported checkpoint format in {path}.")
    for name, component_state in state["components"].items():
        if hasattr(model, name):
            getattr(model, name).load_state_dict(component_state)
    if resume:
        if optimizer is not None and state.get("optimizer") is not None:
            optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if scaler is not None and state.get("scaler") is not None:
            scaler.load_state_dict(state["scaler"])
        return int(state.get("step", 0))
    return 0


def _json_log(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def run_training(cfg: dict[str, Any], stage: str, resume: str | None = None, init_checkpoint: str | None = None):
    torch = require_torch()
    if cfg.get("ablation", {}).get("analysis_only", False):
        raise ValueError(
            f"Ablation {cfg['ablation'].get('name')} is analysis-only and cannot be used as a training config."
        )
    output_dir = Path(cfg.get("output_dir", f"outputs/train/latent_wam_{stage}"))
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(cfg.get("seed", 7)))
    model = build_latent_policy(cfg, include_teacher=not bool(cfg.get("teacher", {}).get("cache_path")))
    configure_stage(model, stage)
    cfg["parameter_counts"] = {
        "openvla_frozen": sum(parameter.numel() for parameter in model.backbone.model.parameters()),
        "visual_teacher_frozen": (
            sum(parameter.numel() for parameter in model.visual_teacher.parameters())
            if model.visual_teacher is not None
            else 0
        ),
        **{
            name: sum(parameter.numel() for parameter in getattr(model, name).parameters())
            for name in TRAINABLE_COMPONENTS
        },
    }
    cfg["parameter_counts"]["trainable_total"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    dataloader = build_sequence_dataloader(cfg, model)
    (output_dir / "config_resolved.json").write_text(
        json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
    )
    optimizer = build_optimizer(cfg, model)
    max_steps = int(cfg["training"].get("max_steps", 1000))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    precision = str(cfg["training"].get("precision", "bf16"))
    device = resolve_device(cfg)
    scaler = make_grad_scaler(device, precision)

    start_step = 0
    if init_checkpoint:
        load_checkpoint(init_checkpoint, model, resume=False)
    if resume:
        if init_checkpoint:
            raise ValueError("Use either resume or init_checkpoint, not both.")
        start_step = load_checkpoint(resume, model, optimizer, scheduler, scaler, resume=True)

    grad_accumulation = int(cfg["training"].get("grad_accumulation_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    save_freq = int(cfg["training"].get("save_freq", 100))
    log_freq = int(cfg["training"].get("log_freq", 10))
    weights = cfg["loss_weights"]
    log_path = output_dir / "train_log.jsonl"
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    iterator = iter(dataloader)

    while step < max_steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("Sequence dataset yielded no valid windows.") from exc

        route = router_schedule(
            step,
            max_steps,
            soft_warmup_ratio=float(cfg["router"].get("soft_warmup_ratio", 0.10)),
            anneal_end_ratio=float(cfg["router"].get("anneal_end_ratio", 0.30)),
        )
        forcing = 1.0 if stage == "pretrain" else teacher_forcing_probability(
            step,
            max_steps,
            nominal_start_ratio=float(cfg["action_condition"].get("nominal_start_ratio", 0.30)),
            nominal_end_ratio=float(cfg["action_condition"].get("nominal_end_ratio", 0.70)),
            final_nominal_probability=float(
                cfg["action_condition"].get("final_nominal_probability", 0.80)
            ),
        )
        condition_mode = "ground_truth" if stage == "pretrain" else "scheduled"
        with autocast_context(device, precision):
            outputs = model(
                batch,
                action_condition_mode=condition_mode,
                teacher_forcing_probability=forcing,
                router_hard_topk=bool(route["hard_topk"]),
                router_temperature=float(route["temperature"]),
                compute_teacher_targets=True,
            )
            losses = latent_wam_training_losses(outputs, batch, weights, stage=stage)
            scaled_loss = losses["total_loss"] / grad_accumulation
        scaler.scale(scaled_loss).backward()
        micro_step += 1
        if micro_step % grad_accumulation:
            continue

        scaler.unscale_(optimizer)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        step += 1

        if step % log_freq == 0 or step == 1:
            target_actions = batch["target_actions"].to(outputs["final_actions"].device).float()
            future_errors = (
                (outputs["predicted_future_latents"].float() - outputs["future_latent_targets"].float())
                .square()
                .mean(dim=(0, 2, 3))
                .detach()
                .cpu()
                .tolist()
            )
            forcing_fraction = float(outputs["teacher_forcing_mask"].float().mean())
            record = {
                "step": step,
                "stage": stage,
                "model_variant": cfg.get("model", {}).get("variant", "latent_wam_residual_moe"),
                "ablation": cfg.get("ablation", {}).get("name"),
                "lr": [group["lr"] for group in optimizer.param_groups],
                "gradient_norm": float(gradient_norm),
                "teacher_forcing_probability": forcing,
                "nominal_condition_probability": 1.0 - forcing,
                "teacher_forcing_fraction": forcing_fraction,
                "action_condition_source": (
                    "ground_truth" if forcing_fraction == 1.0 else "nominal" if forcing_fraction == 0.0 else "mixed"
                ),
                "router_hard_topk": bool(route["hard_topk"]),
                "router_temperature": float(route["temperature"]),
                "router_entropy": float(outputs["router_entropy"].float().mean()),
                "expert_usage": outputs["expert_weights"].float().mean(dim=0).detach().cpu().tolist(),
                "residual_gate_fraction": float(outputs["residual_gate"].float().mean()),
                "action_distance_gate_mean": float(outputs["action_distance_gate"].float().mean()),
                "nominal_target_distance": float(outputs["nominal_target_distance"].nanmean()),
                "nominal_target_l1": float(
                    (outputs["nominal_actions"].float() - target_actions).abs().mean().detach()
                ),
                "final_target_l1": float(
                    (outputs["final_actions"].float() - target_actions).abs().mean().detach()
                ),
                "residual_norm": float(outputs["action_residual"].float().norm(dim=-1).mean().detach()),
                "future_horizon_mse": {
                    str(horizon): float(error)
                    for horizon, error in zip(cfg["data"]["future_horizons"], future_errors)
                },
                "teacher_cache": cfg.get("teacher", {}).get("cache_path"),
                **{name: float(value.detach()) for name, value in losses.items()},
            }
            _json_log(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)

        if step % save_freq == 0 or step == max_steps:
            save_checkpoint(
                output_dir / "checkpoint_latest.pt", model, optimizer, scheduler, scaler, step, cfg, stage
            )
    return output_dir / "checkpoint_latest.pt"
