#!/usr/bin/env python3
"""Run automated architecture-risk gates before long flow-WAM training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training import flow_wam_skill_losses
from mowe_wam.training.flow_runtime import (
    build_flow_dataloader,
    build_flow_policy,
    configure_flow_stage,
    deep_update,
)
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def require(condition, risk_id, message, passed):
    if not condition:
        raise RuntimeError(f"{risk_id}: {message}")
    passed.append(risk_id)


def synthetic_preflight(backward: bool):
    from check_flow_wam_forward import build_model, make_batch

    torch = require_torch()
    model = build_model(torch, 2)
    batch = make_batch(torch, 2)
    output = model(
        batch,
        action_condition_mode="scheduled",
        teacher_forcing_probability=0.5,
        route_mode="oracle",
        flow_seed=7,
    )
    losses = flow_wam_skill_losses(
        output,
        batch,
        {
            "flow_nominal": 1.0,
            "flow_expert": 1.0,
            "gripper_bce": 1.0,
            "route": 1.0,
            "world": 1.0,
            "delta": 0.5,
            "load_balance": 0.01,
            "residual": 0.001,
            "endpoint": 0.05,
        },
        stage="joint",
    )
    if backward:
        losses["total_loss"].backward()
    return model, batch, output, losses


def real_data_lightweight_preflight(args):
    """Exercise real RLDS + real DINO while replacing only the 7B context encoder."""

    from check_flow_wam_forward import build_model
    from inspect_latent_sequence_dataset import load_processor

    from mowe_wam.backbones import VisualTargetEncoder
    from mowe_wam.data import LatentWAMCollator, LiberoSequenceDataset

    torch = require_torch()
    processor = load_processor(args.processor_checkpoint, args.openvla_root)
    resolution = tuple(processor.image_processor.input_sizes[0][-2:])
    dataset = LiberoSequenceDataset(
        args.data_root,
        processor,
        dataset_names=[args.dataset_name],
        future_horizons=(1, 4, 8, 16),
        action_chunk_size=16,
        resize_resolution=resolution,
        limit=1,
        openvla_root=args.openvla_root,
        skill_sidecar_path=args.skill_sidecar,
    )
    batch = next(
        iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=1,
                collate_fn=LatentWAMCollator(),
                num_workers=0,
            )
        )
    )

    class RealBatchContext:
        hidden_dim = 64

        def extract_context_features(self, values):
            def sequence_features(pixels):
                pooled = pixels.float().mean(dim=(2, 3, 4)).unsqueeze(-1)
                return pooled.repeat(1, 1, self.hidden_dim)

            def current_features(pixels):
                pooled = pixels.float().mean(dim=(1, 2, 3)).unsqueeze(-1)
                return pooled.repeat(1, self.hidden_dim)

            current_primary = current_features(values["pixel_values_primary"])
            current_wrist = current_features(values["pixel_values_wrist"])
            history_primary = sequence_features(values["history_pixel_values_primary"])
            history_wrist = sequence_features(values["history_pixel_values_wrist"])
            long_primary = sequence_features(values["long_history_pixel_values_primary"])
            long_wrist = sequence_features(values["long_history_pixel_values_wrist"])
            return {
                "current_visual_views": torch.stack([current_primary, current_wrist], dim=1),
                "history_visual_views": torch.stack([history_primary, history_wrist], dim=2),
                "long_history_visual_views": torch.stack([long_primary, long_wrist], dim=2),
                "language": current_primary.new_zeros((current_primary.shape[0], self.hidden_dim)),
            }

        def keep_frozen_backbone_eval(self):
            return None

    teacher = VisualTargetEncoder(
        args.teacher_checkpoint,
        spatial_grid=4,
        target_dim=384,
        num_spatial_tokens=16,
        device="cpu",
        dtype="float32",
    )
    with torch.no_grad():
        batch["current_latent_target"] = teacher.encode(batch["current_raw_pixel_values"])[..., :32]
        future = batch["future_raw_pixel_values"]
        batch["future_latent_targets"] = teacher.encode(future.flatten(0, 1)).reshape(
            1, future.shape[1], 16, 384
        )[..., :32]
    model = build_model(torch, 1)
    model.backbone = RealBatchContext()
    output = model(
        batch,
        action_condition_mode="ground_truth",
        route_mode="oracle",
        flow_seed=7,
    )
    losses = flow_wam_skill_losses(
        output,
        batch,
        {
            "flow_nominal": 1.0,
            "flow_expert": 1.0,
            "gripper_bce": 1.0,
            "route": 1.0,
            "world": 1.0,
            "delta": 0.5,
            "load_balance": 0.01,
            "residual": 0.001,
            "endpoint": 0.05,
        },
        stage="joint",
    )
    if args.backward:
        losses["total_loss"].backward()
    return model, batch, output, losses


def real_preflight(args):
    from mowe_wam.backbones import resolve_original_openvla_identity

    require_torch()
    cfg = load_config(args.config)
    identity = resolve_original_openvla_identity(
        args.checkpoint,
        revision=args.backbone_revision or cfg.get("backbone", {}).get("revision"),
        repo_id=cfg.get("backbone", {}).get("repo_id", "openvla/openvla-7b"),
    )
    deep_update(
        cfg,
        {
            "data": {
                "data_root": args.data_root,
                "skill_sidecar_path": args.skill_sidecar,
                "limit_batches": 1,
            },
            "backbone": {
                "checkpoint": args.checkpoint,
                "repo_id": identity["repo_id"],
                "revision": identity["revision"],
                "identity": identity,
                "dtype": args.precision,
            },
            "teacher": {
                "checkpoint": args.teacher_checkpoint,
                "dtype": args.precision,
            },
            "training": {"precision": args.precision},
        },
    )
    stage = str(cfg.get("stage", "nominal_flow_pretrain"))
    model = build_flow_policy(cfg, include_teacher=not bool(cfg["teacher"].get("cache_path")))
    configure_flow_stage(model, stage)
    batch = next(iter(build_flow_dataloader(cfg, model)))
    route_mode = "oracle" if stage != "nominal_flow_pretrain" else "predicted"
    output = model(
        batch,
        action_condition_mode="ground_truth",
        route_mode=route_mode,
        flow_seed=7,
        compute_residual=stage != "nominal_flow_pretrain",
    )
    losses = flow_wam_skill_losses(output, batch, cfg["loss_weights"], stage=stage)
    if args.backward:
        losses["total_loss"].backward()
    return model, batch, output, losses


def validate(model, batch, output, losses, backward):
    torch = require_torch()
    passed = []
    batch_size = output["actions"].shape[0]
    chunk_size = int(model.nominal_action_head.chunk_size)
    require(output["nominal_motion"].shape == (batch_size, chunk_size, 6), "R1", "nominal motion shape", passed)
    require(output["gripper_logits"].shape == (batch_size, chunk_size, 1), "R9", "gripper head shape", passed)
    require(output["actions"].shape == (batch_size, chunk_size, 7), "R9", "final action shape", passed)
    require(output["current_view_weights"].shape == (batch_size, 2), "R27", "dual-view weights", passed)
    require(
        bool(
            torch.allclose(
                output["current_view_weights"].sum(dim=-1),
                torch.ones_like(output["current_view_weights"].sum(dim=-1)),
            )
        ),
        "R27",
        "dual-view weights do not sum to one",
        passed,
    )
    require(output["route_world_tokens"].shape[:2] == (batch_size, chunk_size), "R25", "stepwise route-world tokens missing", passed)
    require(output["router_logits"].shape == (batch_size, chunk_size, 7), "R2", "temporal route shape", passed)
    require(bool(torch.isfinite(losses["total_loss"])), "R6", "non-finite total loss", passed)
    require(
        int(output["null_motion_zero_violation_count"]) == 0,
        "R21",
        "null_finish produced non-zero motion residual",
        passed,
    )
    require(
        bool(((batch["target_gripper"] == 0) | (batch["target_gripper"] == 1)).all()),
        "R9",
        "gripper target is not canonical binary",
        passed,
    )
    require(
        bool(batch["expert_skill_mask"].any()),
        "R7",
        "real/synthetic batch has no joined timestep skill labels",
        passed,
    )
    require(
        not any("transition_difference" in name or "delta_h" in name for name, _ in model.router.named_modules()),
        "R4",
        "router contains a forbidden delta-h/pre-post branch",
        passed,
    )
    require(
        not (
            {
                "target_actions",
                "target_motion",
                "target_gripper",
                "expert_skill_labels",
                "expert_skill_mask",
                "future_latent_targets",
            }
            & set(output["context_input_keys"])
        ),
        "R2",
        "action/skill/teacher targets crossed the backbone context boundary",
        passed,
    )
    if model.visual_teacher is not None:
        require(not model.visual_teacher.training, "R7", "visual teacher is in train mode", passed)
        require(
            not any(parameter.requires_grad for parameter in model.visual_teacher.parameters()),
            "R7",
            "visual teacher has trainable parameters",
            passed,
        )
    if backward:
        gradient_found = any(
            parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        require(gradient_found, "R6", "no finite trainable gradient", passed)
        if "expert_flow" in output:
            labels = batch["expert_skill_labels"]
            mask = batch["expert_skill_mask"].bool()
            for index, head in enumerate(model.residual_experts.velocity_heads):
                if not bool((mask & labels.eq(index)).any()):
                    continue
                valid_gradient = any(
                    parameter.grad is not None
                    and bool(torch.isfinite(parameter.grad).all())
                    and float(parameter.grad.detach().abs().sum()) > 0
                    for parameter in head.parameters()
                )
                require(valid_gradient, "R20", f"covered motor expert {index} has no finite gradient", passed)
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_nominal_flow_wam.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--precision", choices=["bf16", "fp16", "float32"], default=None)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real-data-lightweight", action="store_true")
    parser.add_argument(
        "--processor-checkpoint",
        default="openvla/openvla-7b",
    )
    parser.add_argument("--teacher-checkpoint", default="facebook/dinov2-small")
    parser.add_argument("--openvla-root", default="external/openvla-oft")
    parser.add_argument("--dataset-name", default="libero_spatial_no_noops")
    parser.add_argument("--skill-sidecar", default="datasets/libero_cot_rlds/cot_file.json")
    args = parser.parse_args()
    if args.synthetic and args.real_data_lightweight:
        raise SystemExit("Choose only one of --synthetic or --real-data-lightweight.")
    if args.synthetic:
        model, batch, output, losses = synthetic_preflight(args.backward)
        mode = "synthetic"
    elif args.real_data_lightweight:
        if not args.data_root:
            raise SystemExit("--real-data-lightweight requires --data-root.")
        model, batch, output, losses = real_data_lightweight_preflight(args)
        mode = "real_rlds_real_dino_lightweight_context"
    else:
        if not args.data_root or not args.checkpoint or not args.backbone_revision:
            raise SystemExit(
                "Real preflight requires --data-root, --checkpoint, and "
                "--backbone-revision, or pass --synthetic."
            )
        model, batch, output, losses = real_preflight(args)
        mode = "real_batch"
    passed = validate(model, batch, output, losses, args.backward)
    print(
        json.dumps(
            {
                "status": "preflight_passed",
                "mode": mode,
                "backward": args.backward,
                "risk_gates": passed,
                "total_loss": float(losses["total_loss"].detach()),
                "note": "Preflight is not benchmark evidence.",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
