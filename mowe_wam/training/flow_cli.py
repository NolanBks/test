"""Shared command-line plumbing for the three flow-WAM training stages."""

from __future__ import annotations

import argparse

from mowe_wam.training.flow_runtime import deep_update, run_flow_training
from mowe_wam.utils.config import load_config


def parse_flow_train_args(default_config: str):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--data-root")
    parser.add_argument("--feature-store")
    parser.add_argument("--checkpoint")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-step", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--grad-accumulation-steps", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--init-wam")
    parser.add_argument("--precision", choices=["bf16", "fp16", "float32"])
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--teacher-cache")
    parser.add_argument("--skill-expert-config")
    parser.add_argument(
        "--route-mode",
        choices=["oracle", "scheduled", "st_gumbel", "predicted", "soft"],
        default="scheduled",
    )
    parser.add_argument("--flow-solver-steps", type=int)
    parser.add_argument("--allow-world-size-change", action="store_true")
    parser.add_argument("--long-run-readiness-report")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None)
    return parser.parse_args()


def run_flow_stage(stage: str, default_config: str) -> None:
    args = parse_flow_train_args(default_config)
    cfg = load_config(args.config)
    configured_stage = cfg.get("stage")
    if configured_stage is not None and configured_stage != stage:
        raise SystemExit(
            f"Config stage {configured_stage!r} does not match this entrypoint stage {stage!r}."
        )
    deep_update(
        cfg,
        {
            "data": {
                "data_root": args.data_root,
                "feature_store_path": args.feature_store,
                "limit_batches": args.limit_batches,
            },
            "backbone": {
                "checkpoint": args.checkpoint,
                "revision": args.backbone_revision,
                "dtype": args.precision,
            },
            "teacher": {
                "checkpoint": args.teacher_checkpoint,
                "cache_path": args.teacher_cache,
                "dtype": args.precision,
            },
            "flow": {"num_inference_steps": args.flow_solver_steps},
            "training": {
                "max_steps": args.max_steps,
                "stop_step": args.stop_step,
                "grad_accumulation_steps": args.grad_accumulation_steps,
                "save_freq": args.save_freq,
                "log_freq": args.log_freq,
                "precision": args.precision,
            },
            "output_dir": args.output_dir,
            "skill_expert_config": args.skill_expert_config,
            "long_run_readiness": {
                "report_path": args.long_run_readiness_report,
            },
        },
    )
    def unresolved(value) -> bool:
        return value is None or "TBD" in str(value)

    feature_backend = cfg["data"].get("backend", "rlds") == "mowe_feature_store_v1"
    if feature_backend:
        if unresolved(cfg["data"].get("feature_store_path")):
            raise SystemExit("Feature-store training requires --feature-store.")
    elif unresolved(cfg["data"].get("data_root")):
        raise SystemExit("RLDS flow-WAM training requires --data-root.")
    if not feature_backend and unresolved(cfg["backbone"].get("checkpoint")):
        raise SystemExit("Real flow-WAM training requires --checkpoint.")
    if stage == "nominal_flow_pretrain" and args.init_wam:
        raise SystemExit("Stage 1 starts from the frozen backbone and must not use --init-wam.")
    if stage in {"expert_warmstart", "joint"} and not (args.init_wam or args.resume):
        predecessor = "Stage 1" if stage == "expert_warmstart" else "Stage 2"
        raise SystemExit(f"{stage} requires --init-wam from {predecessor}, or --resume for the same stage.")
    run_flow_training(
        cfg,
        stage=stage,
        resume=args.resume,
        init_checkpoint=args.init_wam,
        route_mode_override=None if args.route_mode == "scheduled" else args.route_mode,
        allow_world_size_change=args.allow_world_size_change,
    )
