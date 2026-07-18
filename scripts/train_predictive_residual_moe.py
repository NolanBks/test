#!/usr/bin/env python3
"""Stage 2: jointly train predictive routing and residual action experts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.latent_runtime import deep_update, run_training
from mowe_wam.utils.config import load_config


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_predictive_residual_moe.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--limit-batches", type=int)
    parser.add_argument("--grad-accumulation-steps", type=int)
    parser.add_argument("--save-freq", type=int)
    parser.add_argument("--log-freq", type=int)
    parser.add_argument("--precision", choices=("bf16", "fp16", "float32"))
    parser.add_argument("--teacher-cache")
    parser.add_argument("--init-wam", help="Stage-1 component checkpoint used only for initialization.")
    parser.add_argument("--resume", help="Stage-2 checkpoint including optimizer/scheduler state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    deep_update(
        cfg,
        {
            "data": {"data_root": args.data_root, "limit_batches": args.limit_batches},
            "backbone": {"checkpoint": args.checkpoint},
            "teacher": {"checkpoint": args.teacher_checkpoint, "cache_path": args.teacher_cache},
            "training": {
                "max_steps": args.max_steps,
                "grad_accumulation_steps": args.grad_accumulation_steps,
                "save_freq": args.save_freq,
                "log_freq": args.log_freq,
                "precision": args.precision,
            },
            "output_dir": args.output_dir,
        },
    )
    run_training(cfg, stage="joint", resume=args.resume, init_checkpoint=args.init_wam)


if __name__ == "__main__":
    main()
