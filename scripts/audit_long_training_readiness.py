#!/usr/bin/env python3
"""Combine all formal evidence required before a continuous 8-GPU run."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.flow_runtime import read_flow_checkpoint_metadata
from mowe_wam.training.long_run_readiness import (
    audit_long_run_readiness,
    load_json_report,
)
from mowe_wam.utils.config import load_config


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--store", required=True)
    parser.add_argument("--feature-audit", required=True)
    parser.add_argument("--equivalence-report", required=True)
    parser.add_argument("--soak-report", required=True)
    parser.add_argument("--ddp-runtime-audit", required=True)
    parser.add_argument("--skill-expert-config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-mode", choices=["resume", "init"], default="resume")
    parser.add_argument("--allow-world-size-change", action="store_true")
    parser.add_argument(
        "--allow-missing-cgroup-metrics",
        action="store_true",
        help="Accept explicitly degraded node evidence when cgroup metrics are unavailable.",
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--min-equivalence-samples", type=int, default=100)
    parser.add_argument("--min-soak-steps", type=int, default=10000)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_metadata = (
        read_flow_checkpoint_metadata(args.checkpoint) if args.checkpoint else None
    )
    report = audit_long_run_readiness(
        load_config(args.config),
        store=args.store,
        feature_audit=load_json_report(args.feature_audit),
        equivalence_report=load_json_report(args.equivalence_report),
        soak_report=load_json_report(args.soak_report),
        ddp_runtime_audit=load_json_report(args.ddp_runtime_audit),
        world_size=args.world_size,
        min_equivalence_samples=args.min_equivalence_samples,
        min_soak_steps=args.min_soak_steps,
        skill_expert_config=args.skill_expert_config,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_mode=args.checkpoint_mode,
        allow_world_size_change=args.allow_world_size_change,
        allow_missing_cgroup_metrics=args.allow_missing_cgroup_metrics,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _atomic_write(Path(args.output), rendered)
    print(rendered, end="")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
