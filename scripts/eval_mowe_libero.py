#!/usr/bin/env python3
"""Legacy dry-run wrapper for the upstream OpenVLA LIBERO baseline command.

Flow-WAM one-task and resumable full-suite evaluation lives in
``scripts/eval_libero_temporal_skill.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.utils.config import dump_resolved_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mowe_wam/eval_libero_smoke.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.checkpoint:
        cfg["checkpoint"] = args.checkpoint
    if args.dataset_root:
        cfg["dataset_root"] = args.dataset_root
    if args.suite:
        cfg["task_suite_name"] = args.suite
    if args.num_episodes is not None:
        cfg["num_episodes"] = args.num_episodes
    if args.max_tasks is not None:
        cfg["max_tasks"] = args.max_tasks
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    print("Resolved eval config:")
    print(dump_resolved_config(cfg))
    upstream_checkpoint = cfg.get("checkpoint", "moojink/openvla-7b-oft-finetuned-libero-spatial")
    task_suite = cfg.get("task_suite_name", "libero_spatial")
    print("Upstream smoke template:")
    print(
        "cd external/openvla-oft && "
        "python experiments/robot/libero/run_libero_eval.py "
        f"--pretrained_checkpoint {upstream_checkpoint} "
        f"--task_suite_name {task_suite} "
        "--num_trials_per_task 1"
    )
    if not args.dry_run:
        raise SystemExit(
            "This legacy script only renders the upstream baseline command. Use "
            "scripts/eval_libero_temporal_skill.py --simulator for Flow-WAM evaluation."
        )
    print("Dry run only; no simulation, checkpoint download, or benchmark result was run.")


if __name__ == "__main__":
    main()
