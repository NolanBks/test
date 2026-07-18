#!/usr/bin/env python3
"""Analyze mock or real MoWE-WAM JSONL logs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.analysis.expert_usage import summarize_expert_usage
from mowe_wam.analysis.memory_usage import summarize_memory_usage
from mowe_wam.analysis.predicate_timeline import build_predicate_timeline, compute_phase_expert_alignment
from mowe_wam.predicates.labeler import build_mock_trajectory, label_trajectory


def _write_mock_log(path: Path) -> None:
    trajectory, task_meta = build_mock_trajectory()
    labels = label_trajectory(trajectory, task_meta=task_meta)
    top1 = [0, 1, 2, 2, 3]
    memory_events = ["none", "none", "grasp_acquired", "none", "subgoal_complete"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx, predicates in enumerate(labels):
            f.write(
                json.dumps(
                    {"step": idx, "predicates": predicates, "router_top1": top1[idx], "memory_event": memory_events[idx]}
                )
                + "\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--log-path", default=None)
    parser.add_argument("--output-dir", default="outputs/analysis/mock")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log_path) if args.log_path else output_dir / "mock_eval_log.jsonl"
    if args.mock:
        _write_mock_log(log_path)
    if not log_path.exists():
        raise SystemExit(f"Missing log path: {log_path}")

    summary = {
        **summarize_expert_usage(log_path),
        **summarize_memory_usage(log_path),
        **build_predicate_timeline(log_path),
        **compute_phase_expert_alignment(log_path),
    }
    output_path = output_dir / "summary.json"
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote analysis summary to {output_path}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
