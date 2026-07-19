#!/usr/bin/env python3
"""Audit CALVIN ABC train (official NPZ or RLDS) and derive action stats."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.benchmarks.calvin.dataset import resolve_calvin_training_dataset


def _atomic_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument(
        "--dataset-format",
        choices=["auto", "official_npz", "rlds"],
        default="auto",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--skill-config-output")
    parser.add_argument("--min-segment-length", type=int, default=9)
    parser.add_argument("--limit-segments", type=int)
    parser.add_argument(
        "--official-repo-commit",
        default="fa03f01f19c65920e18cf37398a9ce859274af76",
    )
    args = parser.parse_args()
    dataset = resolve_calvin_training_dataset(
        args.dataset_root,
        dataset_format=args.dataset_format,
        min_segment_length=args.min_segment_length,
        official_repo_commit=args.official_repo_commit,
    )
    report = dataset.audit(limit_segments=args.limit_segments)
    output = Path(args.output)
    _atomic_json(output, report)
    if args.skill_config_output:
        _atomic_json(
            Path(args.skill_config_output),
            dataset.skill_config(report, audit_path=str(output.resolve())),
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
