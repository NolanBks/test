#!/usr/bin/env python3
"""Generate predicate pseudo labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.predicates.labeler import build_mock_trajectory, label_trajectory


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use the deterministic mock trajectory.")
    parser.add_argument("--output", default="outputs/mock_predicates/mock_labels.jsonl")
    args = parser.parse_args()

    if not args.mock:
        raise SystemExit("Real trajectory labeling is TBD. Re-run with --mock for local smoke tests.")

    trajectory, task_meta = build_mock_trajectory()
    labels = label_trajectory(trajectory, task_meta=task_meta)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for idx, predicates in enumerate(labels):
            f.write(json.dumps({"step_index": idx, "predicates": predicates}, sort_keys=True) + "\n")
    print(f"Wrote {len(labels)} mock predicate rows to {output}")


if __name__ == "__main__":
    main()
