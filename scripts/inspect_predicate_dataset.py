#!/usr/bin/env python3
"""Inspect mock or future LIBERO predicate dataset samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data.libero_predicate_dataset import LiberoPredicateDataset, infer_shape


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dataset-root", default="MOCK")
    parser.add_argument("--split", default="train")
    parser.add_argument("--predicate-label-path", default=None)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()

    root = "MOCK" if args.mock else args.dataset_root
    try:
        dataset = LiberoPredicateDataset(
            dataset_root=root,
            split=args.split,
            predicate_label_path=args.predicate_label_path,
            limit=args.limit,
        )
    except (FileNotFoundError, NotImplementedError) as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Dataset length: {len(dataset)}")
    for idx in range(min(len(dataset), args.limit)):
        sample = dataset[idx]
        summary = {key: infer_shape(value) for key, value in sample.items() if key != "task_meta"}
        summary["task_meta"] = sample["task_meta"]
        print(json.dumps({"idx": idx, "shapes": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
