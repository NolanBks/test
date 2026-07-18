#!/usr/bin/env python3
"""Inspect mock or cached trajectory transition labels before predictive training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data import LiberoPredicateDataset, TransitionLabelStore
from mowe_wam.data.libero_predicate_dataset import infer_shape
from mowe_wam.memory import EVENT_TYPES


def _inspect_label_store(path: str, horizon: int) -> dict:
    store = TransitionLabelStore(path)
    episodes = list(store.episodes.items())
    valid = sum(max(0, len(steps) - horizon) for _, steps in episodes)
    event_counts: dict[str, int] = {}
    for _, steps in episodes:
        for step in steps:
            raw_event = step.get("event_target", "missing")
            event = EVENT_TYPES[int(raw_event)] if isinstance(raw_event, int) and 0 <= raw_event < len(EVENT_TYPES) else str(raw_event)
            event_counts[event] = event_counts.get(event, 0) + 1
    return {"episodes": len(episodes), "valid_windows": valid, "event_counts": event_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--label-path", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--prediction-horizon", type=int, default=8)
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    if args.history_steps < 1 or args.prediction_horizon < 1:
        raise SystemExit("History and horizon must be positive.")

    if args.label_path:
        print(json.dumps(_inspect_label_store(args.label_path, args.prediction_horizon), indent=2, sort_keys=True))
    if args.data_root:
        root = Path(args.data_root)
        if not root.exists():
            raise SystemExit(f"Missing data root: {root}")
        candidate = root / str(args.dataset_name) if args.dataset_name else None
        print(
            json.dumps(
                {
                    "data_root": str(root),
                    "dataset_name": args.dataset_name,
                    "dataset_path_exists": candidate.exists() if candidate is not None else None,
                    "note": "RLDS episode identifiers and simulator-state fields must be checked while building --label-path; this script never fabricates them.",
                },
                indent=2,
                sort_keys=True,
            )
        )
    if args.mock:
        dataset = LiberoPredicateDataset(
            dataset_root="MOCK",
            limit=args.limit,
            cfg={
                "predictive": True,
                "history_steps": args.history_steps,
                "prediction_horizon": args.prediction_horizon,
                "action_dim": 7,
                "chunk_size": 8,
            },
        )
        for index, sample in enumerate(dataset):
            if index >= args.limit:
                break
            shapes = {key: infer_shape(value) for key, value in sample.items() if key not in {"language", "task_meta", "episode_id"}}
            print(json.dumps({"idx": index, "episode_id": sample["episode_id"], "shapes": shapes}, indent=2, sort_keys=True))
    if not args.mock and not args.label_path and not args.data_root:
        raise SystemExit("Pass --mock, --label-path, and/or --data-root.")


if __name__ == "__main__":
    main()
