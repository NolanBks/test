#!/usr/bin/env python3
"""Audit the stable episode-level train/validation partition."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data import ShardedVisualTargetCache, episode_partition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--split-seed", type=int, default=17)
    parser.add_argument("--output")
    args = parser.parse_args()
    cache = ShardedVisualTargetCache(args.cache)
    suites = defaultdict(lambda: {"train_episodes": 0, "validation_episodes": 0, "train_windows": 0, "validation_windows": 0})
    train_ids = set()
    validation_ids = set()
    episode_lengths = defaultdict(int)
    for key in cache.index:
        episode_id, step_text = key.rsplit(":", 1)
        episode_lengths[episode_id] = max(episode_lengths[episode_id], int(step_text) + 1)
    max_horizon = max(int(value) for value in cache.metadata.get("future_horizons", [1, 4, 8]))
    for episode_id, episode_length in episode_lengths.items():
        partition = episode_partition(
            episode_id,
            validation_fraction=args.validation_fraction,
            split_seed=args.split_seed,
        )
        dataset_name = episode_id.split(":", 1)[0]
        windows = max(0, episode_length - max_horizon)
        suites[dataset_name][f"{partition}_episodes"] += 1
        suites[dataset_name][f"{partition}_windows"] += windows
        (validation_ids if partition == "validation" else train_ids).add(episode_id)
    report = {
        "format": "flow_wam_episode_split_v1",
        "validation_fraction": args.validation_fraction,
        "split_seed": args.split_seed,
        "cache_record_count": cache.record_count,
        "train_episodes": len(train_ids),
        "validation_episodes": len(validation_ids),
        "overlap": sorted(train_ids & validation_ids),
        "train_windows": sum(value["train_windows"] for value in suites.values()),
        "validation_windows": sum(value["validation_windows"] for value in suites.values()),
        "suites": dict(sorted(suites.items())),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
