#!/usr/bin/env python3
"""Audit a MoWE feature store and its episode-aware rank assignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data import (
    EpisodeAwareDistributedSampler,
    MoWEFeatureWindowDataset,
    audit_feature_store,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle-block-size", type=int, default=256)
    parser.add_argument("--verify-all-checksums", action="store_true")
    parser.add_argument("--sample-windows", type=int, default=16)
    parser.add_argument("--max-window-imbalance-ratio", type=float)
    parser.add_argument("--max-suite-imbalance-ratio", type=float)
    parser.add_argument("--max-skill-imbalance-ratio", type=float)
    parser.add_argument("--output")
    args = parser.parse_args()
    for name in (
        "max_window_imbalance_ratio",
        "max_suite_imbalance_ratio",
        "max_skill_imbalance_ratio",
    ):
        value = getattr(args, name)
        if value is not None and value < 1.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be >= 1 when provided.")
    report = audit_feature_store(
        args.store, verify_all_checksums=args.verify_all_checksums
    )
    dataset = MoWEFeatureWindowDataset(args.store, partition="train")
    samplers = [
        EpisodeAwareDistributedSampler(
            dataset,
            rank=rank,
            world_size=args.world_size,
            seed=args.seed,
            shuffle_block_size=args.shuffle_block_size,
        )
        for rank in range(args.world_size)
    ]
    owners = {}
    overlap = []
    for sampler in samplers:
        for episode_index in sampler.local_episode_indices:
            previous = owners.setdefault(int(episode_index), sampler.rank)
            if previous != sampler.rank:
                overlap.append(int(episode_index))
    expected = set(dataset.partition_episode_indices)
    assignment_reports = [
        sampler.assignment_report(include_skill_counts=True) for sampler in samplers
    ]

    def imbalance_ratio(key: str, nested_key: str | None = None):
        if nested_key is None:
            values = [int(item.get(key, 0)) for item in assignment_reports]
        else:
            values = [int(item.get(key, {}).get(nested_key, 0)) for item in assignment_reports]
        mean = sum(values) / max(len(values), 1)
        return max(values, default=0) / mean if mean > 0 else 0.0

    suite_names = sorted(
        set().union(*(item.get("suite_window_counts", {}).keys() for item in assignment_reports))
    )
    skill_ids = sorted(
        set().union(*(item.get("target_skill_counts", {}).keys() for item in assignment_reports)),
        key=int,
    )
    observed_skill_counts = {
        skill: sum(int(item.get("target_skill_counts", {}).get(skill, 0)) for item in assignment_reports)
        for skill in skill_ids
    }
    expected_skill_counts = dataset.partition_target_skill_counts()
    assignment = {
        "world_size": args.world_size,
        "reports": assignment_reports,
        "episode_union_complete": set(owners) == expected,
        "episode_overlap": sorted(set(overlap)),
        "fingerprints_agree": len({sampler.assignment_fingerprint for sampler in samplers}) == 1,
        "target_skill_union_complete": observed_skill_counts == expected_skill_counts,
        "target_skill_counts": observed_skill_counts,
        "imbalance_ratios": {
            "windows": imbalance_ratio("window_count"),
            "by_suite": {
                suite: imbalance_ratio("suite_window_counts", suite) for suite in suite_names
            },
            "by_skill": {
                skill: imbalance_ratio("target_skill_counts", skill) for skill in skill_ids
            },
        },
    }
    report["assignment"] = assignment
    threshold_checks = {
        "windows": (
            args.max_window_imbalance_ratio is None
            or assignment["imbalance_ratios"]["windows"] <= args.max_window_imbalance_ratio
        ),
        "suites": (
            args.max_suite_imbalance_ratio is None
            or max(assignment["imbalance_ratios"]["by_suite"].values(), default=0.0)
            <= args.max_suite_imbalance_ratio
        ),
        "skills": (
            args.max_skill_imbalance_ratio is None
            or max(assignment["imbalance_ratios"]["by_skill"].values(), default=0.0)
            <= args.max_skill_imbalance_ratio
        ),
    }
    assignment["configured_imbalance_limits"] = {
        "windows": args.max_window_imbalance_ratio,
        "suites": args.max_suite_imbalance_ratio,
        "skills": args.max_skill_imbalance_ratio,
    }
    assignment["imbalance_checks"] = threshold_checks
    sampled = []
    for index in range(min(len(dataset), max(0, args.sample_windows))):
        sample = dataset[index]
        sampled.append(
            {
                "episode_id": sample["episode_id"],
                "step_id": sample["step_id"],
                "current_views": list(sample["current_visual_views"].shape),
                "future_targets": list(sample["future_latent_targets"].shape),
                "actions": list(sample["target_actions"].shape),
            }
        )
    report["sampled_windows"] = sampled
    report["valid"] = bool(report["valid"]) and all(
        (
            report["assignment"]["episode_union_complete"],
            not report["assignment"]["episode_overlap"],
            report["assignment"]["fingerprints_agree"],
            report["assignment"]["target_skill_union_complete"],
            all(threshold_checks.values()),
        )
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
