#!/usr/bin/env python3
"""Continuously read feature-store windows and gate anonymous-memory growth."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data import EpisodeAwareDistributedSampler, MoWEFeatureWindowDataset
from mowe_wam.training.distributed import (
    consolidate_runtime_identities,
    initialize_distributed,
    local_runtime_identity,
    process_resource_metrics,
)


def _linear_slope_per_1k_steps(snapshots, metric: str):
    points = [
        (float(item["step"]), float(item[metric]))
        for item in snapshots
        if item.get(metric) is not None
    ]
    if len(points) < 2:
        return None
    x_mean = sum(point[0] for point in points) / len(points)
    y_mean = sum(point[1] for point in points) / len(points)
    denominator = sum((point[0] - x_mean) ** 2 for point in points)
    if denominator == 0:
        return 0.0
    slope_per_step = sum(
        (x - x_mean) * (y - y_mean) for x, y in points
    ) / denominator
    return slope_per_step * 1000.0


def _missing_required_metrics(snapshots, required_metrics):
    missing = {}
    for snapshot in snapshots:
        absent = sorted(
            metric
            for metric in required_metrics
            if snapshot.get(metric) is None
        )
        if absent:
            missing[str(snapshot.get("step", "baseline"))] = absent
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--sample-every", type=int, default=250)
    parser.add_argument("--max-anon-growth-mib", type=float, default=512.0)
    parser.add_argument("--max-working-set-growth-mib", type=float, default=2048.0)
    parser.add_argument("--max-anon-slope-mib-per-1k-steps", type=float, default=64.0)
    parser.add_argument(
        "--max-working-set-slope-mib-per-1k-steps", type=float, default=256.0
    )
    parser.add_argument("--min-post-warmup-samples", type=int, default=3)
    parser.add_argument("--max-open-feature-shards", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--shuffle-block-size", type=int, default=256)
    parser.add_argument(
        "--disable-system-monitoring",
        "--disable-cgroup-monitoring",
        action="store_true",
        dest="disable_system_monitoring",
        help="Allow a degraded data-read soak when the platform does not expose cgroup metrics.",
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.disable_system_monitoring:
        os.environ["MOWE_DISABLE_SYSTEM_MONITORING"] = "1"
    if args.steps < 1 or not 0 <= args.warmup_steps < args.steps:
        raise SystemExit("Require steps > warmup_steps >= 0.")
    if args.sample_every < 1 or args.min_post_warmup_samples < 2:
        raise SystemExit("Require sample_every >= 1 and min_post_warmup_samples >= 2.")
    if any(
        value < 0
        for value in (
            args.max_anon_growth_mib,
            args.max_working_set_growth_mib,
            args.max_anon_slope_mib_per_1k_steps,
            args.max_working_set_slope_mib_per_1k_steps,
        )
    ):
        raise SystemExit("Memory growth and slope limits must be non-negative.")
    cfg = {
        "training": {
            "device": "cpu",
            "distributed": {"enabled": "auto", "backend": "gloo", "timeout_seconds": 1800},
        }
    }
    context = initialize_distributed(cfg)
    try:
        dataset = MoWEFeatureWindowDataset(
            args.store,
            partition="train",
            max_open_feature_shards=args.max_open_feature_shards,
        )
        sampler = EpisodeAwareDistributedSampler(
            dataset,
            rank=context.rank,
            world_size=context.world_size,
            seed=args.seed,
            shuffle=True,
            shuffle_block_size=args.shuffle_block_size,
        )
        if len(sampler) == 0:
            raise RuntimeError(f"Rank {context.rank} has no assigned windows.")
        iterator = iter(sampler)
        snapshots = []
        checksum = 0.0
        started = time.perf_counter()
        baseline_events = process_resource_metrics(context)
        required_cgroup_metrics = set() if args.disable_system_monitoring else {
            "cgroup_memory_max_mib",
            "cgroup_memory_anon_mib",
            "cgroup_memory_working_set_mib",
            "cgroup_event_oom",
            "cgroup_event_oom_kill",
        }
        missing_metrics = sorted(required_cgroup_metrics - set(baseline_events))
        if missing_metrics:
            raise RuntimeError(
                "Feature-store soak requires cgroup-v2 memory/oom metrics; missing "
                f"{missing_metrics}. Do not treat an unmeasured local smoke as a memory gate."
            )
        for step in range(1, args.steps + 1):
            try:
                index = next(iterator)
            except StopIteration:
                iterator = iter(sampler)
                index = next(iterator)
            sample = dataset[index]
            # Touch every large mmap-backed tensor; this prevents a metadata-only
            # loop from producing a falsely reassuring RSS profile.
            checksum += float(sample["current_visual_views"].float().sum())
            checksum += float(sample["history_visual_views"].float().sum())
            checksum += float(sample["long_history_visual_views"].float().sum())
            checksum += float(sample["current_latent_target"].float().sum())
            checksum += float(sample["future_latent_targets"].float().sum())
            if step == args.warmup_steps or step % args.sample_every == 0 or step == args.steps:
                snapshots.append({"step": step, **process_resource_metrics(context)})
        elapsed = time.perf_counter() - started
        post_warmup = [item for item in snapshots if int(item["step"]) >= args.warmup_steps]
        if len(post_warmup) < args.min_post_warmup_samples:
            raise RuntimeError(
                "Feature-store soak collected too few post-warmup samples: "
                f"{len(post_warmup)} < {args.min_post_warmup_samples}. Increase --steps, "
                "decrease --sample-every, or lower --warmup-steps."
            )
        first = post_warmup[0]
        last = post_warmup[-1]
        missing_snapshot_metrics = _missing_required_metrics(
            snapshots, required_cgroup_metrics
        )

        def growth(name: str):
            if name not in first or name not in last:
                return None
            return float(last[name]) - float(first[name])

        anon_growth = growth("cgroup_memory_anon_mib")
        working_growth = growth("cgroup_memory_working_set_mib")
        anon_slope = _linear_slope_per_1k_steps(
            post_warmup, "cgroup_memory_anon_mib"
        )
        working_slope = _linear_slope_per_1k_steps(
            post_warmup, "cgroup_memory_working_set_mib"
        )
        event_deltas = {
            name: int(last.get(name, 0)) - int(baseline_events.get(name, 0))
            for name in (
                "cgroup_event_high",
                "cgroup_event_oom",
                "cgroup_event_oom_kill",
            )
            if name in baseline_events or name in last
        }
        local = {
            "rank": context.rank,
            "world_size": context.world_size,
            "steps": args.steps,
            "elapsed_seconds": elapsed,
            "windows_per_second": args.steps / max(elapsed, 1e-9),
            "checksum": checksum,
            "tensorflow_imported": "tensorflow" in sys.modules,
            "sampler": sampler.assignment_report(),
            "sampler_state": sampler.state_dict(),
            "runtime_identity": local_runtime_identity(context, baseline_events),
            "anon_growth_mib": anon_growth,
            "working_set_growth_mib": working_growth,
            "anon_slope_mib_per_1k_steps": anon_slope,
            "working_set_slope_mib_per_1k_steps": working_slope,
            "cgroup_event_deltas": event_deltas,
            "missing_required_metrics": missing_snapshot_metrics,
            "snapshots": snapshots,
        }
        cgroup_gate_passed = args.disable_system_monitoring or (
            not missing_snapshot_metrics
            and all(
                event_deltas.get(name, 0) == 0
                for name in ("cgroup_event_oom", "cgroup_event_oom_kill")
            )
            and anon_growth is not None
            and anon_growth <= args.max_anon_growth_mib
            and working_growth is not None
            and working_growth <= args.max_working_set_growth_mib
            and anon_slope is not None
            and anon_slope <= args.max_anon_slope_mib_per_1k_steps
            and working_slope is not None
            and working_slope <= args.max_working_set_slope_mib_per_1k_steps
        )
        local["cgroup_monitoring_enabled"] = not args.disable_system_monitoring
        local["passed"] = not local["tensorflow_imported"] and cgroup_gate_passed
        reports = context.all_gather_objects(local)
        runtime_identity = consolidate_runtime_identities(
            [value["runtime_identity"] for value in reports],
            require_cuda=False,
            require_node_identity=not args.disable_system_monitoring,
        )
        output = {
            "format": "mowe_feature_store_soak_v1",
            "store": str(Path(args.store).resolve()),
            "cgroup_monitoring_enabled": not args.disable_system_monitoring,
            "resource_monitoring_mode": (
                "cgroup_v1_or_v2" if not args.disable_system_monitoring
                else "degraded_gpu_and_process_unavailable"
            ),
            "runtime_identity": runtime_identity,
            "limits": {
                "max_anon_growth_mib": args.max_anon_growth_mib,
                "max_working_set_growth_mib": args.max_working_set_growth_mib,
                "max_anon_slope_mib_per_1k_steps": args.max_anon_slope_mib_per_1k_steps,
                "max_working_set_slope_mib_per_1k_steps": (
                    args.max_working_set_slope_mib_per_1k_steps
                ),
                "min_post_warmup_samples": args.min_post_warmup_samples,
            },
            "reports": sorted(reports, key=lambda value: int(value["rank"])),
            "passed": all(bool(value["passed"]) for value in reports),
        }
        if context.is_main:
            rendered = json.dumps(output, indent=2, sort_keys=True)
            if args.output:
                path = Path(args.output)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(rendered + "\n", encoding="utf-8")
            print(rendered, flush=True)
        context.barrier()
        if not output["passed"]:
            raise RuntimeError("Feature-store memory soak gate failed.")
    finally:
        context.close()


if __name__ == "__main__":
    main()
