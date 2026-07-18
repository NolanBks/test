#!/usr/bin/env python3
"""Audit torchrun identity, effective batch, GPU visibility, and cgroup memory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.training.distributed import (
    consolidate_runtime_identities,
    distributed_contract,
    enforce_cgroup_memory_guard,
    enforce_gpu_memory_guard,
    enforce_resource_metric_contract,
    initialize_distributed,
    local_runtime_identity,
    process_resource_metrics,
)
from mowe_wam.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/ddp8_nominal_flow_wam.yaml")
    parser.add_argument("--output")
    parser.add_argument("--memory-guard-fraction", type=float)
    parser.add_argument("--gpu-memory-guard-fraction", type=float)
    parser.add_argument(
        "--disable-system-monitoring",
        action="store_true",
        help="Skip host/cgroup metric collection; retain CUDA, NCCL, and GPU-memory checks.",
    )
    args = parser.parse_args()
    if args.disable_system_monitoring:
        os.environ["MOWE_DISABLE_SYSTEM_MONITORING"] = "1"

    cfg = load_config(args.config)
    context = initialize_distributed(cfg)
    try:
        guard_cfg = cfg.get("training", {}).get("distributed", {})
        memory_fraction = float(
            args.memory_guard_fraction
            if args.memory_guard_fraction is not None
            else guard_cfg.get("memory_guard_fraction", 0.80)
        )
        gpu_fraction = float(
            args.gpu_memory_guard_fraction
            if args.gpu_memory_guard_fraction is not None
            else guard_cfg.get("gpu_memory_guard_fraction", 0.85)
        )
        if not 0 < memory_fraction <= 1 or not 0 < gpu_fraction <= 1:
            raise ValueError("Resource guard fractions must be in (0,1].")
        resources = process_resource_metrics(context)
        cgroup_monitoring_enabled = bool(
            guard_cfg.get("require_cgroup_metrics", True)
        ) and not args.disable_system_monitoring
        enforce_resource_metric_contract(
            context,
            resources,
            require_cgroup=cgroup_monitoring_enabled,
            require_gpu=context.device.startswith("cuda"),
        )
        enforce_cgroup_memory_guard(context, resources, memory_fraction)
        enforce_gpu_memory_guard(context, resources, gpu_fraction)
        local = {
            "rank": context.rank,
            "local_rank": context.local_rank,
            "device": context.device,
            "resources": resources,
            "runtime_identity": local_runtime_identity(context, resources),
        }
        ranks = context.all_gather_objects(local)
        runtime_identity = consolidate_runtime_identities(
            [value["runtime_identity"] for value in ranks],
            require_cuda=True,
            require_node_identity=cgroup_monitoring_enabled,
        )
        report = {
            "format": "flow_wam_ddp_runtime_audit_v1",
            "passed": True,
            "cgroup_monitoring_enabled": cgroup_monitoring_enabled,
            "resource_monitoring_mode": (
                "cgroup_and_gpu" if cgroup_monitoring_enabled else "gpu_only"
            ),
            "distributed_contract": distributed_contract(cfg, context),
            "ranks": sorted(ranks, key=lambda value: int(value["rank"])),
            "runtime_identity": runtime_identity,
        }
        expected = set(range(context.world_size))
        actual = {int(value["rank"]) for value in ranks}
        local_ranks = {int(value["local_rank"]) for value in ranks}
        report["checks"] = {
            "rank_union_complete": actual == expected,
            "local_ranks_unique": len(local_ranks) == context.world_size,
            "all_devices_bound": all(value["device"] != "auto" for value in ranks),
        }
        report["resource_guard_thresholds"] = {
            "cgroup_working_set_fraction": memory_fraction,
            "gpu_peak_allocated_fraction": gpu_fraction,
        }
        if not all(report["checks"].values()):
            raise RuntimeError(f"DDP runtime audit failed: {report['checks']}")
        if context.is_main:
            rendered = json.dumps(report, indent=2, sort_keys=True)
            if args.output:
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(rendered + "\n", encoding="utf-8")
            print(rendered, flush=True)
        context.barrier()
    finally:
        context.close()


if __name__ == "__main__":
    main()
