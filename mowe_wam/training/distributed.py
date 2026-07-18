"""Single-node distributed helpers for the Flow-WAM training runtime."""

from __future__ import annotations

import hashlib
import json
import os
import socket
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from mowe_wam.utils.optional import require_torch


def system_monitoring_enabled() -> bool:
    """Whether this process may inspect host/container resource files.

    Managed training platforms sometimes intentionally hide ``/proc`` and the
    cgroup hierarchy.  CUDA/NCCL monitoring remains available in that mode;
    only host/container resource inspection is disabled.
    """

    return os.environ.get("MOWE_DISABLE_SYSTEM_MONITORING", "").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class DistributedContext:
    """Resolved torchrun process identity with a single-process fallback."""

    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    backend: str
    device: str
    initialized_here: bool = False

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def barrier(self) -> None:
        if self.enabled:
            torch = require_torch()
            torch.distributed.barrier()

    def all_gather_objects(self, value: Any) -> list[Any]:
        if not self.enabled:
            return [value]
        torch = require_torch()
        gathered: list[Any] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered, value)
        return gathered

    def broadcast_object(self, value: Any, *, source_rank: int = 0) -> Any:
        if not self.enabled:
            return value
        torch = require_torch()
        payload = [value if self.rank == source_rank else None]
        torch.distributed.broadcast_object_list(payload, src=source_rank)
        return payload[0]

    def close(self) -> None:
        if not self.enabled or not self.initialized_here:
            return
        torch = require_torch()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def _distributed_requested(value: Any, world_size: int) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto":
            return world_size > 1
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"Unsupported training.distributed.enabled value: {value!r}")
    return bool(value)


def initialize_distributed(cfg: dict[str, Any]) -> DistributedContext:
    """Initialize a torchrun process group and bind the process to one GPU."""

    torch = require_torch()
    dist_cfg = cfg.setdefault("training", {}).setdefault("distributed", {})
    env_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    requested = _distributed_requested(dist_cfg.get("enabled", "auto"), env_world_size)
    if env_world_size > 1 and not requested:
        raise RuntimeError(
            "torchrun launched multiple processes but training.distributed.enabled is false."
        )
    if requested and env_world_size <= 1:
        raise RuntimeError(
            "Distributed training was explicitly enabled but WORLD_SIZE is 1; launch with torchrun."
        )

    if not requested:
        requested_device = str(cfg.get("training", {}).get("device", "auto"))
        device = (
            "cuda" if requested_device == "auto" and torch.cuda.is_available() else
            "cpu" if requested_device == "auto" else requested_device
        )
        return DistributedContext(False, 0, 0, 1, "none", device)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    backend = str(dist_cfg.get("backend", "nccl")).lower()
    if backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA.")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} exceeds visible CUDA device count {torch.cuda.device_count()}."
            )
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    elif backend == "gloo":
        device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda"):
            torch.cuda.set_device(local_rank)
    else:
        raise ValueError(f"Unsupported distributed backend: {backend!r}")

    initialized_here = False
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=int(dist_cfg.get("timeout_seconds", 1800))),
        )
        initialized_here = True
    actual_rank = torch.distributed.get_rank()
    actual_world_size = torch.distributed.get_world_size()
    if actual_rank != rank or actual_world_size != env_world_size:
        raise RuntimeError(
            "torch.distributed identity differs from torchrun environment: "
            f"rank {actual_rank}/{rank}, world_size {actual_world_size}/{env_world_size}."
        )
    return DistributedContext(
        True,
        rank,
        local_rank,
        actual_world_size,
        backend,
        device,
        initialized_here,
    )


def effective_global_batch(cfg: dict[str, Any], world_size: int) -> int:
    training = cfg.get("training", {})
    return (
        int(training.get("batch_size", 1))
        * int(training.get("grad_accumulation_steps", 1))
        * int(world_size)
    )


def _collective_device(context: DistributedContext) -> str:
    # NCCL requires CUDA tensors; Gloo control-plane reductions are safest on CPU
    # even when the model itself is running on a visible CUDA device.
    return context.device if context.backend == "nccl" else "cpu"


def distributed_contract(cfg: dict[str, Any], context: DistributedContext) -> dict[str, Any]:
    training = cfg.get("training", {})
    return {
        "enabled": bool(context.enabled),
        "backend": context.backend,
        "world_size": int(context.world_size),
        "per_device_batch_size": int(training.get("batch_size", 1)),
        "grad_accumulation_steps": int(training.get("grad_accumulation_steps", 1)),
        "effective_global_batch": effective_global_batch(cfg, context.world_size),
        "broadcast_buffers": bool(
            training.get("distributed", {}).get("broadcast_buffers", False)
        ),
        "find_unused_parameters": bool(
            training.get("distributed", {}).get("find_unused_parameters", False)
        ),
    }


def local_rng_state(context: DistributedContext) -> dict[str, Any]:
    torch = require_torch()
    cuda_state = None
    if context.device.startswith("cuda"):
        cuda_state = torch.cuda.get_rng_state(context.local_rank)
    import random

    return {
        "rank": int(context.rank),
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": cuda_state,
    }


def restore_local_rng_state(state: dict[str, Any], context: DistributedContext) -> None:
    torch = require_torch()
    import random

    random.setstate(state["python_rng_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if context.device.startswith("cuda") and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], context.local_rank)


def _parse_cgroup_key_values(text: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in str(text).splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1])
        except ValueError:
            continue
    return values


def process_resource_metrics(context: DistributedContext) -> dict[str, Any]:
    """Return cheap per-rank RSS/GPU and shared cgroup memory observations."""

    torch = require_torch()
    metrics: dict[str, Any] = {"rank": int(context.rank)}
    monitor_system = system_monitoring_enabled()
    if monitor_system:
        status = Path("/proc/self/status")
        if status.exists():
            for line in status.read_text(encoding="utf-8").splitlines():
                if line.startswith("VmRSS:"):
                    metrics["process_rss_mib"] = int(line.split()[1]) / 1024.0
                    break
    if context.device.startswith("cuda"):
        metrics.update(
            {
                "cuda_allocated_mib": torch.cuda.memory_allocated(context.local_rank) / 2**20,
                "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated(context.local_rank) / 2**20,
                "cuda_reserved_mib": torch.cuda.memory_reserved(context.local_rank) / 2**20,
                "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved(context.local_rank) / 2**20,
                "cuda_total_mib": torch.cuda.get_device_properties(context.local_rank).total_memory / 2**20,
            }
        )
    if not monitor_system:
        return metrics
    cgroup_candidates = {
        "cgroup_memory_current_mib": [
            Path("/sys/fs/cgroup/memory.current"),
            Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
        ],
        "cgroup_memory_max_mib": [
            Path("/sys/fs/cgroup/memory.max"),
            Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
        ],
    }
    for name, candidates in cgroup_candidates.items():
        for path in candidates:
            if not path.exists():
                continue
            raw = path.read_text(encoding="utf-8").strip()
            if raw != "max":
                metrics[name] = int(raw) / 2**20
            break
    stat_path = next(
        (
            path
            for path in (
                Path("/sys/fs/cgroup/memory.stat"),
                Path("/sys/fs/cgroup/memory/memory.stat"),
            )
            if path.exists()
        ),
        None,
    )
    if stat_path is not None:
        values = _parse_cgroup_key_values(stat_path.read_text(encoding="utf-8"))
        # cgroup v1 commonly prefixes hierarchical totals with ``total_``.
        for name in ("anon", "file", "inactive_file", "slab_reclaimable"):
            raw = values.get(name, values.get(f"total_{name}"))
            if raw is not None:
                metrics[f"cgroup_memory_{name}_mib"] = raw / 2**20
    events_path = Path("/sys/fs/cgroup/memory.events")
    if events_path.exists():
        events = _parse_cgroup_key_values(events_path.read_text(encoding="utf-8"))
        for name in ("high", "oom", "oom_kill"):
            if name in events:
                metrics[f"cgroup_event_{name}"] = int(events[name])
    else:
        failcnt_path = Path("/sys/fs/cgroup/memory/memory.failcnt")
        if failcnt_path.exists():
            metrics["cgroup_event_oom"] = int(
                failcnt_path.read_text(encoding="utf-8").strip()
            )
    current = metrics.get("cgroup_memory_current_mib")
    inactive_file = metrics.get("cgroup_memory_inactive_file_mib")
    if current is not None and inactive_file is not None:
        metrics["cgroup_memory_working_set_mib"] = max(
            0.0, float(current) - float(inactive_file)
        )
    return metrics


def local_runtime_identity(
    context: DistributedContext, metrics: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Describe the current container boot/cgroup and rank-local accelerator."""

    torch = require_torch()
    metrics = metrics or process_resource_metrics(context)
    if system_monitoring_enabled():
        boot_id_path = Path("/proc/sys/kernel/random/boot_id")
        cgroup_path = Path("/proc/self/cgroup")
        boot_id = (
            boot_id_path.read_text(encoding="utf-8").strip()
            if boot_id_path.exists()
            else None
        )
        cgroup_membership = (
            cgroup_path.read_text(encoding="utf-8").strip()
            if cgroup_path.exists()
            else None
        )
        cgroup_hash = (
            hashlib.sha256(cgroup_membership.encode("utf-8")).hexdigest()
            if cgroup_membership
            else None
        )
        node = {
            "hostname": socket.gethostname(),
            "boot_id": boot_id,
            "cgroup_membership_sha256": cgroup_hash,
            "cgroup_memory_max_mib": metrics.get("cgroup_memory_max_mib"),
        }
    else:
        node = {"system_monitoring_disabled": True}
    accelerator = None
    if context.device.startswith("cuda"):
        properties = torch.cuda.get_device_properties(context.local_rank)
        accelerator = {
            "local_rank": int(context.local_rank),
            "device": str(context.device),
            "name": str(properties.name),
            "total_memory_mib": float(properties.total_memory / 2**20),
            "compute_capability": [int(properties.major), int(properties.minor)],
        }
    return {
        "rank": int(context.rank),
        "node": node,
        "accelerator": accelerator,
    }


def consolidate_runtime_identities(
    identities: list[dict[str, Any]], *, require_cuda: bool, require_node_identity: bool = True
) -> dict[str, Any]:
    """Prove every rank belongs to one container/node and expose GPU topology."""

    if not identities:
        raise ValueError("Runtime identity requires at least one rank.")
    ordered = sorted(identities, key=lambda value: int(value.get("rank", -1)))
    expected_ranks = set(range(len(ordered)))
    observed_ranks = {int(value.get("rank", -1)) for value in ordered}
    if observed_ranks != expected_ranks:
        raise ValueError(
            f"Runtime identity ranks are incomplete: {observed_ranks} != {expected_ranks}."
        )
    nodes = {
        json.dumps(value.get("node"), sort_keys=True, separators=(",", ":"))
        for value in ordered
    }
    if len(nodes) != 1:
        raise ValueError("Runtime ranks do not share one node/boot/cgroup identity.")
    node = ordered[0].get("node") or {}
    missing_node_fields = [
        name
        for name in (
            "hostname",
            "boot_id",
            "cgroup_membership_sha256",
            "cgroup_memory_max_mib",
        )
        if node.get(name) in {None, ""}
    ]
    if require_node_identity and missing_node_fields:
        raise ValueError(f"Runtime node identity is missing {missing_node_fields}.")
    accelerators = [value.get("accelerator") for value in ordered]
    if require_cuda:
        if any(value is None for value in accelerators):
            raise ValueError("CUDA runtime identity is missing one or more accelerators.")
        local_ranks = {int(value["local_rank"]) for value in accelerators}
        if local_ranks != expected_ranks:
            raise ValueError("CUDA runtime identity does not bind one unique GPU per rank.")
    return {
        "node": node,
        "rank_count": len(ordered),
        "accelerators": accelerators if require_cuda else [],
    }


def enforce_resource_metric_contract(
    context: DistributedContext,
    metrics: dict[str, Any],
    *,
    require_cgroup: bool,
    require_gpu: bool,
) -> None:
    """Fail closed when a requested safety limit cannot actually be measured."""

    missing = []
    if require_cgroup:
        for name in (
            "cgroup_memory_current_mib",
            "cgroup_memory_max_mib",
            "cgroup_memory_working_set_mib",
            "cgroup_event_oom",
            "cgroup_event_oom_kill",
        ):
            if metrics.get(name) is None:
                missing.append(name)
    if require_gpu:
        for name in (
            "cuda_total_mib",
            "cuda_peak_allocated_mib",
            "cuda_peak_reserved_mib",
        ):
            if metrics.get(name) is None:
                missing.append(name)
    local_missing = sorted(set(missing))
    gathered = context.all_gather_objects(local_missing)
    missing_by_rank = {
        rank: values for rank, values in enumerate(gathered) if values
    }
    if missing_by_rank:
        raise RuntimeError(
            "Distributed resource guard cannot be enforced because metrics are missing: "
            f"{missing_by_rank}."
        )


def enforce_cgroup_memory_guard(
    context: DistributedContext,
    metrics: dict[str, Any],
    fraction: float,
) -> None:
    current = metrics.get(
        "cgroup_memory_working_set_mib", metrics.get("cgroup_memory_current_mib")
    )
    maximum = metrics.get("cgroup_memory_max_mib")
    if current is None or maximum in {None, 0}:
        return
    ratio = float(current) / float(maximum)
    torch = require_torch()
    if context.enabled:
        device = _collective_device(context)
        value = torch.tensor(ratio, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        ratio = float(value.cpu())
    if ratio >= float(fraction):
        raise RuntimeError(
            "Cgroup working-set memory guard triggered at "
            f"{ratio:.1%}; configured limit is {float(fraction):.1%}."
        )


def enforce_gpu_memory_guard(
    context: DistributedContext,
    metrics: dict[str, Any],
    fraction: float,
) -> None:
    """Fail when any rank's peak allocated GPU memory reaches the safety limit."""

    observed_peaks = [
        float(metrics[name])
        for name in ("cuda_peak_allocated_mib", "cuda_peak_reserved_mib")
        if metrics.get(name) is not None
    ]
    peak = max(observed_peaks) if observed_peaks else None
    total = metrics.get("cuda_total_mib")
    if peak is None or total in {None, 0}:
        return
    ratio = float(peak) / float(total)
    torch = require_torch()
    if context.enabled:
        device = _collective_device(context)
        value = torch.tensor(ratio, dtype=torch.float64, device=device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        ratio = float(value.cpu())
    if ratio >= float(fraction):
        raise RuntimeError(
            "GPU peak allocated-memory guard triggered at "
            f"{ratio:.1%}; configured limit is {float(fraction):.1%}."
        )


def enforce_no_new_oom_events(
    context: DistributedContext,
    metrics: dict[str, Any],
    baseline: dict[str, Any],
) -> None:
    """Fail if the shared cgroup reports an OOM/OOM-kill after launch."""

    local_delta = max(
        int(metrics.get(name, 0)) - int(baseline.get(name, 0))
        for name in ("cgroup_event_oom", "cgroup_event_oom_kill")
    )
    torch = require_torch()
    if context.enabled:
        device = _collective_device(context)
        value = torch.tensor(local_delta, dtype=torch.int64, device=device)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
        local_delta = int(value.cpu())
    if local_delta > 0:
        raise RuntimeError(
            f"Cgroup reported {local_delta} new OOM/OOM-kill event(s) after training launch."
        )
