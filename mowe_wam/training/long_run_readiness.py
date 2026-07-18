"""Fail-closed audit for the formal single-node feature-store training path."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from mowe_wam.training.flow_runtime import (
    STAGE_PREDECESSOR,
    long_run_launch_contract,
    long_run_launch_fingerprint,
    readiness_checkpoint_identity,
    resolve_feature_store_contract,
    validate_checkpoint_contract,
    validate_flow_config,
    validate_resume_schedule_contract,
    validate_skill_audit_contract,
    validate_skill_config,
)
from mowe_wam.utils.config import load_config


DEFAULT_EQUIVALENCE_LIMITS = {
    "feature_atol": 0.03,
    "output_atol": 0.10,
    "loss_atol": 0.05,
}
DEFAULT_SOAK_LIMITS = {
    "max_anon_growth_mib": 512.0,
    "max_working_set_growth_mib": 2048.0,
    "max_anon_slope_mib_per_1k_steps": 64.0,
    "max_working_set_slope_mib_per_1k_steps": 256.0,
}


def _same_path(left: Any, right: Any) -> bool:
    if left is None or left == "" or right is None or right == "":
        return False
    return Path(str(left)).expanduser().resolve() == Path(str(right)).expanduser().resolve()


def _checkpoint_effective_batch(metadata: dict[str, Any]) -> int:
    contract = metadata.get("distributed_contract") or {}
    if contract.get("effective_global_batch") is not None:
        return int(contract["effective_global_batch"])
    training = metadata.get("config", {}).get("training", {})
    world_size = int(contract.get("world_size", 1))
    return (
        int(training.get("batch_size", 1))
        * int(training.get("grad_accumulation_steps", 1))
        * world_size
    )


def audit_long_run_readiness(
    cfg: dict[str, Any],
    *,
    store: str | Path,
    feature_audit: dict[str, Any],
    equivalence_report: dict[str, Any],
    soak_report: dict[str, Any],
    ddp_runtime_audit: dict[str, Any],
    world_size: int = 8,
    min_equivalence_samples: int = 100,
    min_soak_steps: int = 10_000,
    skill_expert_config: str | Path | None = None,
    checkpoint_metadata: dict[str, Any] | None = None,
    checkpoint_mode: str = "resume",
    allow_world_size_change: bool = False,
    allow_missing_cgroup_metrics: bool = False,
) -> dict[str, Any]:
    """Return a structured report and never silently waive a formal launch gate."""

    if world_size != 8:
        raise ValueError("The current formal launch contract requires exactly eight ranks.")
    if min_equivalence_samples < 1 or min_soak_steps < 1:
        raise ValueError("Readiness sample/step minimums must be positive.")
    if checkpoint_mode not in {"resume", "init"}:
        raise ValueError("checkpoint_mode must be resume or init.")

    resolved = copy.deepcopy(cfg)
    resolved.setdefault("data", {})["feature_store_path"] = str(store)
    if skill_expert_config is not None:
        resolved["skill_expert_config"] = str(skill_expert_config)
    checks: dict[str, bool] = {}
    errors: list[str] = []

    def check(name: str, condition: Any, message: str) -> None:
        passed = bool(condition)
        checks[name] = passed
        if not passed:
            errors.append(message)

    try:
        manifest = resolve_feature_store_contract(resolved)
        validate_flow_config(resolved)
        config_resolved = True
    except Exception as exc:
        manifest = {}
        config_resolved = False
        errors.append(f"config/store contract: {exc}")
    checks["config_store_contract"] = config_resolved

    data = resolved.get("data", {})
    training = resolved.get("training", {})
    distributed = training.get("distributed", {})
    validation = resolved.get("validation", {})
    sampler_shuffle_block_size = int(data.get("sampler_shuffle_block_size", 256))
    current_effective_batch = (
        int(training.get("batch_size", 0))
        * int(training.get("grad_accumulation_steps", 0))
        * int(world_size)
    )
    check(
        "feature_hot_path",
        data.get("backend") == "mowe_feature_store_v1"
        and resolved.get("backbone", {}).get("mode") == "precomputed_features"
        and not bool(data.get("image_aug", False))
        and int(data.get("num_workers", -1)) == 0
        and not bool(data.get("pin_memory", True)),
        "Formal long training must use precomputed mowe_feature_store_v1 with image_aug=false, "
        "num_workers=0, and pin_memory=false.",
    )
    system_monitoring_enabled = not allow_missing_cgroup_metrics
    check(
        "ddp_training_contract",
        str(distributed.get("enabled", "auto")).lower() in {"auto", "true", "1"}
        and distributed.get("backend") == "nccl"
        and (
            bool(distributed.get("require_cgroup_metrics", False))
            if system_monitoring_enabled
            else not bool(distributed.get("require_cgroup_metrics", True))
        )
        and str(training.get("precision", "")).lower() == "bf16"
        and int(training.get("batch_size", 0)) == 1
        and int(training.get("grad_accumulation_steps", 0)) == 1
        and int(training.get("max_steps", 0)) > 0
        and int(training.get("save_freq", 0)) > 0
        and int(training.get("log_freq", 0)) > 0,
        "Formal 8-GPU launch requires NCCL, BF16, per-device batch 1, accumulation 1, "
        "and fail-closed cgroup metrics.",
    )
    check(
        "resource_thresholds",
        (
            0 < float(distributed.get("memory_guard_fraction", 2.0)) <= 0.80
            if system_monitoring_enabled
            else True
        )
        and 0 < float(distributed.get("gpu_memory_guard_fraction", 2.0)) <= 0.85,
        "Configured GPU guard must be no looser than 85%; cgroup guard is required unless "
        "the launcher explicitly disabled system monitoring.",
    )
    check(
        "formal_store",
        bool(manifest.get("formal_training_ready", False))
        and manifest.get("completion_contract", {}).get("counts_match") is True
        and int(manifest.get("partition_counts", {}).get("train", 0)) > 0
        and (
            not bool(validation.get("enabled", False))
            or int(manifest.get("partition_counts", {}).get("validation", 0)) > 0
        ),
        "Feature store must be formal, match expected counts, and contain every enabled partition.",
    )

    assignment = feature_audit.get("assignment", {})
    assignment_reports = assignment.get("reports", [])
    configured_limits = assignment.get("configured_imbalance_limits", {})
    imbalance_checks = assignment.get("imbalance_checks", {})
    check(
        "feature_audit_identity",
        feature_audit.get("format") == "mowe_feature_store_v1"
        and _same_path(feature_audit.get("root"), store),
        "Feature audit does not describe the requested store.",
    )
    check(
        "feature_audit_integrity",
        bool(feature_audit.get("valid", False))
        and bool(feature_audit.get("formal_training_ready", False))
        and bool(feature_audit.get("checksums_verified", False)),
        "Feature audit must pass in formal mode with every shard checksum verified.",
    )
    check(
        "feature_assignment",
        int(assignment.get("world_size", 0)) == world_size
        and len(assignment_reports) == world_size
        and {int(item.get("rank", -1)) for item in assignment_reports}
        == set(range(world_size))
        and all(
            int(item.get("episode_count", 0)) > 0
            and int(item.get("window_count", 0)) > 0
            and item.get("order_strategy") == "shard_aware_block_shuffle_v1"
            and int(item.get("shuffle_block_size", 0))
            == sampler_shuffle_block_size
            for item in assignment_reports
        )
        and bool(assignment.get("episode_union_complete", False))
        and not assignment.get("episode_overlap")
        and bool(assignment.get("fingerprints_agree", False))
        and bool(assignment.get("target_skill_union_complete", False)),
        "Feature audit must prove complete, disjoint, fingerprint-consistent episode/skill assignment.",
    )
    check(
        "reviewed_imbalance_limits",
        all(configured_limits.get(name) is not None for name in ("windows", "suites", "skills"))
        and all(bool(imbalance_checks.get(name, False)) for name in ("windows", "suites", "skills")),
        "Re-run the feature audit with explicit human-reviewed window/suite/skill imbalance limits.",
    )

    equivalence_tolerances = equivalence_report.get("tolerances", {})
    source_contract = manifest.get("source_contract", {})
    source_dataset_names = set(
        str(value) for value in source_contract.get("dataset_names", [])
    )
    store_openvla_identity_sha256 = (
        source_contract.get("openvla_identity", {}).get("identity_sha256")
    )
    expected_benchmark = (
        "calvin_abc_d"
        if source_dataset_names == {"calvin_abc_language_segments"}
        else "libero"
        if source_dataset_names
        and all("libero" in value.lower() for value in source_dataset_names)
        else None
    )
    check(
        "equivalence_identity",
        equivalence_report.get("format") == "mowe_feature_store_equivalence_v1"
        and _same_path(equivalence_report.get("store"), store)
        and expected_benchmark is not None
        and equivalence_report.get("benchmark") == expected_benchmark
        and store_openvla_identity_sha256 is not None
        and equivalence_report.get("openvla_identity_sha256")
        == store_openvla_identity_sha256,
        "Equivalence report does not match the requested store, benchmark, and immutable "
        "OpenVLA identity.",
    )
    check(
        "equivalence_gate",
        bool(equivalence_report.get("passed", False))
        and int(equivalence_report.get("compared_samples", 0)) >= min_equivalence_samples
        and not equivalence_report.get("missing_pairs")
        and equivalence_report.get("comparison_contract", {}).get("name")
        == "mask_aware_training_metric_v1"
        and equivalence_report.get("masks_match") is True
        and float(equivalence_report.get("max_feature_gate_error", float("inf")))
        <= float(equivalence_tolerances.get("feature_atol", float("-inf")))
        and float(equivalence_report.get("max_output_gate_error", float("inf")))
        <= float(equivalence_tolerances.get("output_atol", float("-inf")))
        and float(equivalence_report.get("max_loss_gate_error", float("inf")))
        <= float(equivalence_tolerances.get("loss_atol", float("-inf")))
        and all(
            float(equivalence_tolerances.get(name, float("inf"))) <= maximum
            for name, maximum in DEFAULT_EQUIVALENCE_LIMITS.items()
        ),
        f"Equivalence must pass on at least {min_equivalence_samples} windows with matching "
        "history masks, the mask-aware comparison contract, and no looser tolerances.",
    )

    soak_limits = soak_report.get("limits", {})
    soak_ranks = soak_report.get("reports", [])
    soak_runtime_identity = soak_report.get("runtime_identity", {})
    check(
        "soak_identity",
        soak_report.get("format") == "mowe_feature_store_soak_v1"
        and _same_path(soak_report.get("store"), store),
        "Soak report does not describe the requested store.",
    )
    check(
        "soak_thresholds",
        (
            all(
                float(soak_limits.get(name, float("inf"))) <= maximum
                for name, maximum in DEFAULT_SOAK_LIMITS.items()
            )
            and int(soak_limits.get("min_post_warmup_samples", 0)) >= 3
            if system_monitoring_enabled
            else soak_report.get("cgroup_monitoring_enabled") is False
        ),
        "Soak thresholds are looser than the formal defaults, or the report does not declare "
        "the requested degraded system-monitoring mode.",
    )
    soak_rank_set = {int(item.get("rank", -1)) for item in soak_ranks}
    check(
        "soak_gate",
        bool(soak_report.get("passed", False))
        and len(soak_ranks) == world_size
        and soak_rank_set == set(range(world_size))
        and all(
            int(item.get("world_size", 0)) == world_size
            and int(item.get("steps", 0)) >= min_soak_steps
            and bool(item.get("passed", False))
            and item.get("sampler", {}).get("order_strategy")
            == "shard_aware_block_shuffle_v1"
            and int(item.get("sampler", {}).get("shuffle_block_size", 0))
            == sampler_shuffle_block_size
            and not bool(item.get("tensorflow_imported", True))
            and (
                (
                    not item.get("missing_required_metrics")
                    and item.get("anon_growth_mib") is not None
                    and item.get("working_set_growth_mib") is not None
                    and item.get("anon_slope_mib_per_1k_steps") is not None
                    and item.get("working_set_slope_mib_per_1k_steps") is not None
                    and int(item.get("cgroup_event_deltas", {}).get("cgroup_event_oom", 0)) == 0
                    and int(item.get("cgroup_event_deltas", {}).get("cgroup_event_oom_kill", 0)) == 0
                )
                if system_monitoring_enabled
                else item.get("cgroup_monitoring_enabled") is False
            )
            for item in soak_ranks
        ),
        f"CPU feature-store soak must pass on all {world_size} ranks for at least {min_soak_steps} steps.",
    )

    runtime_contract = ddp_runtime_audit.get("distributed_contract", {})
    runtime_checks = ddp_runtime_audit.get("checks", {})
    runtime_ranks = ddp_runtime_audit.get("ranks", [])
    runtime_thresholds = ddp_runtime_audit.get("resource_guard_thresholds", {})
    runtime_identity = ddp_runtime_audit.get("runtime_identity", {})
    check(
        "ddp_runtime_gate",
        ddp_runtime_audit.get("format") == "flow_wam_ddp_runtime_audit_v1"
        and int(runtime_contract.get("world_size", 0)) == world_size
        and int(runtime_contract.get("effective_global_batch", -1)) == current_effective_batch
        and len(runtime_ranks) == world_size
        and {int(item.get("rank", -1)) for item in runtime_ranks}
        == set(range(world_size))
        and all(
            bool(runtime_checks.get(name, False))
            for name in (
                "rank_union_complete",
                "local_ranks_unique",
                "all_devices_bound",
            )
        )
        and (
            float(runtime_thresholds.get("cgroup_working_set_fraction", 2.0)) <= 0.80
            if system_monitoring_enabled
            else ddp_runtime_audit.get("cgroup_monitoring_enabled") is False
        )
        and float(runtime_thresholds.get("gpu_peak_allocated_fraction", 2.0)) <= 0.85,
        "DDP runtime audit must prove all GPU ranks, effective batch, and 80%/85% resource guards.",
    )
    runtime_node = runtime_identity.get("node", {})
    soak_node = soak_runtime_identity.get("node", {})
    accelerators = runtime_identity.get("accelerators", [])
    required_node_fields = (
        "hostname",
        "boot_id",
        "cgroup_membership_sha256",
        "cgroup_memory_max_mib",
    )
    check(
        "runtime_environment_identity",
        soak_runtime_identity.get("rank_count") == world_size
        and runtime_identity.get("rank_count") == world_size
        and (
            (
                all(soak_node.get(name) not in {None, ""} for name in required_node_fields)
                and soak_node == runtime_node
            )
            if system_monitoring_enabled
            else (
                soak_node.get("system_monitoring_disabled") is True
                and runtime_node.get("system_monitoring_disabled") is True
            )
        )
        and len(accelerators) == world_size
        and {int(value.get("local_rank", -1)) for value in accelerators}
        == set(range(world_size))
        and all(
            value.get("name")
            and float(value.get("total_memory_mib", 0.0)) > 0
            and len(value.get("compute_capability", [])) == 2
            for value in accelerators
        ),
        "CPU soak and DDP audit must identify the same node/cgroup, unless the launcher "
        "explicitly used degraded system monitoring; both modes still require eight GPUs.",
    )

    skill_path = resolved.get("skill_expert_config")
    try:
        skill_cfg = load_config(skill_path)
        validate_skill_config(skill_cfg)
        if config_resolved:
            validate_skill_audit_contract(resolved, skill_cfg)
        resolved["skill_experts_resolved"] = skill_cfg
        skill_ok = config_resolved
    except Exception as exc:
        skill_ok = False
        errors.append(f"skill audit contract: {exc}")
    checks["skill_audit_contract"] = skill_ok

    stage = str(resolved.get("stage", "nominal_flow_pretrain"))
    if checkpoint_metadata is None:
        checkpoint_ok = stage == "nominal_flow_pretrain"
        if not checkpoint_ok:
            errors.append(f"Stage {stage!r} requires a predecessor or resume checkpoint audit.")
    else:
        expected_stage = stage if checkpoint_mode == "resume" else STAGE_PREDECESSOR.get(stage)
        checkpoint_ok = expected_stage is not None and checkpoint_metadata.get("stage") == expected_stage
        if not checkpoint_ok:
            errors.append(
                f"Checkpoint stage {checkpoint_metadata.get('stage')!r} does not match "
                f"{checkpoint_mode} expectation {expected_stage!r}."
            )
        checkpoint_contract = checkpoint_metadata.get("distributed_contract") or {}
        checkpoint_world_size = int(checkpoint_contract.get("world_size", 1))
        checkpoint_batch = _checkpoint_effective_batch(checkpoint_metadata)
        if checkpoint_world_size != world_size and not allow_world_size_change:
            checkpoint_ok = False
            errors.append(
                "Checkpoint world size differs; explicit --allow-world-size-change is required."
            )
        if checkpoint_batch != current_effective_batch:
            checkpoint_ok = False
            errors.append(
                f"Checkpoint effective batch {checkpoint_batch} differs from current "
                f"{current_effective_batch}."
            )
        if checkpoint_mode == "resume" and int(checkpoint_metadata.get("step", -1)) >= int(
            training.get("max_steps", 0)
        ):
            checkpoint_ok = False
            errors.append("Resume checkpoint is already at or beyond training.max_steps.")
        if checkpoint_ok and config_resolved and skill_ok:
            try:
                validate_checkpoint_contract(checkpoint_metadata, resolved)
                if checkpoint_mode == "resume":
                    validate_resume_schedule_contract(checkpoint_metadata, resolved)
            except Exception as exc:
                checkpoint_ok = False
                errors.append(f"checkpoint contract: {exc}")
    checks["checkpoint_contract"] = checkpoint_ok

    checkpoint_identity = readiness_checkpoint_identity(
        checkpoint_metadata,
        checkpoint_mode=checkpoint_mode if checkpoint_metadata is not None else "none",
    )
    launch_contract = long_run_launch_contract(
        resolved, stage=stage, world_size=world_size
    )
    return {
        "format": "flow_wam_long_run_readiness_v1",
        "store": str(Path(store).expanduser().resolve()),
        "stage": stage,
        "world_size": int(world_size),
        "system_monitoring_enabled": system_monitoring_enabled,
        "effective_global_batch": int(current_effective_batch),
        "launch_contract": launch_contract,
        "launch_contract_sha256": long_run_launch_fingerprint(
            resolved, stage=stage, world_size=world_size
        ),
        "checkpoint_identity": checkpoint_identity,
        "runtime_identity": runtime_identity,
        "minimums": {
            "equivalence_samples": int(min_equivalence_samples),
            "soak_steps_per_rank": int(min_soak_steps),
        },
        "checks": checks,
        "errors": errors,
        "passed": all(checks.values()) and not errors,
    }


def load_json_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload
