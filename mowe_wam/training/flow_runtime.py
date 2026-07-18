"""Three-stage runtime for the flow-WAM temporal skill policy."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from mowe_wam.backbones import (
    OpenVLAContextAdapter,
    PrecomputedFeatureBackbone,
    VisualTargetEncoder,
    openvla_identities_match,
    resolve_original_openvla_identity,
    teacher_transform_metadata,
    validate_openvla_identity,
)
from mowe_wam.data import (
    CACHE_FORMAT,
    EpisodeAwareDistributedSampler,
    LatentWAMCollator,
    LiberoSequenceDataset,
    LABEL_VERSION,
    SKILL_NAMES,
    SKILL_TO_ID,
    MoWEFeatureWindowDataset,
    ShardedVisualTargetCache,
    load_feature_store_manifest,
    validate_episode_assignment_reports,
    rlds_manifest_fingerprint,
    validate_visual_cache_metadata,
)
from mowe_wam.memory import MultiScaleMemoryEncoder
from mowe_wam.models import (
    FlowWAMSkillPolicy,
    FutureGroundedRouter,
    LatentWorldModel,
    LanguageConditionedViewFusion,
    NominalActionHead,
    ResidualFlowExperts,
    first_skill_segment_steps,
)
from mowe_wam.training.latent_losses import flow_wam_skill_losses
from mowe_wam.training.distributed import (
    consolidate_runtime_identities,
    DistributedContext,
    distributed_contract,
    effective_global_batch,
    enforce_cgroup_memory_guard,
    enforce_gpu_memory_guard,
    enforce_no_new_oom_events,
    enforce_resource_metric_contract,
    initialize_distributed,
    local_runtime_identity,
    local_rng_state,
    process_resource_metrics,
    restore_local_rng_state,
)
from mowe_wam.training.schedules import teacher_forcing_probability, temporal_router_schedule
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


FLOW_COMPONENTS = (
    "view_fusion",
    "memory_encoder",
    "nominal_action_head",
    "world_model",
    "router",
    "residual_experts",
    "expert_context",
)

STAGE_PREDECESSOR = {
    "expert_warmstart": "nominal_flow_pretrain",
    "joint": "expert_warmstart",
}


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def set_seed(seed: int, rank: int = 0) -> None:
    torch = require_torch()
    resolved = int(seed) + int(rank)
    random.seed(resolved)
    torch.manual_seed(resolved)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(resolved)


def resolve_device(cfg: dict[str, Any]) -> str:
    torch = require_torch()
    requested = str(cfg.get("training", {}).get("device", "auto"))
    return ("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested


def autocast_context(device: str, precision: str):
    torch = require_torch()
    if not device.startswith("cuda") or precision.lower() in {"fp32", "float32"}:
        return nullcontext()
    dtype = torch.bfloat16 if precision.lower() in {"bf16", "bfloat16"} else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def make_grad_scaler(device: str, precision: str):
    torch = require_torch()
    enabled = device.startswith("cuda") and precision.lower() in {"fp16", "float16"}
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def validate_flow_config(cfg: dict[str, Any]) -> None:
    if cfg.get("model", {}).get("variant") != "flow_wam_skill_moe":
        raise ValueError("Flow entrypoints require model.variant=flow_wam_skill_moe.")
    if not bool(cfg.get("backbone", {}).get("freeze_backbone", True)):
        raise ValueError("Flow-WAM v1 keeps the OpenVLA backbone frozen.")
    backbone_cfg = cfg.get("backbone", {})
    backbone_mode = str(backbone_cfg.get("mode", "online_openvla"))
    feature_source = backbone_cfg.get("feature_source")
    if backbone_mode == "online_openvla":
        if feature_source != "pre_action_context":
            raise ValueError("Online Flow-WAM requires the non-leaky pre_action_context feature source.")
    elif backbone_mode == "precomputed_features":
        if feature_source != "pre_action_context_cache":
            raise ValueError(
                "Precomputed Flow-WAM requires feature_source=pre_action_context_cache."
            )
        if cfg.get("data", {}).get("backend") != "mowe_feature_store_v1":
            raise ValueError("precomputed_features requires data.backend=mowe_feature_store_v1.")
        if bool(cfg.get("data", {}).get("image_aug", False)):
            raise ValueError("Precomputed features are invalid when image augmentation is enabled.")
    else:
        raise ValueError(f"Unsupported backbone.mode: {backbone_mode!r}.")
    if bool(cfg.get("teacher", {}).get("inference_enabled", False)):
        raise ValueError("The visual teacher must be disabled for deployment and is training-target-only.")
    if int(cfg.get("backbone", {}).get("num_images_in_input", 0)) != 2:
        raise ValueError("Flow-WAM dual-view v2 requires backbone.num_images_in_input=2.")
    if cfg.get("data", {}).get("observation_views") != ["primary", "wrist"]:
        raise ValueError("Flow-WAM dual-view v2 requires ordered data views [primary, wrist].")
    if cfg.get("teacher", {}).get("target_views") != ["primary"]:
        raise ValueError("Flow-WAM v2 keeps DINO future supervision primary-only.")
    view_cfg = cfg.get("view_fusion", {})
    if (
        view_cfg.get("type") != "language_conditioned_scalar_attention"
        or int(view_cfg.get("num_views", 0)) != 2
        or view_cfg.get("view_order") != ["primary", "wrist"]
    ):
        raise ValueError("Invalid dual-view fusion contract.")
    data = cfg["data"]
    flow = cfg["flow"]
    router = cfg["router"]
    experts = cfg["experts"]
    expected = {
        "action_dim": (int(data.get("action_dim", 7)), 7),
        "motion_dim": (int(data.get("motion_dim", 6)), 6),
        "gripper_dim": (int(data.get("gripper_dim", 1)), 1),
        "action_chunk_size": (int(data.get("action_chunk_size", 16)), 16),
        "num_routes": (int(router.get("num_routes", 7)), 7),
        "num_motor_experts": (int(experts.get("num_motor_experts", 6)), 6),
        "null_route": (int(router.get("null_route", 6)), 6),
    }
    invalid = {name: value for name, (value, required) in expected.items() if value != required}
    if invalid:
        raise ValueError(f"Flow-WAM synchronous-16 contract mismatch: {invalid}")
    if data.get("future_horizons") != [1, 4, 8, 16]:
        raise ValueError("Flow-WAM synchronous-16 requires data.future_horizons=[1,4,8,16].")
    if int(router.get("schedule_length", 0)) != 16:
        raise ValueError("Flow-WAM synchronous-16 requires router.schedule_length=16.")
    if flow.get("formulation") != "conditional_rectified_flow" or flow.get("solver") != "euler":
        raise ValueError("Flow-WAM v1 supports conditional_rectified_flow with the Euler solver.")
    if data.get("boundary_label_policy") != "direct_per_timestep_no_extra_mask":
        raise ValueError("Flow-WAM v1 requires direct per-timestep boundary labels.")
    if int(flow.get("num_inference_steps", 0)) < 1:
        raise ValueError("flow.num_inference_steps must be positive.")
    max_residual_l2 = float(experts.get("max_residual_l2", 0.0))
    if not 0.0 < max_residual_l2 <= 1.0:
        raise ValueError("experts.max_residual_l2 must be within (0,1] normalized motion units.")
    execution = cfg["execution"]
    if execution.get("mode") != "synchronous_risk_gated":
        raise ValueError("Flow-WAM execution.mode must be synchronous_risk_gated.")
    required_execution = {
        "prediction_horizon": 16,
        "min_steps": 1,
        "caution_steps": 4,
        "default_steps": 8,
        "max_steps": 8,
    }
    mismatched_execution = {
        key: execution.get(key)
        for key, required in required_execution.items()
        if int(execution.get(key, -1)) != required
    }
    if mismatched_execution:
        raise ValueError(
            "Flow-WAM synchronous execution contract mismatch: "
            f"{mismatched_execution}."
        )
    required_flags = {
        "allow_confident_skill_boundary_crossing": True,
        "discard_unexecuted_suffix": True,
        "stop_before_first_skill_change": False,
        "async_generation": False,
        "reuse_previous_tail": False,
    }
    mismatched_flags = {
        key: execution.get(key)
        for key, required in required_flags.items()
        if execution.get(key) is not required
    }
    if mismatched_flags:
        raise ValueError(f"Flow-WAM execution flags mismatch: {mismatched_flags}.")
    risk = execution.get("risk", {})
    ordered_pairs = (
        ("normalized_entropy_caution", "normalized_entropy_high", 0.0, 1.0),
        ("motion_jump_l2_caution", "motion_jump_l2_high", 0.0, float("inf")),
        ("residual_l2_caution", "residual_l2_high", 0.0, float("inf")),
    )
    for caution_key, high_key, lower, upper in ordered_pairs:
        caution = float(risk.get(caution_key, -1.0))
        high = float(risk.get(high_key, -1.0))
        if not lower <= caution < high <= upper:
            raise ValueError(f"Invalid ordered execution risk thresholds: {caution_key}, {high_key}.")
    margin_caution = float(risk.get("top2_margin_caution", -1.0))
    margin_high = float(risk.get("top2_margin_high", -1.0))
    if not 0.0 <= margin_high < margin_caution <= 1.0:
        raise ValueError("Invalid top-2 margin execution risk thresholds.")
    if data.get("backend") == "mowe_feature_store_v1":
        max_unattested_steps = int(
            cfg.get("long_run_readiness", {}).get("max_unattested_steps", 100)
        )
        if not 1 <= max_unattested_steps <= 100:
            raise ValueError(
                "Feature-store max_unattested_steps must remain within [1,100]; "
                "longer launches require a matching readiness report."
            )


def validate_skill_config(skill_cfg: dict[str, Any]) -> None:
    if skill_cfg.get("label_version") != LABEL_VERSION:
        raise ValueError(
            f"Skill label version mismatch: {skill_cfg.get('label_version')!r} != {LABEL_VERSION!r}."
        )
    if skill_cfg.get("skills") != SKILL_TO_ID:
        raise ValueError(f"Skill taxonomy mismatch: {skill_cfg.get('skills')} != {SKILL_TO_ID}.")
    if int(skill_cfg.get("unknown_label", -999)) != -1 or int(skill_cfg.get("null_route", -1)) != 6:
        raise ValueError("Skill config must use unknown=-1 and null_finish route 6.")
    weights = skill_cfg.get("class_weights_inverse_sqrt")
    if not isinstance(weights, list) or len(weights) != 7 or any(float(value) <= 0 for value in weights):
        raise ValueError("class_weights_inverse_sqrt must contain seven positive values.")


def validate_skill_audit_contract(cfg: dict[str, Any], skill_cfg: dict[str, Any]) -> None:
    audit = skill_cfg.get("audit")
    if not isinstance(audit, dict):
        raise ValueError("Skill config must embed the audited data/sidecar contract.")
    if cfg.get("data", {}).get("backend", "rlds") == "mowe_feature_store_v1":
        current_dataset = cfg["data"].get("source_rlds_manifest_fingerprint")
        if not current_dataset:
            raise ValueError("Feature store does not record its source dataset fingerprint.")
    else:
        current_dataset = rlds_manifest_fingerprint(
            cfg["data"]["data_root"], cfg["data"]["dataset_names"]
        )
    if audit.get("dataset_manifest_fingerprint_sha256") != current_dataset:
        raise ValueError(
            "Skill audit dataset fingerprint does not match the current source dataset; rerun "
            "the benchmark-specific data audit before training."
        )
    sidecar_metadata = cfg["data"].get("skill_sidecar_metadata") or {}
    if audit.get("sidecar_fingerprint_sha256") != sidecar_metadata.get("fingerprint_sha256"):
        raise ValueError(
            "Skill audit sidecar fingerprint does not match the joined annotation file; rerun the audit."
        )
    if bool(audit.get("alignment_verified", True)):
        raise ValueError("The v1 skill audit must preserve alignment_verified=false as an explicit assumption.")
    counts = audit.get("label_counts", {})
    if sum(int(value) for value in counts.values()) != int(audit.get("transitions", -1)):
        raise ValueError("Skill audit label counts do not sum to the recorded transition count.")


def build_flow_policy(cfg: dict[str, Any], *, include_teacher: bool = True, backbone=None):
    validate_flow_config(cfg)
    device = resolve_device(cfg)
    if backbone is None:
        backbone_cfg = dict(cfg["backbone"])
        backbone_mode = backbone_cfg.pop("mode", "online_openvla")
        backbone_cfg["device"] = device
        if backbone_mode == "precomputed_features":
            backbone = PrecomputedFeatureBackbone(
                hidden_dim=int(backbone_cfg.pop("hidden_dim")),
                device=device,
                dtype=backbone_cfg.get("dtype", cfg["training"].get("precision", "bf16")),
                num_images_in_input=int(backbone_cfg.get("num_images_in_input", 2)),
            )
        else:
            backbone_cfg.pop("hidden_dim", None)
            backbone = OpenVLAContextAdapter(**backbone_cfg)
    context_dim = int(backbone.hidden_dim)
    data_cfg = cfg["data"]
    memory_cfg = cfg["memory"]
    flow_cfg = cfg["flow"]
    world_cfg = cfg["world_model"]
    router_cfg = cfg["router"]
    expert_cfg = cfg["experts"]
    hidden_dim = int(flow_cfg.get("hidden_dim", 512))
    memory_dim = int(memory_cfg.get("hidden_dim", 512))
    chunk_size = int(data_cfg.get("action_chunk_size", 16))

    teacher = None
    if include_teacher and cfg.get("backbone", {}).get("mode", "online_openvla") != "precomputed_features":
        teacher_cfg = cfg["teacher"]
        teacher = VisualTargetEncoder(
            checkpoint=teacher_cfg.get("checkpoint", "facebook/dinov2-small"),
            spatial_grid=int(teacher_cfg.get("spatial_grid", 4)),
            target_dim=int(teacher_cfg.get("target_dim", 384)),
            num_spatial_tokens=int(teacher_cfg.get("spatial_tokens", 16)),
            device=device,
            dtype=teacher_cfg.get("dtype", cfg["training"].get("precision", "bf16")),
        )
    memory = MultiScaleMemoryEncoder(
        visual_dim=context_dim,
        language_dim=context_dim,
        action_dim=7,
        hidden_dim=memory_dim,
        max_short_tokens=int(data_cfg.get("history_length", 8)),
        max_long_tokens=int(data_cfg.get("long_memory_slots", 4)),
        heads=int(memory_cfg.get("heads", 8)),
        dropout=float(memory_cfg.get("dropout", 0.0)),
    )
    view_cfg = cfg["view_fusion"]
    view_fusion = LanguageConditionedViewFusion(
        feature_dim=context_dim,
        language_dim=context_dim,
        hidden_dim=int(view_cfg.get("score_hidden_dim", 128)),
        num_views=int(view_cfg.get("num_views", 2)),
        view_order=tuple(view_cfg.get("view_order", ["primary", "wrist"])),
    )
    nominal = NominalActionHead(
        context_dim=context_dim,
        memory_dim=memory_dim,
        hidden_dim=hidden_dim,
        motion_dim=6,
        chunk_size=chunk_size,
        flow_depth=int(flow_cfg.get("depth", 3)),
        dropout=float(flow_cfg.get("dropout", 0.0)),
    )
    world = LatentWorldModel(
        context_dim=context_dim,
        memory_dim=memory_dim,
        action_dim=7,
        action_chunk_size=chunk_size,
        future_horizons=data_cfg.get("future_horizons", [1, 4, 8, 16]),
        hidden_dim=int(world_cfg.get("hidden_dim", 512)),
        route_world_dim=int(world_cfg.get("route_world_dim", 128)),
        layers=int(world_cfg.get("layers", 6)),
        heads=int(world_cfg.get("heads", 8)),
        mlp_ratio=int(world_cfg.get("mlp_ratio", 4)),
        target_tokens=int(cfg["teacher"].get("spatial_tokens", 16)),
        target_dim=int(cfg["teacher"].get("target_dim", 384)),
        dropout=float(world_cfg.get("dropout", 0.0)),
        predict_uncertainty=bool(world_cfg.get("predict_uncertainty", False)),
    )
    router = FutureGroundedRouter(
        world_dim=int(world_cfg.get("hidden_dim", 512)),
        memory_dim=memory_dim,
        latent_dim=int(cfg["teacher"].get("target_dim", 384)),
        route_world_dim=int(world_cfg.get("route_world_dim", 128)),
        action_dim=7,
        hidden_dim=int(router_cfg.get("hidden_dim", 256)),
        num_routes=int(router_cfg.get("num_routes", 7)),
        chunk_size=chunk_size,
        null_route=int(router_cfg.get("null_route", 6)),
        use_uncertainty=bool(world_cfg.get("predict_uncertainty", False)),
    )
    # This trunk is shared by all six residual heads.  It intentionally does
    # not share weights with the frozen nominal proposal in Stage 2.
    experts = ResidualFlowExperts(
        condition_dim=hidden_dim,
        hidden_dim=hidden_dim,
        motion_dim=6,
        chunk_size=chunk_size,
        num_motor_experts=int(expert_cfg.get("num_motor_experts", 6)),
        num_routes=int(expert_cfg.get("num_routes", 7)),
        null_route=int(expert_cfg.get("null_route", 6)),
        flow_depth=int(flow_cfg.get("depth", 3)),
        dropout=float(flow_cfg.get("dropout", 0.0)),
    )
    model = FlowWAMSkillPolicy(
        backbone=backbone,
        memory_encoder=memory,
        nominal_action_head=nominal,
        world_model=world,
        router=router,
        residual_experts=experts,
        view_fusion=view_fusion,
        visual_teacher=teacher,
        context_dim=context_dim,
        memory_dim=memory_dim,
        world_dim=int(world_cfg.get("hidden_dim", 512)),
        expert_condition_dim=hidden_dim,
        flow_steps=int(flow_cfg.get("num_inference_steps", 4)),
        execution_config=cfg["execution"],
        action_distance_beta=float(cfg["action_condition"].get("distance_beta", 2.0)),
        max_residual_l2=float(expert_cfg.get("max_residual_l2", 0.5)),
        ablation=cfg.get("ablation"),
    )
    return model.to(device)


def _feature_store_root(cfg: dict[str, Any]) -> Path:
    data_cfg = cfg.get("data", {})
    root = data_cfg.get("feature_store_path") or data_cfg.get("data_root")
    if not root:
        raise ValueError("mowe_feature_store_v1 requires data.feature_store_path or data.data_root.")
    return Path(root)


def resolve_feature_store_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    """Bind model/data dimensions and source fingerprints to one store manifest."""

    manifest = load_feature_store_manifest(_feature_store_root(cfg))
    feature = manifest["feature_contract"]
    window = manifest["window_contract"]
    data_cfg = cfg["data"]
    expected_window = {
        "history_length": int(data_cfg.get("history_length", 8)),
        "long_memory_slots": int(data_cfg.get("long_memory_slots", 4)),
        "future_horizons": [
            int(value) for value in data_cfg.get("future_horizons", [1, 4, 8, 16])
        ],
        "action_chunk_size": int(data_cfg.get("action_chunk_size", 16)),
    }
    if not _contract_equal(window, expected_window):
        raise ValueError(f"Feature-store window contract mismatch: {window} != {expected_window}")
    if feature.get("view_order") != list(data_cfg.get("observation_views", [])):
        raise ValueError("Feature-store view order differs from the current data config.")
    if feature.get("action_shape") != [7]:
        raise ValueError("Feature-store action contract must be seven-dimensional.")
    dino_shape = feature.get("dino_token_shape", [])
    if dino_shape != [
        int(cfg["teacher"].get("spatial_tokens", 16)),
        int(cfg["teacher"].get("target_dim", 384)),
    ]:
        raise ValueError("Feature-store DINO target shape differs from the teacher contract.")
    cfg["backbone"]["hidden_dim"] = int(feature["openvla_view_shape"][-1])
    source = manifest.get("source_contract", {})
    if manifest.get(
        "formal_training_ready", source.get("formal_training_ready", True)
    ) is False:
        raise ValueError(
            "Feature store is incomplete or was produced by a limited/failed data audit and "
            "is smoke-only; rebuild it from the complete accepted training split before training."
        )
    stored_identity = source.get("openvla_identity")
    if not isinstance(stored_identity, dict):
        raise ValueError(
            "Formal feature store does not bind the immutable original OpenVLA identity; "
            "rebuild it with the original openvla/openvla-7b snapshot."
        )
    validate_openvla_identity(stored_identity)
    configured_identity = cfg.get("backbone", {}).get("identity")
    if configured_identity is not None and not openvla_identities_match(
        configured_identity, stored_identity
    ):
        raise ValueError("Feature store OpenVLA identity differs from the current config.")
    configured_repo = cfg.get("backbone", {}).get("repo_id")
    if configured_repo not in {None, "TBD", stored_identity["repo_id"]}:
        raise ValueError("Feature store OpenVLA repo differs from the current config.")
    configured_revision = cfg.get("backbone", {}).get("revision")
    if configured_revision not in {None, "TBD", stored_identity["revision"]}:
        raise ValueError("Feature store OpenVLA revision differs from the current config.")
    cfg["backbone"]["identity"] = stored_identity
    cfg["backbone"]["repo_id"] = stored_identity["repo_id"]
    cfg["backbone"]["revision"] = stored_identity["revision"]
    expected_openvla = cfg.get("backbone", {}).get("checkpoint") or cfg.get(
        "backbone", {}
    ).get("vla_path")
    stored_openvla = source.get("openvla_checkpoint")
    if stored_openvla:
        if expected_openvla in {None, "TBD"}:
            cfg["backbone"]["checkpoint"] = str(stored_openvla)
    # A local path is provenance only.  The immutable revision and file
    # fingerprints above are the semantic contract, so the same accepted
    # snapshot may be mounted at a different path on the training node.
    stored_teacher = source.get("teacher_checkpoint")
    expected_teacher = cfg.get("teacher", {}).get("checkpoint")
    if stored_teacher:
        if expected_teacher in {None, "TBD"}:
            cfg["teacher"]["checkpoint"] = str(stored_teacher)
        elif str(stored_teacher) != str(expected_teacher):
            # A feature-store training process never loads the frozen teacher;
            # it consumes the cached DINO targets and the transform contract
            # recorded below.  Absolute snapshot paths are therefore provenance,
            # not semantic identity, and may change when the same formal store is
            # mounted on another server.  The required raw/cache equivalence audit
            # re-encodes real windows with the newly supplied teacher path and will
            # fail if its outputs are not equivalent to the cached targets.
            cfg["teacher"]["source_checkpoint"] = str(stored_teacher)
    if source.get("joint_action_statistics") is None:
        raise ValueError("Feature store is missing joint_action_statistics.")
    data_cfg["joint_action_statistics"] = source["joint_action_statistics"]
    data_cfg["skill_sidecar_metadata"] = source.get("skill_sidecar_metadata") or {
        "fingerprint_sha256": source.get("skill_sidecar_fingerprint")
    }
    data_cfg["source_rlds_manifest_fingerprint"] = source.get(
        "rlds_manifest_fingerprint"
    ) or source.get("dataset_fingerprint")
    cfg["feature_store_contract"] = {
        "format": manifest["format"],
        "root": str(_feature_store_root(cfg).resolve()),
        "feature_contract": feature,
        "window_contract": window,
        "source_contract": source,
        "completion_contract": manifest.get("completion_contract"),
        "formal_training_ready": manifest.get(
            "formal_training_ready", source.get("formal_training_ready", True)
        ),
        "episode_count": int(manifest["episode_count"]),
        "frame_count": int(manifest["frame_count"]),
        "window_count": int(manifest["window_count"]),
    }
    return manifest


def _build_flow_dataset(
    cfg: dict[str, Any],
    model,
    *,
    episode_partition_name: str,
    limit,
    window_shuffle_buffer_size: int,
    windows_per_episode=None,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
):
    require_torch()
    data_cfg = cfg["data"]
    cfg["backbone"]["image_resolution"] = list(model.backbone.resize_resolution)
    dataset = LiberoSequenceDataset(
        dataset_root=data_cfg["data_root"],
        processor=model.backbone.processor,
        dataset_names=data_cfg["dataset_names"],
        history_length=int(data_cfg.get("history_length", 8)),
        long_memory_slots=int(data_cfg.get("long_memory_slots", 4)),
        future_horizons=data_cfg.get("future_horizons", [1, 4, 8, 16]),
        split=data_cfg.get("split", "train"),
        resize_resolution=tuple(model.backbone.resize_resolution),
        image_aug=bool(data_cfg.get("image_aug", False)),
        use_proprio=bool(data_cfg.get("use_proprio", True)),
        openvla_root=cfg["backbone"].get("openvla_root", "external/openvla-oft"),
        limit=limit,
        joint_action_normalization=bool(data_cfg.get("joint_action_normalization", True)),
        skill_sidecar_path=data_cfg.get("skill_sidecar_path"),
        assume_sidecar_timestep_aligned=bool(data_cfg.get("assume_sidecar_timestep_aligned", True)),
        window_shuffle_buffer_size=window_shuffle_buffer_size,
        episode_partition_name=episode_partition_name,
        validation_fraction=float(data_cfg.get("validation_fraction", 0.0)),
        split_seed=int(data_cfg.get("split_seed", 17)),
        action_chunk_size=int(data_cfg.get("action_chunk_size", 16)),
        windows_per_episode=windows_per_episode,
        distributed_rank=distributed_rank,
        distributed_world_size=distributed_world_size,
        tf_frame_parallel_calls=data_cfg.get("tf_frame_parallel_calls"),
    )
    cfg["data"]["joint_action_statistics"] = _jsonable(dataset.joint_action_statistics)
    if dataset.skill_sidecar_metadata is not None:
        cfg["data"]["skill_sidecar_metadata"] = _jsonable(dataset.skill_sidecar_metadata)
    return _attach_teacher_cache(cfg, dataset)


def _build_feature_store_dataset(
    cfg: dict[str, Any],
    *,
    partition: str,
):
    return MoWEFeatureWindowDataset(
        _feature_store_root(cfg),
        partition=partition,
        max_open_feature_shards=int(cfg["data"].get("max_open_feature_shards", 2)),
        verify_metadata_checksums=bool(
            cfg["data"].get("verify_feature_store_metadata_checksums", True)
        ),
    )


def build_flow_dataloader(
    cfg: dict[str, Any],
    model,
    distributed: DistributedContext | None = None,
):
    torch = require_torch()
    data_cfg = cfg["data"]
    distributed = distributed or DistributedContext(False, 0, 0, 1, "none", resolve_device(cfg))
    num_workers = int(data_cfg.get("num_workers", 0))
    if data_cfg.get("backend", "rlds") == "mowe_feature_store_v1":
        if num_workers != 0:
            raise ValueError(
                "mowe_feature_store_v1 currently requires data.num_workers=0 so sampler cursor "
                "resume is exact; raise this only after the multi-worker soak gate."
            )
        dataset = _build_feature_store_dataset(cfg, partition="train")
        sampler = EpisodeAwareDistributedSampler(
            dataset,
            rank=distributed.rank,
            world_size=distributed.world_size,
            seed=int(cfg.get("seed", 7)),
            shuffle=True,
            shuffle_block_size=int(data_cfg.get("sampler_shuffle_block_size", 256)),
        )
        if len(sampler) == 0:
            raise RuntimeError(
                f"Feature-store rank {distributed.rank} has no training windows; reduce world size "
                "or rebuild the train partition."
            )
        kwargs = {
            "dataset": dataset,
            "batch_size": int(cfg["training"].get("batch_size", 1)),
            "collate_fn": LatentWAMCollator(),
            "sampler": sampler,
            "num_workers": num_workers,
            "pin_memory": bool(data_cfg.get("pin_memory", False))
            and resolve_device(cfg).startswith("cuda"),
            "persistent_workers": bool(data_cfg.get("persistent_workers", False))
            and num_workers > 0,
        }
        if num_workers > 0 and data_cfg.get("prefetch_factor") is not None:
            kwargs["prefetch_factor"] = int(data_cfg["prefetch_factor"])
        return torch.utils.data.DataLoader(**kwargs)
    if distributed.enabled and num_workers != 0:
        raise ValueError(
            "Flow-WAM DDP requires data.num_workers=0; torchrun already creates one TensorFlow pipeline per rank."
        )
    validation_enabled = bool(cfg.get("validation", {}).get("enabled", False))
    dataset = _build_flow_dataset(
        cfg,
        model,
        episode_partition_name="train" if validation_enabled else "all",
        limit=data_cfg.get("limit_batches"),
        window_shuffle_buffer_size=int(data_cfg.get("window_shuffle_buffer_size", 4096)),
        windows_per_episode=None,
        distributed_rank=distributed.rank,
        distributed_world_size=distributed.world_size,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        collate_fn=LatentWAMCollator(),
        num_workers=num_workers,
        pin_memory=bool(data_cfg.get("pin_memory", True)) and resolve_device(cfg).startswith("cuda"),
    )


def build_flow_validation_dataloader(
    cfg: dict[str, Any],
    model,
    distributed: DistributedContext | None = None,
):
    torch = require_torch()
    data_cfg = cfg["data"]
    validation_cfg = cfg.get("validation", {})
    distributed = distributed or DistributedContext(False, 0, 0, 1, "none", resolve_device(cfg))
    if not bool(validation_cfg.get("enabled", False)):
        return None
    if distributed.enabled and not distributed.is_main:
        return None
    if data_cfg.get("backend", "rlds") == "mowe_feature_store_v1":
        dataset = _build_feature_store_dataset(cfg, partition="validation")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=int(cfg["training"].get("batch_size", 1)),
            collate_fn=LatentWAMCollator(),
            num_workers=0,
            pin_memory=bool(data_cfg.get("pin_memory", False))
            and resolve_device(cfg).startswith("cuda"),
        )
    dataset = _build_flow_dataset(
        cfg,
        model,
        episode_partition_name="validation",
        limit=None,
        window_shuffle_buffer_size=0,
        windows_per_episode=int(validation_cfg.get("windows_per_episode", 1)),
        distributed_rank=0,
        distributed_world_size=1,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=int(cfg["training"].get("batch_size", 1)),
        collate_fn=LatentWAMCollator(),
        num_workers=0,
        pin_memory=bool(data_cfg.get("pin_memory", True)) and resolve_device(cfg).startswith("cuda"),
    )


def _expected_teacher_cache_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    transform = teacher_transform_metadata(
        cfg["teacher"].get("checkpoint"),
        cfg["backbone"].get("image_resolution", [224, 224]),
        int(cfg["teacher"].get("spatial_grid", 4)),
    )
    return {
        "teacher_checkpoint": cfg["teacher"].get("checkpoint"),
        "spatial_tokens": int(cfg["teacher"].get("spatial_tokens", 16)),
        "target_dim": int(cfg["teacher"].get("target_dim", 384)),
        "future_horizons": list(cfg["data"].get("future_horizons", [1, 4, 8, 16])),
        "context_views": list(cfg["data"]["observation_views"]),
        "teacher_target_views": list(cfg["teacher"]["target_views"]),
        "dataset_names": list(cfg["data"]["dataset_names"]),
        "dataset_fingerprint": rlds_manifest_fingerprint(
            cfg["data"]["data_root"], cfg["data"]["dataset_names"]
        ),
        "skill_sidecar_fingerprint": (
            cfg["data"].get("skill_sidecar_metadata") or {}
        ).get("fingerprint_sha256"),
        "image_resolution": transform["image_resolution"],
        "transform_id": transform["transform_id"],
        "transform_hash": transform["transform_hash"],
    }


def _attach_teacher_cache(cfg, dataset):
    torch = require_torch()
    cache_path = cfg.get("teacher", {}).get("cache_path")
    if not cache_path:
        return dataset
    expected = _expected_teacher_cache_metadata(cfg)
    path = Path(cache_path)
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if manifest_path.name == "manifest.json":
        cache = ShardedVisualTargetCache(manifest_path)
        validate_visual_cache_metadata(cache.metadata, expected)
        cfg["teacher"]["cache_metadata"] = dict(cache.metadata)
        base_dataset = dataset

        class ShardedCachedDataset(torch.utils.data.IterableDataset):
            def __iter__(self):
                # Each worker gets an independent file-handle/LRU state.
                worker_cache = ShardedVisualTargetCache(manifest_path)
                for sample in base_dataset:
                    current, future = worker_cache.window(
                        sample["episode_id"],
                        sample["step_id"],
                        sample["future_horizons"].tolist(),
                    )
                    output = dict(sample)
                    output["current_latent_target"] = current
                    output["future_latent_targets"] = future
                    yield output

        return ShardedCachedDataset()

    state = torch.load(Path(cache_path), map_location="cpu")
    if state.get("format") != "latent_teacher_cache_v1":
        raise ValueError(
            f"Unsupported teacher cache format: {state.get('format')}; expected legacy "
            f"latent_teacher_cache_v1 or sharded {CACHE_FORMAT}."
        )
    metadata = state.get("metadata", {})
    validate_visual_cache_metadata(metadata, expected)
    cfg["teacher"]["cache_metadata"] = dict(metadata)
    targets = state.get("targets", {})
    base_dataset = dataset

    class CachedDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            for sample in base_dataset:
                key = f"{sample['episode_id']}:{sample['step_id']}"
                if key not in targets:
                    raise KeyError(f"Teacher cache miss for {key}")
                output = dict(sample)
                output["current_latent_target"] = targets[key]["current"].float()
                output["future_latent_targets"] = targets[key]["future"].float()
                yield output

    return CachedDataset()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _contract_equal(left, right) -> bool:
    return json.dumps(_jsonable(left), sort_keys=True) == json.dumps(_jsonable(right), sort_keys=True)


def validate_checkpoint_contract(metadata: dict[str, Any], cfg: dict[str, Any]) -> None:
    checkpoint_flow = metadata.get("flow_contract", {})
    expected_flow = {
        "formulation": cfg["flow"]["formulation"],
        "solver": cfg["flow"]["solver"],
        "solver_steps": int(cfg["flow"]["num_inference_steps"]),
        "implementation_id": "rectified_flow_euler_v1",
        "seed_policy": int(cfg["flow"]["deterministic_seed"]),
    }
    if not _contract_equal(checkpoint_flow, expected_flow):
        raise ValueError(f"Checkpoint flow contract mismatch: {checkpoint_flow} != {expected_flow}")
    if metadata.get("skill_experts") is not None and not _contract_equal(
        metadata["skill_experts"], cfg.get("skill_experts_resolved")
    ):
        raise ValueError("Checkpoint skill taxonomy/audit contract differs from the current config.")
    checkpoint_stats = metadata.get("data_contract", {}).get("joint_action_statistics")
    if checkpoint_stats is not None and not _contract_equal(
        checkpoint_stats, cfg.get("data", {}).get("joint_action_statistics")
    ):
        raise ValueError("Checkpoint 6D motion normalization statistics differ from the current data.")
    checkpoint_store = metadata.get("data_contract", {}).get("feature_store_contract")
    current_store = cfg.get("feature_store_contract")
    if checkpoint_store is not None:
        normalized_current = (
            {key: value for key, value in current_store.items() if key != "root"}
            if current_store is not None
            else None
        )
        if not _contract_equal(checkpoint_store, normalized_current):
            raise ValueError("Checkpoint feature-store contract differs from the current store.")
    expected_identity = cfg.get("backbone", {}).get("identity")
    observed_identity = metadata.get("backbone_identity")
    if expected_identity is not None or observed_identity is not None:
        if not openvla_identities_match(observed_identity, expected_identity):
            raise ValueError("Checkpoint backbone identity differs from the current frozen backbone.")
    else:
        expected_backbone = cfg.get("backbone", {}).get("checkpoint") or cfg.get(
            "backbone", {}
        ).get("vla_path")
        if metadata.get("backbone_identifier") not in {None, expected_backbone}:
            raise ValueError("Checkpoint backbone identifier differs from the current frozen backbone.")
    expected_views = {
        "observation_views": cfg["data"]["observation_views"],
        "teacher_target_views": cfg["teacher"]["target_views"],
        "num_images_in_input": int(cfg["backbone"]["num_images_in_input"]),
        "fusion": cfg["view_fusion"],
    }
    if not _contract_equal(metadata.get("view_contract", {}), expected_views):
        raise ValueError("Checkpoint dual-view contract differs from the current config.")


def _resume_training_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    """Select optimization semantics that must not drift within one stage."""

    training = cfg.get("training", {})
    data = cfg.get("data", {})
    router = cfg.get("router", {})
    action_condition = cfg.get("action_condition", {})
    experts = cfg.get("experts", {})
    return {
        "seed": int(cfg.get("seed", 7)),
        "stage": cfg.get("stage"),
        "training": {
            name: training.get(name)
            for name in (
                "precision",
                "max_steps",
                "learning_rates",
                "weight_decay",
                "adam_betas",
                "warmup_ratio",
                "min_lr_ratio",
                "max_grad_norm",
            )
        },
        "loss_weights": cfg.get("loss_weights", {}),
        "router_schedule": {
            name: router.get(name)
            for name in (
                "gumbel_temperature_start",
                "gumbel_temperature_end",
                "predicted_route_start_ratio",
                "predicted_route_end_ratio",
            )
        },
        "action_condition_schedule": {
            name: action_condition.get(name)
            for name in (
                "nominal_start_ratio",
                "nominal_end_ratio",
                "final_nominal_probability",
                "distance_beta",
            )
        },
        "residual_contract": {
            "max_residual_l2": float(experts.get("max_residual_l2", 0.5)),
        },
        "execution_contract": cfg.get("execution", {}),
        "window_contract": {
            name: data.get(
                name,
                256 if name == "sampler_shuffle_block_size" else None,
            )
            for name in (
                "history_length",
                "long_memory_slots",
                "future_horizons",
                "action_chunk_size",
                "sampler_shuffle_block_size",
            )
        },
        "route_mode": cfg.get("route_mode", "scheduled"),
        "ablation": cfg.get("ablation"),
    }


def validate_resume_schedule_contract(
    metadata: dict[str, Any], cfg: dict[str, Any]
) -> None:
    """Reject same-stage resumes that silently change optimization schedules."""

    checkpoint_cfg = metadata.get("config")
    if not isinstance(checkpoint_cfg, dict):
        raise ValueError("Checkpoint is missing the resolved training config required for resume.")
    observed = _resume_training_contract(checkpoint_cfg)
    expected = _resume_training_contract(cfg)
    if not _contract_equal(observed, expected):
        changed = sorted(
            key
            for key in set(observed) | set(expected)
            if not _contract_equal(observed.get(key), expected.get(key))
        )
        raise ValueError(
            "Same-stage resume changes the training/schedule contract in sections "
            f"{changed}; keep the original max_steps and optimization semantics. "
            "Use stop_step only to choose a temporary exit boundary."
        )


def _sha256_json(value: Any) -> str:
    rendered = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _resolved_backbone_identity(cfg: dict[str, Any]) -> dict[str, Any] | None:
    identity = cfg.get("backbone", {}).get("identity")
    if identity is None:
        return None
    return validate_openvla_identity(identity)


def _resolved_backbone_identifier(cfg: dict[str, Any]) -> str | None:
    identity = _resolved_backbone_identity(cfg)
    if identity is not None:
        return str(identity["identity_sha256"])
    return cfg.get("backbone", {}).get("checkpoint") or cfg.get("backbone", {}).get(
        "vla_path"
    )


def long_run_launch_contract(
    cfg: dict[str, Any], *, stage: str, world_size: int
) -> dict[str, Any]:
    """Select every launch semantic a readiness attestation must bind."""

    training = cfg.get("training", {})
    store_contract = cfg.get("feature_store_contract")
    if not isinstance(store_contract, dict):
        raise ValueError("Long-run launch contract requires a resolved feature store.")
    teacher_contract = (
        store_contract.get("source_contract", {}).get("teacher_transform_metadata")
        or cfg.get("teacher", {}).get("transform_metadata")
    )
    return {
        "stage": str(stage),
        "world_size": int(world_size),
        "effective_global_batch": (
            int(training.get("batch_size", 0))
            * int(training.get("grad_accumulation_steps", 0))
            * int(world_size)
        ),
        "training_contract": _resume_training_contract(cfg),
        "feature_store_contract": store_contract,
        "skill_experts": cfg.get("skill_experts_resolved"),
        "backbone_identity": _resolved_backbone_identity(cfg),
        "backbone_identifier": _resolved_backbone_identifier(cfg),
        # Feature-store training never loads the teacher snapshot. Bind the
        # attestation to the cached target transform recorded by the store,
        # rather than to a server-specific local snapshot path.
        "teacher_identifier": teacher_contract,
        "max_unattested_steps": int(
            cfg.get("long_run_readiness", {}).get("max_unattested_steps", 100)
        ),
    }


def long_run_launch_fingerprint(
    cfg: dict[str, Any], *, stage: str, world_size: int
) -> str:
    return _sha256_json(
        long_run_launch_contract(cfg, stage=stage, world_size=world_size)
    )


def readiness_checkpoint_identity(
    metadata: dict[str, Any] | None, *, checkpoint_mode: str
) -> dict[str, Any]:
    """Bind an attestation to the checkpoint lineage without hashing multi-GiB tensors."""

    if metadata is None:
        return {"mode": "none", "checkpoint": None}
    config = metadata.get("config")
    config_contract = (
        _sha256_json(_resume_training_contract(config))
        if isinstance(config, dict)
        else None
    )
    distributed_state = metadata.get("distributed_contract") or {}
    data_contract = metadata.get("data_contract") or {}
    return {
        "mode": str(checkpoint_mode),
        "checkpoint": {
            "stage": metadata.get("stage"),
            "step": int(metadata.get("step", -1)),
            "world_size": int(distributed_state.get("world_size", 1)),
            "effective_global_batch": _checkpoint_effective_batch(metadata),
            "training_contract_sha256": config_contract,
            "backbone_identity": metadata.get("backbone_identity"),
            "backbone_identifier": metadata.get("backbone_identifier"),
            "feature_store_contract_sha256": _sha256_json(
                data_contract.get("feature_store_contract")
            ),
        },
    }


def validate_long_run_readiness_attestation(
    report: dict[str, Any],
    cfg: dict[str, Any],
    *,
    stage: str,
    world_size: int,
    checkpoint_metadata: dict[str, Any] | None,
    checkpoint_mode: str,
    runtime_identity: dict[str, Any],
) -> None:
    """Reject stale, cross-stage, cross-store, or cross-checkpoint readiness reports."""

    if report.get("format") != "flow_wam_long_run_readiness_v1":
        raise ValueError("Unsupported long-run readiness report format.")
    if not bool(report.get("passed", False)) or report.get("errors"):
        raise ValueError("Long-run readiness report did not pass all launch gates.")
    checks = report.get("checks")
    if not isinstance(checks, dict) or not checks or not all(
        bool(value) for value in checks.values()
    ):
        raise ValueError("Long-run readiness report contains a failed or missing check.")
    expected_contract = long_run_launch_contract(
        cfg, stage=stage, world_size=world_size
    )
    expected_fingerprint = _sha256_json(expected_contract)
    if report.get("launch_contract_sha256") != expected_fingerprint:
        raise ValueError(
            "Long-run readiness launch contract differs from the current resolved config/store."
        )
    if not _contract_equal(report.get("launch_contract"), expected_contract):
        raise ValueError("Long-run readiness launch contract payload was modified.")
    expected_checkpoint = readiness_checkpoint_identity(
        checkpoint_metadata, checkpoint_mode=checkpoint_mode
    )
    if not _contract_equal(report.get("checkpoint_identity"), expected_checkpoint):
        raise ValueError(
            "Long-run readiness report targets a different checkpoint or checkpoint mode."
        )
    if not _contract_equal(report.get("runtime_identity"), runtime_identity):
        raise ValueError(
            "Long-run readiness report was generated for a different node, boot, cgroup, "
            "or GPU topology. Regenerate target-node evidence."
        )


def enforce_long_run_readiness(
    cfg: dict[str, Any],
    *,
    stage: str,
    world_size: int,
    start_step: int,
    stop_step: int,
    checkpoint_metadata: dict[str, Any] | None,
    checkpoint_mode: str,
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    """Allow bounded smoke runs, but require an attestation for continuous training."""

    policy = cfg.setdefault("long_run_readiness", {})
    planned_steps = int(stop_step) - int(start_step)
    max_unattested_steps = int(policy.get("max_unattested_steps", 100))
    prior_unattested_steps = 0
    if checkpoint_mode == "resume" and isinstance(checkpoint_metadata, dict):
        previous_attestation = (
            checkpoint_metadata.get("config", {})
            .get("long_run_readiness", {})
            .get("attestation", {})
        )
        if previous_attestation.get("mode") == "bounded_smoke":
            prior_unattested_steps = int(
                previous_attestation.get(
                    "unattested_lineage_steps",
                    previous_attestation.get("planned_optimizer_steps", 0),
                )
            )
    cumulative_unattested_steps = prior_unattested_steps + planned_steps
    status = {
        "planned_optimizer_steps": planned_steps,
        "prior_unattested_lineage_steps": prior_unattested_steps,
        "unattested_lineage_steps": cumulative_unattested_steps,
        "max_unattested_steps": max_unattested_steps,
        "required": cumulative_unattested_steps > max_unattested_steps,
        "report_path": policy.get("report_path"),
    }
    if cumulative_unattested_steps <= max_unattested_steps:
        status["validated"] = False
        status["mode"] = "bounded_smoke"
        policy["attestation"] = status
        return status
    report_path = policy.get("report_path")
    if not report_path:
        raise ValueError(
            f"Planned run adds {planned_steps} optimizer steps to "
            f"{prior_unattested_steps} unattested lineage steps, above the smoke limit "
            f"{max_unattested_steps}; pass --long-run-readiness-report."
        )
    path = Path(str(report_path)).expanduser()
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Long-run readiness report must contain a JSON object.")
    validate_long_run_readiness_attestation(
        report,
        cfg,
        stage=stage,
        world_size=world_size,
        checkpoint_metadata=checkpoint_metadata,
        checkpoint_mode=checkpoint_mode,
        runtime_identity=runtime_identity,
    )
    status.update(
        {
            "validated": True,
            "mode": "attested_long_run",
            "unattested_lineage_steps": 0,
            "launch_contract_sha256": report["launch_contract_sha256"],
        }
    )
    policy["attestation"] = status
    return status


def validate_backbone_identifier(
    metadata: dict[str, Any],
    requested_checkpoint: str | Path,
    *,
    requested_identity: dict[str, Any] | None = None,
) -> None:
    """Prevent evaluation with a different or legacy-path-only backbone."""

    observed = metadata.get("backbone_identity")
    if observed is None:
        observed = (
            metadata.get("data_contract", {})
            .get("feature_store_contract", {})
            .get("source_contract", {})
            .get("openvla_identity")
        )
    if observed is None:
        raise ValueError(
            "Checkpoint does not bind the immutable original OpenVLA backbone identity."
        )
    validate_openvla_identity(observed)
    if requested_identity is None:
        raise ValueError(
            "Evaluation must resolve the requested local OpenVLA snapshot identity before model loading."
        )
    validate_openvla_identity(requested_identity)
    if not openvla_identities_match(observed, requested_identity):
        raise ValueError(
            "Evaluation backbone differs from the checkpoint-bound original OpenVLA identity: "
            f"{observed.get('identity_sha256')!r} != "
            f"{requested_identity.get('identity_sha256')!r} ({requested_checkpoint})."
        )


def configure_flow_stage(model, stage: str) -> None:
    enabled = {
        "nominal_flow_pretrain": {"view_fusion", "memory_encoder", "nominal_action_head", "world_model"},
        "expert_warmstart": {
            "view_fusion", "memory_encoder", "world_model", "router", "residual_experts", "expert_context"
        },
        "joint": set(FLOW_COMPONENTS),
    }
    if stage not in enabled:
        raise ValueError(f"Unknown stage: {stage}")
    for name in FLOW_COMPONENTS:
        component = getattr(model, name)
        is_enabled = name in enabled[stage]
        component.train(is_enabled)
        for parameter in component.parameters():
            parameter.requires_grad_(is_enabled)
    # The nominal flow trunk never receives a per-token motion condition;
    # that projection exists only because nominal and residual flows share the
    # ActionFlowTrunk implementation. Leaving these parameters trainable makes
    # DDP fail on the second iteration with an unfinished reduction.
    for parameter in model.nominal_action_head.flow_trunk.token_condition_projection.parameters():
        parameter.requires_grad_(False)
    # Stage 1 has no router loss and keeps the router frozen. The route-world
    # projection therefore has no gradient path until Stage 2/3, while the
    # future/delta heads remain supervised by the Stage-1 world losses.
    if stage == "nominal_flow_pretrain":
        for parameter in model.world_model.route_world_head.parameters():
            parameter.requires_grad_(False)
    freeze = getattr(model.backbone, "freeze", None)
    if freeze is not None:
        freeze()
    if model.visual_teacher is not None:
        model.visual_teacher.freeze()


def build_flow_optimizer(cfg: dict[str, Any], model):
    torch = require_torch()
    training = cfg["training"]
    rates = training.get("learning_rates", {})
    groups = []
    seen: set[int] = set()
    logical_components = [
        ("view_fusion", [model.view_fusion]),
        ("memory_encoder", [model.memory_encoder]),
        (
            "nominal_flow",
            [
                model.nominal_action_head.condition_encoder,
                model.nominal_action_head.flow_trunk,
                model.nominal_action_head.motion_head,
            ],
        ),
        ("gripper_head", [model.nominal_action_head.gripper_head]),
        ("world_model", [model.world_model]),
        ("router", [model.router]),
        ("residual_experts", [model.residual_experts]),
        ("expert_context", [model.expert_context]),
    ]
    for name, modules in logical_components:
        params = []
        for module in modules:
            for parameter in module.parameters():
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    params.append(parameter)
        if params:
            if name not in rates:
                raise ValueError(f"Missing learning rate for enabled component {name}.")
            groups.append({"params": params, "lr": float(rates[name]), "name": name})
    if not groups:
        raise RuntimeError("No trainable flow-WAM parameters.")
    betas = tuple(float(value) for value in training.get("adam_betas", [0.9, 0.95]))
    return torch.optim.AdamW(groups, betas=betas, weight_decay=float(training.get("weight_decay", 0.01)))


def build_warmup_cosine_scheduler(cfg, optimizer):
    torch = require_torch()
    total = int(cfg["training"].get("max_steps", 1))
    warmup = max(1, int(total * float(cfg["training"].get("warmup_ratio", 0.04))))
    minimum = float(cfg["training"].get("min_lr_ratio", 0.1))

    def scale(step):
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(1.0, (step - warmup) / max(total - warmup, 1))
        return minimum + 0.5 * (1.0 - minimum) * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def checkpoint_state(
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    cfg,
    stage,
    schedule_state,
    *,
    distributed_metadata: dict[str, Any] | None = None,
    rng_state_by_rank: list[dict[str, Any]] | None = None,
    sampler_state_by_rank: list[dict[str, Any]] | None = None,
):
    torch = require_torch()
    fallback_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return {
        "format": "flow_wam_skill_components_v2",
        "stage": stage,
        "step": int(step),
        "config": cfg,
        "components": {name: getattr(model, name).state_dict() for name in FLOW_COMPONENTS},
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "schedule_state": dict(schedule_state),
        "flow_contract": {
            "formulation": cfg["flow"]["formulation"],
            "solver": cfg["flow"]["solver"],
            "solver_steps": int(cfg["flow"]["num_inference_steps"]),
            "implementation_id": model.nominal_action_head.sampler.implementation_id,
            "seed_policy": int(cfg["flow"]["deterministic_seed"]),
        },
        "skill_experts": cfg.get("skill_experts_resolved"),
        "data_contract": {
            "joint_action_statistics": cfg.get("data", {}).get("joint_action_statistics"),
            "skill_sidecar_metadata": cfg.get("data", {}).get("skill_sidecar_metadata"),
            "feature_store_contract": (
                {
                    key: value
                    for key, value in cfg.get("feature_store_contract", {}).items()
                    if key != "root"
                }
                or None
            ),
        },
        "teacher_contract": {
            "checkpoint": cfg.get("teacher", {}).get("checkpoint"),
            "transform_metadata": cfg.get("teacher", {}).get("transform_metadata"),
            "cache_metadata": cfg.get("teacher", {}).get("cache_metadata"),
        },
        "view_contract": {
            "observation_views": cfg["data"]["observation_views"],
            "teacher_target_views": cfg["teacher"]["target_views"],
            "num_images_in_input": int(cfg["backbone"]["num_images_in_input"]),
            "fusion": cfg["view_fusion"],
        },
        "backbone_identity": _resolved_backbone_identity(cfg),
        "backbone_identifier": _resolved_backbone_identifier(cfg),
        "distributed_contract": distributed_metadata,
        "rng_state_by_rank": rng_state_by_rank,
        "sampler_state_by_rank": sampler_state_by_rank,
        "python_rng_state": random.getstate(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": fallback_cuda_state,
    }


def save_flow_checkpoint(
    path,
    model,
    optimizer,
    scheduler,
    scaler,
    step,
    cfg,
    stage,
    schedule_state,
    *,
    distributed_metadata: dict[str, Any] | None = None,
    rng_state_by_rank: list[dict[str, Any]] | None = None,
    sampler_state_by_rank: list[dict[str, Any]] | None = None,
):
    torch = require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = checkpoint_state(
        model,
        optimizer,
        scheduler,
        scaler,
        step,
        cfg,
        stage,
        schedule_state,
        distributed_metadata=distributed_metadata,
        rng_state_by_rank=rng_state_by_rank,
        sampler_state_by_rank=sampler_state_by_rank,
    )
    checkpoint_temporary = path.with_suffix(path.suffix + ".tmp")
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_temporary = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    metadata = {
        "format": state["format"],
        "stage": state["stage"],
        "step": state["step"],
        "config": state["config"],
        "flow_contract": state["flow_contract"],
        "skill_experts": state["skill_experts"],
        "data_contract": state["data_contract"],
        "teacher_contract": state["teacher_contract"],
        "view_contract": state["view_contract"],
        "backbone_identity": state["backbone_identity"],
        "backbone_identifier": state["backbone_identifier"],
        "distributed_contract": state["distributed_contract"],
        "sampler_state_by_rank": state["sampler_state_by_rank"],
    }
    # Keep the previous checkpoint recoverable until the new payload has been
    # completely serialized.  Remove the old sidecar just before the atomic
    # payload switch: if the process dies in that narrow window, metadata
    # readers safely fall back to the self-contained checkpoint instead of
    # trusting a sidecar from the previous generation.
    checkpoint_temporary.unlink(missing_ok=True)
    metadata_temporary.unlink(missing_ok=True)
    try:
        torch.save(state, checkpoint_temporary)
        metadata_temporary.write_text(
            json.dumps(_jsonable(metadata), indent=2, sort_keys=True), encoding="utf-8"
        )
        metadata_path.unlink(missing_ok=True)
        checkpoint_temporary.replace(path)
        metadata_temporary.replace(metadata_path)
    finally:
        checkpoint_temporary.unlink(missing_ok=True)
        metadata_temporary.unlink(missing_ok=True)


def read_flow_checkpoint_metadata(path) -> dict[str, Any]:
    """Read the resolved architecture/runtime contract before constructing a model."""

    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if sidecar.exists():
        state = json.loads(sidecar.read_text(encoding="utf-8"))
    else:
        torch = require_torch()
        state = torch.load(path, map_location="cpu")
    if state.get("format") != "flow_wam_skill_components_v2":
        raise ValueError(f"Unsupported flow-WAM checkpoint: {path}")
    return {
        "stage": state.get("stage"),
        "step": state.get("step"),
        "config": state.get("config", {}),
        "flow_contract": state.get("flow_contract", {}),
        "skill_experts": state.get("skill_experts"),
        "data_contract": state.get("data_contract", {}),
        "teacher_contract": state.get("teacher_contract", {}),
        "view_contract": state.get("view_contract", {}),
        "backbone_identity": state.get("backbone_identity"),
        "backbone_identifier": state.get("backbone_identifier"),
        "distributed_contract": state.get("distributed_contract"),
        "sampler_state_by_rank": state.get("sampler_state_by_rank"),
    }


def _checkpoint_effective_batch(state: dict[str, Any]) -> int:
    contract = state.get("distributed_contract") or {}
    if contract.get("effective_global_batch") is not None:
        return int(contract["effective_global_batch"])
    training = state.get("config", {}).get("training", {})
    return (
        int(training.get("batch_size", 1))
        * int(training.get("grad_accumulation_steps", 1))
        * int(contract.get("world_size", 1))
    )


def validate_distributed_resume_contract(
    state: dict[str, Any],
    cfg: dict[str, Any],
    distributed: DistributedContext,
    *,
    allow_world_size_change: bool,
) -> None:
    checkpoint_contract = state.get("distributed_contract") or {
        "enabled": False,
        "world_size": 1,
        "effective_global_batch": _checkpoint_effective_batch(state),
    }
    checkpoint_world_size = int(checkpoint_contract.get("world_size", 1))
    current_effective_batch = effective_global_batch(cfg, distributed.world_size)
    checkpoint_effective_batch = _checkpoint_effective_batch(state)
    if checkpoint_world_size != distributed.world_size and not allow_world_size_change:
        raise ValueError(
            "Checkpoint world size differs from the current launch. Pass --allow-world-size-change "
            "only for an intentional migration that preserves effective global batch: "
            f"{checkpoint_world_size} -> {distributed.world_size}."
        )
    if checkpoint_effective_batch != current_effective_batch:
        raise ValueError(
            "Checkpoint migration changes effective global batch: "
            f"{checkpoint_effective_batch} != {current_effective_batch}."
        )


def load_flow_checkpoint(
    path,
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    *,
    resume=False,
    metadata_out=None,
    allowed_stages=None,
    distributed: DistributedContext | None = None,
    current_cfg: dict[str, Any] | None = None,
    allow_world_size_change: bool = False,
):
    torch = require_torch()
    state = torch.load(Path(path), map_location="cpu")
    if state.get("format") != "flow_wam_skill_components_v2":
        raise ValueError(f"Unsupported flow-WAM checkpoint: {path}")
    if allowed_stages is not None and state.get("stage") not in set(allowed_stages):
        raise ValueError(
            f"Checkpoint stage {state.get('stage')!r} is not allowed here; expected one of {sorted(allowed_stages)}."
        )
    missing_components = sorted(set(FLOW_COMPONENTS) - set(state.get("components", {})))
    if missing_components:
        raise ValueError(f"Flow-WAM checkpoint is missing components: {missing_components}")
    if distributed is not None and current_cfg is not None:
        validate_distributed_resume_contract(
            state,
            current_cfg,
            distributed,
            allow_world_size_change=allow_world_size_change,
        )
    if metadata_out is not None:
        metadata_out.update(
            {
                "stage": state.get("stage"),
                "step": state.get("step"),
                "config": state.get("config", {}),
                "flow_contract": state.get("flow_contract", {}),
                "skill_experts": state.get("skill_experts"),
                "data_contract": state.get("data_contract", {}),
                "teacher_contract": state.get("teacher_contract", {}),
                "view_contract": state.get("view_contract", {}),
                "backbone_identity": state.get("backbone_identity"),
                "backbone_identifier": state.get("backbone_identifier"),
                "distributed_contract": state.get("distributed_contract"),
                "sampler_state_by_rank": state.get("sampler_state_by_rank"),
            }
        )
    for name, component_state in state["components"].items():
        if hasattr(model, name):
            getattr(model, name).load_state_dict(component_state)
    if not resume:
        return 0, state.get("schedule_state", {})
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None:
        scaler.load_state_dict(state["scaler"])
    rank_states = state.get("rng_state_by_rank") or []
    if distributed is not None and len(rank_states) == distributed.world_size:
        restore_local_rng_state(rank_states[distributed.rank], distributed)
    elif distributed is not None and distributed.enabled:
        # A 1->N migration cannot preserve a nonexistent per-rank RNG stream.
        # Derive deterministic independent streams and keep optimizer/schedule
        # continuity explicit in the checkpoint contract.
        set_seed(int((current_cfg or state.get("config", {})).get("seed", 7)), distributed.rank)
    else:
        random.setstate(state["python_rng_state"])
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
    return int(state.get("step", 0)), state.get("schedule_state", {})


def _route_metrics(outputs, batch):
    torch = require_torch()
    labels = batch.get("expert_skill_labels")
    mask = batch.get("expert_skill_mask")
    if labels is None or mask is None:
        return {}
    labels = labels.to(outputs["route_indices"].device)
    mask = mask.to(outputs["route_indices"].device).bool() & labels.ge(0)
    correct = outputs["route_indices"].eq(labels) & mask
    by_position = []
    valid_by_position = []
    for position in range(labels.shape[1]):
        valid = mask[:, position]
        valid_by_position.append(int(valid.sum()))
        by_position.append(float(correct[:, position].float().sum() / valid.sum().clamp_min(1)))
    coverage = [int(((labels == index) & mask).sum()) for index in range(7)]
    confusion = torch.zeros((7, 7), dtype=torch.long, device=labels.device)
    if bool(mask.any()):
        flat = labels[mask] * 7 + outputs["route_indices"][mask]
        confusion = torch.bincount(flat, minlength=49).reshape(7, 7)
    usage = outputs["route_gates"][..., :6].float().mean(dim=(0, 1)).detach().cpu().tolist()
    transition_valid = mask[:, 1:] & mask[:, :-1]
    true_boundary = labels[:, 1:].ne(labels[:, :-1]) & transition_valid
    predicted_boundary = outputs["route_indices"][:, 1:].ne(outputs["route_indices"][:, :-1]) & transition_valid
    true_positive = (true_boundary & predicted_boundary).sum().float()
    predicted_boundary_count = predicted_boundary.sum()
    true_boundary_count = true_boundary.sum()
    precision = true_positive / predicted_boundary_count.clamp_min(1)
    recall = true_positive / true_boundary_count.clamp_min(1)
    boundary_f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    prefix_cap = min(int(outputs["execution_default_steps"]), labels.shape[1])
    fully_observed_prefix = mask[:, :prefix_cap].all(dim=1)
    first_true_segment = first_skill_segment_steps(labels.clamp(min=0), prefix_cap)
    crossing_values = outputs["execution_steps"].gt(first_true_segment).float()
    crossing_count = crossing_values[fully_observed_prefix].sum()
    crossing_valid_count = fully_observed_prefix.sum()
    crossing_rate = (
        crossing_count / crossing_valid_count.clamp_min(1)
        if bool(fully_observed_prefix.any())
        else crossing_values.sum() * 0.0
    )

    def edit_distance(predicted, target):
        previous = list(range(len(target) + 1))
        for row, predicted_value in enumerate(predicted, start=1):
            current = [row]
            for column, target_value in enumerate(target, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[column] + 1,
                        previous[column - 1] + int(predicted_value != target_value),
                    )
                )
            previous = current
        return previous[-1]

    edit_distances = []
    for row in range(labels.shape[0]):
        valid = mask[row]
        truth_schedule = labels[row, valid].detach().cpu().tolist()
        predicted_schedule = outputs["route_indices"][row, valid].detach().cpu().tolist()
        if truth_schedule:
            edit_distances.append(edit_distance(predicted_schedule, truth_schedule) / len(truth_schedule))
    execution_histogram = torch.bincount(
        outputs["execution_steps"].detach().long().cpu(), minlength=prefix_cap + 1
    )[1 : prefix_cap + 1].tolist()
    reason_histogram = torch.bincount(
        outputs["execution_reason_code"].detach().long().cpu(), minlength=3
    )[:3].tolist()
    selected_boundary = outputs["execution_boundary_position"].ge(0)
    selected_boundary_count = selected_boundary.sum()

    def selected_mean(key):
        values = outputs[key].float()
        if bool(selected_boundary.any()):
            return float(values[selected_boundary].mean().detach())
        return 0.0

    return {
        "route_label_coverage": coverage,
        "route_valid_by_position": valid_by_position,
        "route_accuracy_by_position": by_position,
        "router_entropy_by_position": outputs["router_entropy"]
        .float()
        .mean(dim=0)
        .detach()
        .cpu()
        .tolist(),
        "current_skill_accuracy": by_position[0],
        "future_position_accuracy": float(sum(by_position[1:]) / max(len(by_position) - 1, 1)),
        "motor_expert_usage": usage,
        "null_route_usage": float(outputs["route_gates"][..., 6].float().mean().detach()),
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(boundary_f1),
        "boundary_true_positive_count": int(true_positive),
        "boundary_predicted_positive_count": int(predicted_boundary_count),
        "boundary_target_positive_count": int(true_boundary_count),
        "route_confusion_matrix": confusion.detach().cpu().tolist(),
        "schedule_edit_distance": float(sum(edit_distances) / max(len(edit_distances), 1)),
        "schedule_edit_distance_sum": float(sum(edit_distances)),
        "schedule_edit_distance_count": len(edit_distances),
        "execution_steps_histogram": execution_histogram,
        "execution_reason_histogram": reason_histogram,
        "replanning_frequency": float(
            (1.0 / outputs["execution_steps"].float()).mean().detach()
        ),
        "ground_truth_boundary_crossing_rate": float(crossing_rate),
        "ground_truth_boundary_crossing_count": int(crossing_count),
        "ground_truth_boundary_crossing_valid_count": int(crossing_valid_count),
        "predicted_boundary_crossing_rate": float(
            outputs["execution_crosses_predicted_boundary"].float().mean().detach()
        ),
        "execution_selected_boundary_count": int(selected_boundary_count),
        "execution_boundary_entropy_mean": selected_mean("execution_boundary_entropy"),
        "execution_boundary_margin_mean": selected_mean("execution_boundary_margin"),
        "execution_motion_jump_l2_mean": selected_mean("execution_motion_jump_l2"),
        "execution_residual_l2_mean": selected_mean("execution_residual_l2"),
    }


def _mechanism_metrics(outputs, batch):
    torch = require_torch()
    target_motion = batch.get("target_motion", batch["target_actions"][..., :6]).to(
        outputs["nominal_motion"].device
    )
    labels = batch.get("expert_skill_labels")
    label_mask = batch.get("expert_skill_mask")
    per_skill = {}
    per_skill_position = {}
    if labels is not None and label_mask is not None:
        labels = labels.to(target_motion.device)
        label_mask = label_mask.to(target_motion.device).bool()
        nominal_endpoint = (outputs["nominal_motion"].float() - target_motion.float()).abs().mean(dim=-1)
        final_endpoint = (outputs["motion_actions"].float() - target_motion.float()).abs().mean(dim=-1)
        nominal_flow_error = None
        expert_flow_error = None
        if "nominal_flow" in outputs:
            nominal_flow_error = (
                outputs["nominal_flow"]["predicted_velocity"].float()
                - outputs["nominal_flow"]["target_velocity"].float()
            ).abs().mean(dim=-1)
        if "expert_flow" in outputs:
            expert_flow_error = (
                outputs["expert_flow"]["predicted_velocity"].float()
                - outputs["expert_flow"]["target_velocity"].float()
            ).abs().mean(dim=-1)
        for skill_index, skill_name in enumerate(SKILL_NAMES):
            valid = label_mask & labels.eq(skill_index)
            record = {"count": int(valid.sum())}
            if bool(valid.any()):
                record.update(
                    {
                        "nominal_endpoint_l1": float(nominal_endpoint[valid].mean().detach()),
                        "final_endpoint_l1": float(final_endpoint[valid].mean().detach()),
                        "residual_norm": float(
                            outputs["residual_motion"].float().norm(dim=-1)[valid].mean().detach()
                        ),
                    }
                )
                if nominal_flow_error is not None:
                    record["nominal_flow_velocity_l1"] = float(
                        nominal_flow_error[valid].mean().detach()
                    )
                if expert_flow_error is not None and skill_index < 6:
                    record["expert_flow_velocity_l1"] = float(
                        expert_flow_error[valid].mean().detach()
                    )
            per_skill[skill_name] = record
            position_records = []
            for position in range(labels.shape[1]):
                position_valid = valid[:, position]
                position_record = {"count": int(position_valid.sum())}
                if bool(position_valid.any()):
                    position_record.update(
                        {
                            "nominal_endpoint_l1": float(
                                nominal_endpoint[position_valid, position].mean().detach()
                            ),
                            "final_endpoint_l1": float(
                                final_endpoint[position_valid, position].mean().detach()
                            ),
                            "residual_norm": float(
                                outputs["residual_motion"]
                                .float()
                                .norm(dim=-1)[position_valid, position]
                                .mean()
                                .detach()
                            ),
                        }
                    )
                    if nominal_flow_error is not None:
                        position_record["nominal_flow_velocity_l1"] = float(
                            nominal_flow_error[position_valid, position].mean().detach()
                        )
                    if expert_flow_error is not None and skill_index < 6:
                        position_record["expert_flow_velocity_l1"] = float(
                            expert_flow_error[position_valid, position].mean().detach()
                        )
                position_records.append(position_record)
            per_skill_position[skill_name] = position_records

    magnitude = target_motion.float().norm(dim=-1)
    bucket_masks = {
        "small_lt_0.10": magnitude.lt(0.10),
        "medium_0.10_0.30": magnitude.ge(0.10) & magnitude.lt(0.30),
        "large_ge_0.30": magnitude.ge(0.30),
    }
    motion_buckets = {}
    for name, valid in bucket_masks.items():
        record = {"count": int(valid.sum())}
        if bool(valid.any()):
            record["nominal_endpoint_l1"] = float(
                (outputs["nominal_motion"].float() - target_motion.float())
                .abs()
                .mean(dim=-1)[valid]
                .mean()
                .detach()
            )
            record["final_endpoint_l1"] = float(
                (outputs["motion_actions"].float() - target_motion.float())
                .abs()
                .mean(dim=-1)[valid]
                .mean()
                .detach()
            )
        motion_buckets[name] = record

    horizon_metrics = {}
    if "future_latent_targets" in outputs:
        target = outputs["future_latent_targets"].float()
        predicted = outputs["future_latents"].float()
        predicted_delta = outputs["delta_latents"].float()
        target_delta = outputs["delta_latent_targets"].float()
        current = outputs["current_latent_target"].float()
        horizons = batch["future_horizons"][0].detach().cpu().tolist()
        future_mask = batch["future_mask"].to(predicted.device).bool()
        for index, horizon in enumerate(horizons):
            valid = future_mask[:, index]
            if not bool(valid.any()):
                continue
            pred_value = predicted[valid, index]
            target_value = target[valid, index]
            copy_value = current[valid]
            horizon_metrics[str(int(horizon))] = {
                "cosine_distance": float(
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        pred_value, target_value, dim=-1
                    ).mean().detach()
                ),
                "smooth_l1": float(
                    torch.nn.functional.smooth_l1_loss(pred_value, target_value).detach()
                ),
                "delta_smooth_l1": float(
                    torch.nn.functional.smooth_l1_loss(
                        predicted_delta[valid, index], target_delta[valid, index]
                    ).detach()
                ),
                "current_copy_smooth_l1": float(
                    torch.nn.functional.smooth_l1_loss(copy_value, target_value).detach()
                ),
            }

    world_tokens = outputs["route_world_tokens"].float()
    current_view_weights = outputs["current_view_weights"].float()
    view_by_current_skill = {}
    raw_labels = batch.get("expert_skill_labels")
    raw_label_mask = batch.get("expert_skill_mask")
    if raw_labels is not None and raw_label_mask is not None:
        current_labels = raw_labels[:, 0].to(current_view_weights.device)
        current_valid = raw_label_mask[:, 0].to(current_view_weights.device).bool()
        for skill_index, skill_name in enumerate(SKILL_NAMES):
            valid = current_valid & current_labels.eq(skill_index)
            view_by_current_skill[skill_name] = {
                "count": int(valid.sum()),
                "mean_weights": (
                    current_view_weights[valid].mean(dim=0).detach().cpu().tolist()
                    if bool(valid.any())
                    else []
                ),
            }
    return {
        "per_skill_diagnostics": per_skill,
        "per_skill_position_diagnostics": per_skill_position,
        "motion_magnitude_buckets": motion_buckets,
        "future_horizon_metrics": horizon_metrics,
        "route_world_token_norm_by_position": world_tokens.norm(dim=-1).mean(dim=0).detach().cpu().tolist(),
        "route_world_token_variance_by_position": world_tokens.var(dim=(0, 2)).detach().cpu().tolist(),
        "router_branch_norms": {
            name: float(value.detach())
            for name, value in outputs.get("router_branch_norms", {}).items()
        },
        "view_order": list(outputs["view_order"]),
        "current_view_weights_mean": current_view_weights.mean(dim=0).detach().cpu().tolist(),
        "current_view_entropy_mean": float(outputs["current_view_entropy"].float().mean().detach()),
        "history_view_weights_by_position": outputs["history_view_weights"]
        .float()
        .mean(dim=0)
        .detach()
        .cpu()
        .tolist(),
        "view_weights_by_current_skill": view_by_current_skill,
    }


def _gradient_norms(model):
    values = {}
    for name in FLOW_COMPONENTS:
        total = 0.0
        for parameter in getattr(model, name).parameters():
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().square().sum())
        values[name] = math.sqrt(total)
    return values


def _motor_expert_gradient_norms(model):
    values = []
    for adapter, head in zip(model.residual_experts.adapters, model.residual_experts.velocity_heads):
        total = 0.0
        for parameter in list(adapter.parameters()) + list(head.parameters()):
            if parameter.grad is not None:
                total += float(parameter.grad.detach().float().square().sum())
        values.append(math.sqrt(total))
    return values


def _write_jsonl(path, record):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(record), sort_keys=True) + "\n")


def _write_json_atomic(path, payload) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validation_loss_early_stopping_state(
    records: list[dict[str, Any]],
    *,
    stage: str,
    metric: str = "total_loss",
    min_delta: float = 1e-4,
    patience: int = 5,
    min_steps: int = 5000,
) -> dict[str, Any]:
    """Rebuild deterministic loss-only early-stopping state from validation logs."""

    if min_delta < 0:
        raise ValueError("validation.early_stopping.min_delta must be non-negative.")
    if patience < 1:
        raise ValueError("validation.early_stopping.patience must be positive.")
    if min_steps < 0:
        raise ValueError("validation.early_stopping.min_steps must be non-negative.")
    if not metric:
        raise ValueError("validation.early_stopping.metric must be non-empty.")

    # Re-launching a segment evaluates the checkpoint step again. Keep only the
    # newest value for each step so resume does not consume patience twice.
    losses_by_step: dict[int, float] = {}
    for record in records:
        if record.get("stage") != stage:
            continue
        try:
            step = int(record["step"])
            value = float(record.get("metrics", {})[metric])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(value):
            raise RuntimeError(
                f"Validation metric {metric!r} is non-finite at step {step}: {value}"
            )
        losses_by_step[step] = value

    best_value: float | None = None
    current_value: float | None = None
    current_step = 0
    bad_validation_count = 0
    for current_step, current_value in sorted(losses_by_step.items()):
        if best_value is None or best_value - current_value >= min_delta:
            best_value = current_value
            bad_validation_count = 0
        else:
            bad_validation_count += 1

    should_stop = (
        current_value is not None
        and current_step >= min_steps
        and bad_validation_count >= patience
    )
    return {
        "metric": metric,
        "best_value": best_value,
        "current_value": current_value,
        "current_step": current_step,
        "bad_validation_count": bad_validation_count,
        "validation_count": len(losses_by_step),
        "min_delta": float(min_delta),
        "patience": int(patience),
        "min_steps": int(min_steps),
        "should_stop": should_stop,
    }


def _read_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


_DISTRIBUTED_SUM_KEYS = {
    "execution_reason_histogram",
    "execution_selected_boundary_count",
    "execution_steps_histogram",
    "ground_truth_boundary_crossing_count",
    "ground_truth_boundary_crossing_valid_count",
    "route_confusion_matrix",
    "route_label_coverage",
    "route_valid_by_position",
    "sample_count",
    "schedule_edit_distance_sum",
}
_DISTRIBUTED_FIRST_KEYS = {
    "ablation",
    "cache_fingerprint",
    "learning_rates",
    "model_variant",
    "parameter_counts",
    "stage",
    "step",
    "view_order",
}


def _elementwise(values, reducer):
    if not values:
        return []
    return [
        _elementwise([value[index] for value in values], reducer)
        if isinstance(values[0][index], list)
        else reducer([value[index] for value in values])
        for index in range(len(values[0]))
    ]


def _weighted_elementwise(values, weights):
    if not values:
        return []
    if isinstance(values[0], list):
        return [
            _weighted_elementwise(
                [value[index] for value in values],
                weights,
            )
            for index in range(len(values[0]))
        ]
    total_weight = sum(weights)
    return sum(float(value) * weight for value, weight in zip(values, weights)) / max(
        total_weight, 1
    )


def _aggregate_distributed_values(key: str, values: list[Any]):
    present = [value for value in values if value is not None]
    if not present:
        return None
    first = present[0]
    if key in _DISTRIBUTED_FIRST_KEYS:
        return first
    if isinstance(first, dict):
        if "count" in first:
            total = sum(int(value.get("count", 0)) for value in present)
            output = {"count": total}
            for child_key in sorted(set().union(*(value.keys() for value in present)) - {"count"}):
                child_values = [value.get(child_key) for value in present]
                numeric = [value for value in child_values if isinstance(value, (int, float))]
                if numeric:
                    weighted = sum(
                        float(value.get(child_key, 0.0)) * int(value.get("count", 0))
                        for value in present
                        if isinstance(value.get(child_key), (int, float))
                    )
                    output[child_key] = weighted / max(total, 1)
                else:
                    output[child_key] = _aggregate_distributed_values(child_key, child_values)
            return output
        return {
            child_key: _aggregate_distributed_values(
                child_key,
                [value.get(child_key) for value in present],
            )
            for child_key in sorted(set().union(*(value.keys() for value in present)))
        }
    if isinstance(first, list):
        if not all(isinstance(value, list) and len(value) == len(first) for value in present):
            return first
        if first and isinstance(first[0], dict):
            return [
                _aggregate_distributed_values(key, [value[index] for value in present])
                for index in range(len(first))
            ]
        reducer = sum if key in _DISTRIBUTED_SUM_KEYS else lambda items: sum(items) / len(items)
        if first and isinstance(first[0], (int, float, list)):
            return _elementwise(present, reducer)
        return first
    if isinstance(first, bool):
        return all(bool(value) for value in present)
    if isinstance(first, (int, float)):
        if key.endswith("_count") or key in _DISTRIBUTED_SUM_KEYS:
            return sum(present)
        if "latency" in key or key.endswith("_mib_max"):
            return max(present)
        return sum(float(value) for value in present) / len(present)
    if isinstance(first, str):
        return first if all(value == first for value in present) else "mixed"
    return first


def aggregate_distributed_records(
    records: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Merge per-rank training records into the existing rank-0 JSONL contract."""

    if not records:
        raise ValueError("Cannot aggregate an empty distributed record list.")
    ignored = {"rank", "rank_episode_ids", "resource_metrics"}
    keys = sorted(set().union(*(record.keys() for record in records)) - ignored)
    sample_counts = [max(0, int(record.get("sample_count", 1))) for record in records]
    total_samples = sum(sample_counts)
    output = {}
    for key in keys:
        values = [record.get(key) for record in records]
        present = [value for value in values if value is not None]
        if (
            key not in _DISTRIBUTED_FIRST_KEYS
            and key not in _DISTRIBUTED_SUM_KEYS
            and not key.endswith("_count")
            and "latency" not in key
            and present
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                for value in present
            )
        ):
            weighted = sum(
                float(value) * sample_counts[index]
                for index, value in enumerate(values)
                if value is not None
            )
            weight = sum(
                sample_counts[index]
                for index, value in enumerate(values)
                if value is not None
            )
            output[key] = weighted / max(weight, 1)
        elif (
            key not in _DISTRIBUTED_FIRST_KEYS
            and key not in _DISTRIBUTED_SUM_KEYS
            and present
            and len(present) == len(values)
            and all(
                isinstance(value, list)
                and len(value) == len(present[0])
                for value in present
            )
            and (
                not present[0]
                or isinstance(present[0][0], (int, float, list))
            )
        ):
            output[key] = _weighted_elementwise(present, sample_counts)
        else:
            output[key] = _aggregate_distributed_values(key, values)

    # Unknown labels can give ranks different denominators even with equal
    # per-device batches, so position accuracy is weighted by valid labels.
    valid_by_position = output.get("route_valid_by_position")
    if isinstance(valid_by_position, list):
        weighted_accuracy = []
        for position, total_valid in enumerate(valid_by_position):
            numerator = 0.0
            for record in records:
                counts = record.get("route_valid_by_position", [])
                accuracies = record.get("route_accuracy_by_position", [])
                if position < len(counts) and position < len(accuracies):
                    numerator += float(accuracies[position]) * int(counts[position])
            weighted_accuracy.append(numerator / max(int(total_valid), 1))
        output["route_accuracy_by_position"] = weighted_accuracy
        if weighted_accuracy:
            output["current_skill_accuracy"] = weighted_accuracy[0]
        future_valid = sum(int(value) for value in valid_by_position[1:])
        if future_valid:
            output["future_position_accuracy"] = sum(
                weighted_accuracy[index] * int(valid_by_position[index])
                for index in range(1, len(weighted_accuracy))
            ) / future_valid
    boundary_tp = int(output.get("boundary_true_positive_count", 0))
    boundary_predicted = int(output.get("boundary_predicted_positive_count", 0))
    boundary_target = int(output.get("boundary_target_positive_count", 0))
    boundary_precision = boundary_tp / max(boundary_predicted, 1)
    boundary_recall = boundary_tp / max(boundary_target, 1)
    output["boundary_precision"] = boundary_precision
    output["boundary_recall"] = boundary_recall
    output["boundary_f1"] = (
        2.0 * boundary_precision * boundary_recall
        / max(boundary_precision + boundary_recall, 1e-8)
    )
    edit_count = int(output.get("schedule_edit_distance_count", 0))
    if edit_count:
        output["schedule_edit_distance"] = float(
            output.get("schedule_edit_distance_sum", 0.0)
        ) / edit_count
    crossing_valid = int(output.get("ground_truth_boundary_crossing_valid_count", 0))
    if crossing_valid:
        output["ground_truth_boundary_crossing_rate"] = int(
            output.get("ground_truth_boundary_crossing_count", 0)
        ) / crossing_valid
    output["sample_count"] = total_samples
    output["distributed_contract"] = contract
    output["rank_resource_metrics"] = sorted(
        [record.get("resource_metrics", {"rank": record.get("rank")}) for record in records],
        key=lambda value: int(value.get("rank", 0)),
    )
    output["rank_episode_ids"] = sorted(
        [
            {
                "rank": int(record.get("rank", 0)),
                "episode_ids": list(record.get("rank_episode_ids", [])),
            }
            for record in records
        ],
        key=lambda value: value["rank"],
    )
    return output


def distributed_episode_overlap(records: list[dict[str, Any]]) -> list[str]:
    owners: dict[str, int] = {}
    overlap = set()
    for record in records:
        rank = int(record.get("rank", 0))
        for episode_id in set(record.get("rank_episode_ids", [])):
            previous = owners.setdefault(str(episode_id), rank)
            if previous != rank:
                overlap.add(str(episode_id))
    return sorted(overlap)


def evaluate_flow_model(cfg, model, dataloader, *, stage: str, step: int):
    """Evaluate a fixed number of validation batches without perturbing training RNG."""

    torch = require_torch()
    if dataloader is None:
        return None
    validation_cfg = cfg.get("validation", {})
    num_batches = int(validation_cfg.get("num_batches", 32))
    if num_batches < 1:
        raise ValueError("validation.num_batches must be positive.")
    seed = int(validation_cfg.get("seed", 1701))
    device = resolve_device(cfg)
    precision = str(cfg["training"].get("precision", "bf16"))
    devices = [torch.cuda.current_device()] if device.startswith("cuda") else []
    sums: dict[str, float] = {}
    horizon_sums: dict[str, dict[str, float]] = {}
    episodes = set()
    batches = 0
    model.eval()
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.manual_seed(seed)
            if device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
            for batch_index, batch in enumerate(dataloader):
                if batch_index >= num_batches:
                    break
                with autocast_context(device, precision):
                    outputs = model(
                        batch,
                        action_condition_mode="ground_truth",
                        teacher_forcing_probability=1.0,
                        route_mode="predicted" if stage == "nominal_flow_pretrain" else "oracle",
                        gumbel_temperature=1.0,
                        flow_seed=seed + batch_index,
                        compute_teacher_targets=True,
                        compute_residual=stage != "nominal_flow_pretrain",
                        compute_route_diagnostics=False,
                    )
                    losses = flow_wam_skill_losses(
                        outputs,
                        batch,
                        cfg["loss_weights"],
                        stage=stage,
                    )
                mechanism = _mechanism_metrics(outputs, batch)
                for name, value in losses.items():
                    if hasattr(value, "numel") and value.numel() == 1:
                        sums[name] = sums.get(name, 0.0) + float(value.detach())
                for horizon, values in mechanism.get("future_horizon_metrics", {}).items():
                    target = horizon_sums.setdefault(horizon, {})
                    for name, value in values.items():
                        target[name] = target.get(name, 0.0) + float(value)
                episodes.update(str(value) for value in batch.get("episode_id", []))
                batches += 1
    finally:
        model.train()
        configure_flow_stage(model, stage)
    if batches == 0:
        raise RuntimeError("Validation partition yielded no valid windows.")
    return {
        "kind": "flow_wam_validation",
        "step": int(step),
        "stage": stage,
        "batches": batches,
        "unique_episodes": len(episodes),
        "episode_partition": "validation",
        "validation_fraction": float(cfg["data"].get("validation_fraction", 0.0)),
        "split_seed": int(cfg["data"].get("split_seed", 17)),
        "seed": seed,
        "metrics": {name: value / batches for name, value in sorted(sums.items())},
        "future_horizon_metrics": {
            horizon: {name: value / batches for name, value in sorted(values.items())}
            for horizon, values in sorted(horizon_sums.items(), key=lambda item: int(item[0]))
        },
    }


def _run_flow_training_impl(
    cfg: dict[str, Any],
    *,
    stage: str,
    resume: str | None = None,
    init_checkpoint: str | None = None,
    route_mode_override: str | None = None,
    distributed: DistributedContext,
    allow_world_size_change: bool = False,
):
    torch = require_torch()
    configured_stage = cfg.get("stage")
    if configured_stage is not None and configured_stage != stage:
        raise ValueError(f"Config stage {configured_stage!r} does not match requested stage {stage!r}.")
    if cfg.get("ablation", {}).get("analysis_only", False):
        raise ValueError("This ablation config is analysis-only and cannot start training.")
    if stage == "nominal_flow_pretrain" and init_checkpoint:
        raise ValueError("Stage 1 must not use an initialization checkpoint.")
    if stage in STAGE_PREDECESSOR and not (init_checkpoint or resume):
        raise ValueError(
            f"Stage {stage} requires a {STAGE_PREDECESSOR[stage]} initialization checkpoint or same-stage resume."
        )
    cfg["training"]["device"] = distributed.device
    cfg["distributed_contract"] = distributed_contract(cfg, distributed)
    resource_guard_cfg = cfg["training"].get("distributed", {})
    memory_guard_fraction = float(resource_guard_cfg.get("memory_guard_fraction", 0.80))
    gpu_memory_guard_fraction = float(
        resource_guard_cfg.get("gpu_memory_guard_fraction", 0.85)
    )
    if not 0 < memory_guard_fraction <= 1 or not 0 < gpu_memory_guard_fraction <= 1:
        raise ValueError("Distributed resource guard fractions must be in (0,1].")
    resource_baseline = process_resource_metrics(distributed)
    require_cgroup_metrics = bool(
        resource_guard_cfg.get("require_cgroup_metrics", False)
    ) and distributed.enabled
    enforce_resource_metric_contract(
        distributed,
        resource_baseline,
        require_cgroup=require_cgroup_metrics,
        require_gpu=distributed.enabled and distributed.device.startswith("cuda"),
    )
    enforce_cgroup_memory_guard(
        distributed,
        resource_baseline,
        memory_guard_fraction,
    )
    feature_backend = cfg.get("data", {}).get("backend", "rlds") == "mowe_feature_store_v1"
    if feature_backend:
        resolve_feature_store_contract(cfg)
    else:
        identity = None
        if distributed.is_main:
            identity = resolve_original_openvla_identity(
                cfg.get("backbone", {}).get("checkpoint")
                or cfg.get("backbone", {}).get("vla_path"),
                revision=cfg.get("backbone", {}).get("revision"),
                repo_id=cfg.get("backbone", {}).get(
                    "repo_id", "openvla/openvla-7b"
                ),
            )
        identity = distributed.broadcast_object(identity)
        cfg["backbone"].update(
            {
                "identity": identity,
                "repo_id": identity["repo_id"],
                "revision": identity["revision"],
            }
        )
    output_dir = Path(cfg.get("output_dir", f"outputs/train/{stage}"))
    if distributed.is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
    distributed.barrier()
    set_seed(int(cfg.get("seed", 7)), distributed.rank)
    include_teacher = not feature_backend and not bool(cfg.get("teacher", {}).get("cache_path"))
    raw_model = build_flow_policy(cfg, include_teacher=include_teacher)
    raw_model.train()
    configure_flow_stage(raw_model, stage)
    dataloader = build_flow_dataloader(cfg, raw_model, distributed)
    validation_dataloader = build_flow_validation_dataloader(cfg, raw_model, distributed)
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, EpisodeAwareDistributedSampler):
        assignment_reports = distributed.all_gather_objects(
            sampler.assignment_report(include_skill_counts=True)
        )
        cfg["feature_store_assignment_reports"] = assignment_reports
        cfg["feature_store_assignment_validation"] = (
            validate_episode_assignment_reports(
                dataloader.dataset,
                assignment_reports,
                world_size=distributed.world_size,
            )
        )
    setup_resources = process_resource_metrics(distributed)
    enforce_cgroup_memory_guard(
        distributed,
        setup_resources,
        memory_guard_fraction,
    )
    enforce_gpu_memory_guard(
        distributed,
        setup_resources,
        gpu_memory_guard_fraction,
    )
    enforce_no_new_oom_events(distributed, setup_resources, resource_baseline)
    cfg["launch_resource_metrics_by_rank"] = distributed.all_gather_objects(setup_resources)
    runtime_identity = consolidate_runtime_identities(
        distributed.all_gather_objects(
            local_runtime_identity(distributed, setup_resources)
        ),
        require_cuda=distributed.device.startswith("cuda"),
        require_node_identity=require_cgroup_metrics,
    )
    cfg["runtime_identity"] = runtime_identity
    skill_cfg = load_config(cfg["skill_expert_config"])
    validate_skill_config(skill_cfg)
    validate_skill_audit_contract(cfg, skill_cfg)
    cfg["skill_experts_resolved"] = skill_cfg
    cached_teacher_transform = (
        cfg.get("feature_store_contract", {})
        .get("source_contract", {})
        .get("teacher_transform_metadata")
    )
    cfg["teacher"]["transform_metadata"] = cached_teacher_transform or teacher_transform_metadata(
        cfg["teacher"].get("checkpoint"),
        cfg["backbone"].get("image_resolution", [224, 224]),
        int(cfg["teacher"].get("spatial_grid", 4)),
    )
    cfg["parameter_counts"] = {
        name: sum(parameter.numel() for parameter in getattr(raw_model, name).parameters())
        for name in FLOW_COMPONENTS
    }
    cfg["parameter_counts"]["trainable_total"] = sum(
        parameter.numel() for parameter in raw_model.parameters() if parameter.requires_grad
    )
    optimizer = build_flow_optimizer(cfg, raw_model)
    scheduler = build_warmup_cosine_scheduler(cfg, optimizer)
    precision = str(cfg["training"].get("precision", "bf16"))
    device = resolve_device(cfg)
    scaler = make_grad_scaler(device, precision)
    start_step = 0
    schedule_state = {}
    checkpoint_metadata = None
    checkpoint_mode = "none"
    if init_checkpoint:
        checkpoint_metadata = {}
        checkpoint_mode = "init"
        start_step, schedule_state = load_flow_checkpoint(
            init_checkpoint,
            raw_model,
            resume=False,
            metadata_out=checkpoint_metadata,
            allowed_stages={STAGE_PREDECESSOR[stage]},
            distributed=distributed,
            current_cfg=cfg,
            allow_world_size_change=allow_world_size_change,
        )
        validate_checkpoint_contract(checkpoint_metadata, cfg)
        start_step = 0
    if resume:
        if init_checkpoint:
            raise ValueError("Use either init_checkpoint or resume, not both.")
        checkpoint_metadata = {}
        checkpoint_mode = "resume"
        start_step, schedule_state = load_flow_checkpoint(
            resume,
            raw_model,
            optimizer,
            scheduler,
            scaler,
            resume=True,
            metadata_out=checkpoint_metadata,
            allowed_stages={stage},
            distributed=distributed,
            current_cfg=cfg,
            allow_world_size_change=allow_world_size_change,
        )
        validate_checkpoint_contract(checkpoint_metadata, cfg)
        validate_resume_schedule_contract(checkpoint_metadata, cfg)
        sampler_states = checkpoint_metadata.get("sampler_state_by_rank") or []
        if isinstance(sampler, EpisodeAwareDistributedSampler) and len(sampler_states) == distributed.world_size:
            sampler.load_state_dict(sampler_states[distributed.rank])

    max_steps = int(cfg["training"].get("max_steps", 1))
    if max_steps < 1:
        raise ValueError("training.max_steps must be positive.")
    stop_step = int(cfg["training"].get("stop_step", max_steps))
    if stop_step < start_step or stop_step > max_steps:
        raise ValueError(
            f"training.stop_step must be between resumed step {start_step} and max_steps {max_steps}."
        )
    if feature_backend:
        enforce_long_run_readiness(
            cfg,
            stage=stage,
            world_size=distributed.world_size,
            start_step=start_step,
            stop_step=stop_step,
            checkpoint_metadata=checkpoint_metadata,
            checkpoint_mode=checkpoint_mode,
            runtime_identity=runtime_identity,
        )

    train_model = raw_model
    if distributed.enabled:
        ddp_config = cfg["training"].get("distributed", {})
        sync_buffers = bool(ddp_config.get("broadcast_buffers", False))
        ddp_kwargs = {
            "device_ids": (
                [distributed.local_rank]
                if distributed.device.startswith("cuda")
                else None
            ),
            "output_device": (
                distributed.local_rank
                if distributed.device.startswith("cuda")
                else None
            ),
            "find_unused_parameters": bool(
                ddp_config.get("find_unused_parameters", False)
            ),
        }
        ddp_parameters = inspect.signature(
            torch.nn.parallel.DistributedDataParallel
        ).parameters
        if "forward_sync_buffers" in ddp_parameters:
            # PyTorch 2.13 deprecates broadcast_buffers in favor of this
            # forward-only equivalent. Initial synchronization remains enabled.
            ddp_kwargs["forward_sync_buffers"] = sync_buffers
        else:
            ddp_kwargs["broadcast_buffers"] = sync_buffers
        train_model = torch.nn.parallel.DistributedDataParallel(raw_model, **ddp_kwargs)
    post_ddp_resources = process_resource_metrics(distributed)
    enforce_cgroup_memory_guard(
        distributed,
        post_ddp_resources,
        memory_guard_fraction,
    )
    enforce_gpu_memory_guard(
        distributed,
        post_ddp_resources,
        gpu_memory_guard_fraction,
    )
    enforce_no_new_oom_events(distributed, post_ddp_resources, resource_baseline)
    cfg["post_ddp_resource_metrics_by_rank"] = distributed.all_gather_objects(
        post_ddp_resources
    )

    class_weights = torch.as_tensor(
        skill_cfg.get("class_weights_inverse_sqrt", [1.0] * 7),
        device=device,
        dtype=torch.float32,
    )
    accumulation = int(cfg["training"].get("grad_accumulation_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    log_freq = int(cfg["training"].get("log_freq", 10))
    save_freq = int(cfg["training"].get("save_freq", 500))
    iterator = iter(dataloader)
    optimizer.zero_grad(set_to_none=True)
    step = start_step
    micro_step = 0
    log_path = output_dir / "train_log.jsonl"
    validation_log_path = output_dir / "validation_log.jsonl"
    validation_cfg = cfg.get("validation", {})
    validation_freq = int(validation_cfg.get("eval_freq", 500))
    early_stopping_cfg = validation_cfg.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_metric = str(early_stopping_cfg.get("metric", "total_loss"))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 1e-4))
    early_stopping_patience = int(early_stopping_cfg.get("patience", 5))
    early_stopping_min_steps = int(early_stopping_cfg.get("min_steps", 5000))
    early_stopping_report_path = output_dir / "early_stopping.json"
    if bool(validation_cfg.get("enabled", False)) and validation_freq < 1:
        raise ValueError("validation.eval_freq must be positive.")
    if early_stopping_enabled and not bool(validation_cfg.get("enabled", False)):
        raise ValueError("validation must be enabled when early stopping is enabled.")
    if early_stopping_enabled:
        validation_loss_early_stopping_state(
            [],
            stage=stage,
            metric=early_stopping_metric,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            min_steps=early_stopping_min_steps,
        )
    if accumulation < 1 or log_freq < 1 or save_freq < 1:
        raise ValueError(
            "grad_accumulation_steps, log_freq, and save_freq must be positive."
        )
    if distributed.is_main:
        _write_json_atomic(output_dir / "config_resolved.json", cfg)
    distributed.barrier()
    if bool(validation_cfg.get("enabled", False)) and bool(validation_cfg.get("run_at_start", True)):
        distributed.barrier()
        if distributed.is_main:
            validation_record = evaluate_flow_model(
                cfg,
                raw_model,
                validation_dataloader,
                stage=stage,
                step=step,
            )
            _write_jsonl(validation_log_path, validation_record)
            print(json.dumps(_jsonable(validation_record), sort_keys=True), flush=True)
        distributed.barrier()
        validation_resources = process_resource_metrics(distributed)
        enforce_cgroup_memory_guard(
            distributed, validation_resources, memory_guard_fraction
        )
        enforce_gpu_memory_guard(
            distributed, validation_resources, gpu_memory_guard_fraction
        )
        enforce_no_new_oom_events(
            distributed, validation_resources, resource_baseline
        )
    while step < stop_step:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(dataloader)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise RuntimeError("Flow-WAM dataset yielded no valid windows.") from exc
        if micro_step % accumulation == 0:
            optimizer_step_start = time.perf_counter()

        forcing = teacher_forcing_probability(
            step,
            max_steps,
            nominal_start_ratio=float(cfg["action_condition"].get("nominal_start_ratio", 0.30)),
            nominal_end_ratio=float(cfg["action_condition"].get("nominal_end_ratio", 0.70)),
            final_nominal_probability=float(cfg["action_condition"].get("final_nominal_probability", 0.80)),
        )
        router_state = temporal_router_schedule(
            step,
            max_steps,
            predicted_start_ratio=float(cfg["router"].get("predicted_route_start_ratio", 0.20)),
            predicted_end_ratio=float(cfg["router"].get("predicted_route_end_ratio", 0.70)),
            start_temperature=float(cfg["router"].get("gumbel_temperature_start", 1.0)),
            end_temperature=float(cfg["router"].get("gumbel_temperature_end", 0.1)),
        )
        configured_route_mode = str(cfg.get("route_mode", "scheduled"))
        effective_override = route_mode_override or (
            configured_route_mode if configured_route_mode != "scheduled" else None
        )
        if stage == "nominal_flow_pretrain":
            route_mode = "predicted"
        elif stage == "expert_warmstart":
            route_mode = "oracle"
        elif effective_override in {"oracle", "predicted", "soft"}:
            route_mode = effective_override
        elif effective_override == "st_gumbel":
            route_mode = "st_gumbel"
        else:
            route_random = random.Random(int(cfg.get("seed", 7)) + step).random()
            route_mode = "oracle" if route_random < router_state["oracle_route_probability"] else "st_gumbel"
        schedule_state = {**router_state, "route_mode": route_mode, "step": step}
        model_batch = batch
        if cfg.get("ablation", {}).get("shuffle_skill_labels", False):
            model_batch = dict(batch)
            labels = batch["expert_skill_labels"].clone()
            mask = batch["expert_skill_mask"].bool()
            values = labels[mask]
            if values.numel() > 1:
                values = values[torch.randperm(values.numel())]
                labels[mask] = values
            model_batch["expert_skill_labels"] = labels
        will_optimizer_step = (micro_step + 1) % accumulation == 0
        next_step = step + 1
        compute_route_diagnostics = will_optimizer_step and (
            next_step % log_freq == 0 or next_step == 1
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        forward_start = time.perf_counter()
        sync_context = (
            train_model.no_sync()
            if distributed.enabled and not will_optimizer_step
            else nullcontext()
        )
        with sync_context:
            with autocast_context(device, precision):
                outputs = train_model(
                    model_batch,
                    action_condition_mode="scheduled",
                    teacher_forcing_probability=forcing,
                    route_mode=route_mode,
                    gumbel_temperature=router_state["gumbel_temperature"],
                    flow_seed=(
                        int(cfg["flow"].get("deterministic_seed", 7))
                        + step * distributed.world_size
                        + distributed.rank
                    ),
                    compute_teacher_targets=True,
                    compute_residual=stage != "nominal_flow_pretrain",
                    compute_route_diagnostics=compute_route_diagnostics,
                )
                losses = flow_wam_skill_losses(
                    outputs,
                    model_batch,
                    cfg["loss_weights"],
                    schedule_state={"enable_load_balance": route_mode != "oracle"},
                    stage=stage,
                    class_weights=class_weights,
                )
                scaled_loss = losses["total_loss"] / accumulation
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            forward_latency_ms = (time.perf_counter() - forward_start) * 1000.0
            scaler.scale(scaled_loss).backward()
        micro_step += 1
        if micro_step % accumulation:
            continue

        scaler.unscale_(optimizer)
        gradient_norms = _gradient_norms(raw_model)
        motor_expert_gradient_norms = _motor_expert_gradient_norms(raw_model)
        trainable = [parameter for parameter in raw_model.parameters() if parameter.requires_grad]
        global_norm = torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        step_latency_ms = (time.perf_counter() - optimizer_step_start) * 1000.0
        step += 1
        schedule_state["step"] = step

        if step % log_freq == 0 or step == 1:
            future_errors = (
                (outputs["future_latents"].float() - outputs["future_latent_targets"].float())
                .square()
                .mean(dim=(0, 2, 3))
                .detach()
                .cpu()
                .tolist()
                if "future_latent_targets" in outputs
                else []
            )
            record = {
                "rank": distributed.rank,
                "sample_count": int(model_batch["target_actions"].shape[0]),
                "rank_episode_ids": sorted(
                    {str(value) for value in model_batch.get("episode_id", [])}
                ),
                "step": step,
                "stage": stage,
                "model_variant": "flow_wam_skill_moe",
                "total_loss": float(losses["total_loss"].detach()),
                "learning_rates": {group["name"]: group["lr"] for group in optimizer.param_groups},
                "gradient_norm": float(global_norm),
                "component_gradient_norms": gradient_norms,
                "motor_expert_gradient_norms": motor_expert_gradient_norms,
                "forward_latency_ms": forward_latency_ms,
                "step_latency_ms": step_latency_ms,
                "parameter_counts": cfg["parameter_counts"],
                "ablation": cfg.get("ablation", {}).get("name", "main"),
                "future_shuffle_router_change_rate": float(
                    outputs.get(
                        "future_shuffle_router_change_rate",
                        outputs["router_logits"].new_tensor(0.0),
                    ).detach()
                ),
                "future_shuffle_router_logit_l1": float(
                    outputs.get(
                        "future_shuffle_router_logit_l1",
                        outputs["router_logits"].new_tensor(0.0),
                    ).detach()
                ),
                "route_mode_diagnostics": outputs.get("route_mode_diagnostics", {}),
                "action_condition_source": (
                    "ground_truth"
                    if bool(outputs["teacher_forcing_mask"].all())
                    else "nominal"
                    if not bool(outputs["teacher_forcing_mask"].any())
                    else "mixed"
                ),
                "action_distance_gate_mean": float(
                    outputs["action_distance_gate"].float().mean().detach()
                ),
                "router_entropy": float(outputs["router_entropy"].float().mean().detach()),
                "route_source": outputs["route_source"],
                **router_state,
                **_route_metrics(outputs, model_batch),
                **_mechanism_metrics(outputs, model_batch),
                "null_motion_zero_violation_count": int(outputs["null_motion_zero_violation_count"]),
                "execution_steps_mean": float(outputs["execution_steps"].float().mean().detach()),
                "motion_residual_norm": float(
                    outputs["residual_motion"].float().norm(dim=-1).mean().detach()
                ),
                "motion_residual_norm_max": float(
                    outputs["residual_motion"].float().norm(dim=-1).max().detach()
                ),
                "motion_residual_clip_fraction": float(
                    outputs["residual_clip_fraction"].detach()
                ),
                "residual_target_clip_fraction": float(
                    outputs["residual_target_clip_fraction"].detach()
                ),
                "route_world_token_norm": float(
                    outputs["route_world_tokens"].float().norm(dim=-1).mean().detach()
                ),
                "route_world_token_variance": float(
                    outputs["route_world_tokens"].float().var().detach()
                ),
                "future_horizon_errors": {
                    str(horizon): float(error)
                    for horizon, error in zip(cfg["data"]["future_horizons"], future_errors)
                },
                "cache_fingerprint": cfg.get("teacher", {}).get("cache_path"),
                "resource_metrics": process_resource_metrics(distributed),
                **{name: float(value.detach()) for name, value in losses.items()},
            }
            gathered_records = distributed.all_gather_objects(_jsonable(record))
            overlap = distributed_episode_overlap(gathered_records)
            if overlap:
                raise RuntimeError(
                    "Distributed RLDS shard overlap detected for episode IDs: "
                    f"{overlap[:8]}"
                )
            if distributed.is_main:
                aggregated = aggregate_distributed_records(
                    gathered_records,
                    cfg["distributed_contract"],
                )
                _write_jsonl(log_path, aggregated)
                print(json.dumps(_jsonable(aggregated), sort_keys=True), flush=True)
            enforce_cgroup_memory_guard(
                distributed,
                record["resource_metrics"],
                memory_guard_fraction,
            )
            enforce_gpu_memory_guard(
                distributed,
                record["resource_metrics"],
                gpu_memory_guard_fraction,
            )
            enforce_no_new_oom_events(
                distributed,
                record["resource_metrics"],
                resource_baseline,
            )
        if step % save_freq == 0 or step in {max_steps, stop_step}:
            rng_state_by_rank = distributed.all_gather_objects(local_rng_state(distributed))
            local_sampler_state = sampler.state_dict() if isinstance(
                sampler, EpisodeAwareDistributedSampler
            ) else None
            sampler_state_by_rank = distributed.all_gather_objects(local_sampler_state)
            if distributed.is_main:
                save_flow_checkpoint(
                    output_dir / "checkpoint_latest.pt",
                    raw_model,
                    optimizer,
                    scheduler,
                    scaler,
                    step,
                    cfg,
                    stage,
                    schedule_state,
                    distributed_metadata=cfg["distributed_contract"],
                    rng_state_by_rank=rng_state_by_rank,
                    sampler_state_by_rank=sampler_state_by_rank,
                )
            distributed.barrier()
        if bool(validation_cfg.get("enabled", False)) and (
            step % validation_freq == 0 or step == stop_step
        ):
            distributed.barrier()
            early_stopping_state = None
            if distributed.is_main:
                validation_record = evaluate_flow_model(
                    cfg,
                    raw_model,
                    validation_dataloader,
                    stage=stage,
                    step=step,
                )
                _write_jsonl(validation_log_path, validation_record)
                print(json.dumps(_jsonable(validation_record), sort_keys=True), flush=True)
                if early_stopping_enabled:
                    early_stopping_state = validation_loss_early_stopping_state(
                        _read_jsonl_records(validation_log_path),
                        stage=stage,
                        metric=early_stopping_metric,
                        min_delta=early_stopping_min_delta,
                        patience=early_stopping_patience,
                        min_steps=early_stopping_min_steps,
                    )
            early_stopping_state = distributed.broadcast_object(early_stopping_state)
            distributed.barrier()
            validation_resources = process_resource_metrics(distributed)
            enforce_cgroup_memory_guard(
                distributed, validation_resources, memory_guard_fraction
            )
            enforce_gpu_memory_guard(
                distributed, validation_resources, gpu_memory_guard_fraction
            )
            enforce_no_new_oom_events(
                distributed, validation_resources, resource_baseline
            )
            if (
                early_stopping_state
                and bool(early_stopping_state["should_stop"])
                and step < max_steps
            ):
                rng_state_by_rank = distributed.all_gather_objects(
                    local_rng_state(distributed)
                )
                local_sampler_state = sampler.state_dict() if isinstance(
                    sampler, EpisodeAwareDistributedSampler
                ) else None
                sampler_state_by_rank = distributed.all_gather_objects(
                    local_sampler_state
                )
                if distributed.is_main:
                    save_flow_checkpoint(
                        output_dir / "checkpoint_latest.pt",
                        raw_model,
                        optimizer,
                        scheduler,
                        scaler,
                        step,
                        cfg,
                        stage,
                        schedule_state,
                        distributed_metadata=cfg["distributed_contract"],
                        rng_state_by_rank=rng_state_by_rank,
                        sampler_state_by_rank=sampler_state_by_rank,
                    )
                    report = {
                        "format": "mowe_validation_loss_early_stop_v1",
                        "stage": stage,
                        "reason": "validation_loss_plateau",
                        "stopped_early": True,
                        "step": int(step),
                        "max_steps": int(max_steps),
                        **early_stopping_state,
                    }
                    _write_json_atomic(early_stopping_report_path, report)
                    print(json.dumps(_jsonable(report), sort_keys=True), flush=True)
                distributed.barrier()
                break
    if distributed.is_main and stop_step == max_steps and step >= max_steps:
        final_state = validation_loss_early_stopping_state(
            _read_jsonl_records(validation_log_path),
            stage=stage,
            metric=early_stopping_metric,
            min_delta=early_stopping_min_delta,
            patience=early_stopping_patience,
            min_steps=early_stopping_min_steps,
        ) if early_stopping_enabled else {}
        _write_json_atomic(
            early_stopping_report_path,
            {
                "format": "mowe_validation_loss_early_stop_v1",
                "stage": stage,
                "reason": "max_steps",
                "stopped_early": False,
                "step": int(step),
                "max_steps": int(max_steps),
                **final_state,
            },
        )
    distributed.barrier()
    return output_dir / "checkpoint_latest.pt"


def run_flow_training(
    cfg: dict[str, Any],
    *,
    stage: str,
    resume: str | None = None,
    init_checkpoint: str | None = None,
    route_mode_override: str | None = None,
    allow_world_size_change: bool = False,
):
    """Run one flow stage with transparent single-process or torchrun DDP execution."""

    distributed = initialize_distributed(cfg)
    try:
        return _run_flow_training_impl(
            cfg,
            stage=stage,
            resume=resume,
            init_checkpoint=init_checkpoint,
            route_mode_override=route_mode_override,
            distributed=distributed,
            allow_world_size_change=allow_world_size_change,
        )
    finally:
        distributed.close()
