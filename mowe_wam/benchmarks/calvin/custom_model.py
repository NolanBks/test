"""Environment-configured CALVIN ``CustomModel`` implementation.

The official evaluator can import this class after the CALVIN environment and
MoWE project are placed on ``PYTHONPATH``.  It intentionally refuses a LIBERO
action-normalized checkpoint.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

from mowe_wam.backbones import resolve_original_openvla_identity
from mowe_wam.benchmarks.calvin.action_adapter import CalvinActionAdapter
from mowe_wam.benchmarks.calvin.policy_adapter import CalvinTemporalPolicyAdapter
from mowe_wam.training.flow_runtime import (
    build_flow_policy,
    deep_update,
    load_flow_checkpoint,
    read_flow_checkpoint_metadata,
    validate_backbone_identifier,
)
from mowe_wam.utils.config import load_config


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"CALVIN CustomModel requires environment variable {name}.")
    return value


def _assert_checkpoint_action_contract(metadata: dict, action_adapter: CalvinActionAdapter) -> None:
    statistics = metadata.get("data_contract", {}).get("joint_action_statistics")
    if not isinstance(statistics, dict):
        raise ValueError(
            "CALVIN evaluation requires a fine-tuned checkpoint with recorded CALVIN action statistics."
        )
    observed_low = [float(value) for value in statistics.get("q01", [])[:6]]
    observed_high = [float(value) for value in statistics.get("q99", [])[:6]]
    if observed_low != list(action_adapter.motion_q01) or observed_high != list(
        action_adapter.motion_q99
    ):
        raise ValueError(
            "Checkpoint action normalization does not match the CALVIN adapter; refusing a likely "
            "LIBERO-to-CALVIN silent action mismatch."
        )


def load_custom_model(
    *,
    benchmark_config: str | Path,
    flow_checkpoint: str | Path,
    backbone_checkpoint: str | Path,
    backbone_revision: str,
    local_config: str | Path = "configs/mowe_wam/train_flow_wam_skill_moe.yaml",
):
    benchmark_cfg = load_config(benchmark_config)
    local_cfg = load_config(local_config)
    metadata = read_flow_checkpoint_metadata(flow_checkpoint)
    saved_cfg = metadata.get("config")
    if not isinstance(saved_cfg, dict) or not saved_cfg:
        raise ValueError("Flow checkpoint does not contain a resolved training config.")
    if metadata.get("stage") != "joint":
        raise ValueError("Formal CALVIN evaluation requires a Stage 3 joint checkpoint.")
    requested_identity = resolve_original_openvla_identity(
        backbone_checkpoint,
        revision=backbone_revision,
        repo_id=local_cfg.get("backbone", {}).get("repo_id", "openvla/openvla-7b"),
    )
    validate_backbone_identifier(
        metadata,
        backbone_checkpoint,
        requested_identity=requested_identity,
    )
    action_adapter = CalvinActionAdapter.from_config(benchmark_cfg["action"])
    _assert_checkpoint_action_contract(metadata, action_adapter)
    cfg = copy.deepcopy(saved_cfg)
    deep_update(
        cfg,
        {
            "backbone": {
                "mode": "online_openvla",
                "feature_source": "pre_action_context",
                "checkpoint": str(backbone_checkpoint),
                "repo_id": requested_identity["repo_id"],
                "revision": requested_identity["revision"],
                "identity": requested_identity,
                "openvla_root": local_cfg.get("backbone", {}).get(
                    "openvla_root", "external/openvla-oft"
                ),
                "freeze_backbone": True,
                "num_images_in_input": 2,
            },
            "teacher": {"cache_path": None, "inference_enabled": False},
            "data": {
                "backend": "rlds",
                "observation_views": ["primary", "wrist"],
                "image_aug": False,
            },
            "training": {
                "device": local_cfg.get("training", {}).get("device", "auto"),
                "precision": local_cfg.get("training", {}).get("precision", "bf16"),
            },
        },
    )
    model = build_flow_policy(cfg, include_teacher=False)
    load_flow_checkpoint(flow_checkpoint, model, resume=False, metadata_out=metadata)
    model.eval()
    policy_cfg = benchmark_cfg.get("policy", {})
    adapter = CalvinTemporalPolicyAdapter(
        model,
        model.backbone.processor.image_processor.apply_transform,
        action_adapter,
        history_length=int(policy_cfg.get("history_length", 8)),
        long_memory_slots=int(policy_cfg.get("long_memory_slots", 4)),
        flow_seed=int(policy_cfg.get("flow_seed", 1701)),
        use_proprio=bool(benchmark_cfg.get("observation", {}).get("use_proprio", False)),
        preserve_memory_across_subtasks=bool(
            policy_cfg.get("preserve_memory_across_subtasks", True)
        ),
    )
    return adapter, metadata, benchmark_cfg


class CustomModel:
    """No-argument class matching CALVIN's official custom-model interface."""

    def __init__(self):
        self.policy, self.checkpoint_metadata, self.benchmark_config = load_custom_model(
            benchmark_config=_required_env("MOWE_CALVIN_CONFIG"),
            flow_checkpoint=_required_env("MOWE_FLOW_CHECKPOINT"),
            backbone_checkpoint=_required_env("MOWE_BACKBONE_CHECKPOINT"),
            backbone_revision=_required_env("MOWE_BACKBONE_REVISION"),
            local_config=os.environ.get(
                "MOWE_LOCAL_CONFIG", "configs/mowe_wam/train_flow_wam_skill_moe.yaml"
            ),
        )

    def reset(self):
        return self.policy.reset()

    def reset_sequence(self):
        return self.policy.reset_sequence()

    def step(self, obs, goal):
        return self.policy.step(obs, goal)
