#!/usr/bin/env python3
"""Synthetic forward/backward contract check for the flow-WAM main path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.memory import MultiScaleMemoryEncoder
from mowe_wam.models import (
    FlowWAMSkillPolicy,
    FutureGroundedRouter,
    LatentWorldModel,
    LanguageConditionedViewFusion,
    NominalActionHead,
    ResidualFlowExperts,
    execution_steps,
    risk_gated_execution,
)
from mowe_wam.training import flow_wam_skill_losses
from mowe_wam.utils.optional import require_torch


class SyntheticContextBackbone:
    def __init__(self, hidden_dim):
        self.hidden_dim = int(hidden_dim)

    def extract_context_features(self, batch):
        return {
            "current_visual_views": batch["synthetic_current_visual_views"],
            "history_visual_views": batch["synthetic_history_visual_views"],
            "long_history_visual_views": batch["synthetic_long_visual_views"],
            "language": batch["synthetic_language"],
        }

    def keep_frozen_backbone_eval(self):
        return None


def build_model(torch, batch_size=2):
    del batch_size
    context_dim = hidden_dim = 64
    target_dim, route_world_dim = 32, 32
    memory = MultiScaleMemoryEncoder(
        context_dim, context_dim, action_dim=7, hidden_dim=hidden_dim, heads=4
    )
    nominal = NominalActionHead(
        context_dim,
        memory_dim=hidden_dim,
        hidden_dim=hidden_dim,
        motion_dim=6,
        chunk_size=16,
        flow_depth=2,
    )
    world = LatentWorldModel(
        context_dim,
        memory_dim=hidden_dim,
        hidden_dim=hidden_dim,
        route_world_dim=route_world_dim,
        layers=2,
        heads=4,
        target_tokens=16,
        target_dim=target_dim,
        action_chunk_size=16,
        future_horizons=(1, 4, 8, 16),
    )
    router = FutureGroundedRouter(
        world_dim=hidden_dim,
        memory_dim=hidden_dim,
        latent_dim=target_dim,
        route_world_dim=route_world_dim,
        hidden_dim=hidden_dim,
        chunk_size=16,
    )
    experts = ResidualFlowExperts(
        condition_dim=hidden_dim,
        hidden_dim=hidden_dim,
        chunk_size=16,
        flow_depth=2,
    )
    view_fusion = LanguageConditionedViewFusion(
        context_dim, context_dim, hidden_dim=32, num_views=2
    )
    return FlowWAMSkillPolicy(
        SyntheticContextBackbone(context_dim),
        memory,
        nominal,
        world,
        router,
        experts,
        view_fusion,
        context_dim=context_dim,
        memory_dim=hidden_dim,
        world_dim=hidden_dim,
        expert_condition_dim=hidden_dim,
        flow_steps=3,
        execution_config={"default_steps": 8, "caution_steps": 4},
    )


def make_batch(torch, batch_size):
    target_motion = torch.rand(batch_size, 16, 6) * 2.0 - 1.0
    target_gripper = torch.randint(0, 2, (batch_size, 16, 1)).float()
    patterns = torch.tensor(
        [
            [0, 0, 0, 2, 2, 1, 1, 1, 1, 1, 5, 5, 5, 6, 6, 6],
            [3, 4, 5, 6, 0, 1, 2, 2, 2, 2, 2, 2, 6, 6, 6, 6],
        ],
        dtype=torch.long,
    )
    labels = patterns[torch.arange(batch_size) % patterns.shape[0]]
    return {
        "synthetic_current_visual_views": torch.randn(batch_size, 2, 64),
        "synthetic_history_visual_views": torch.randn(batch_size, 7, 2, 64),
        "synthetic_long_visual_views": torch.randn(batch_size, 4, 2, 64),
        "synthetic_language": torch.randn(batch_size, 64),
        "history_actions": torch.randn(batch_size, 7, 7),
        "history_mask": torch.ones(batch_size, 8, dtype=torch.bool),
        "long_history_actions": torch.randn(batch_size, 4, 7),
        "long_history_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "target_actions": torch.cat([target_motion, target_gripper], dim=-1),
        "target_motion": target_motion,
        "target_gripper": target_gripper,
        "future_mask": torch.ones(batch_size, 4, dtype=torch.bool),
        "future_horizons": torch.tensor([[1, 4, 8, 16]] * batch_size, dtype=torch.long),
        "current_latent_target": torch.randn(batch_size, 16, 32),
        "future_latent_targets": torch.randn(batch_size, 4, 16, 32),
        "expert_skill_labels": labels,
        "expert_skill_mask": torch.ones(batch_size, 16, dtype=torch.bool),
        "expert_label_source": [["synthetic"] * 16 for _ in range(batch_size)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    torch = require_torch()
    torch.manual_seed(7)
    model = build_model(torch, args.batch_size)
    batch = make_batch(torch, args.batch_size)

    output = model(
        batch,
        action_condition_mode="scheduled",
        teacher_forcing_probability=0.5,
        route_mode="oracle",
        flow_seed=7,
    )
    losses = flow_wam_skill_losses(
        output,
        batch,
        {
            "flow_nominal": 1.0,
            "flow_expert": 1.0,
            "gripper_bce": 1.0,
            "route": 1.0,
            "world": 1.0,
            "delta": 0.5,
            "load_balance": 0.01,
            "residual": 0.001,
            "endpoint": 0.05,
        },
        stage="joint",
    )
    losses["total_loss"].backward()
    expert_gradient_norms = [
        float(
            sum(
                parameter.grad.detach().float().square().sum()
                for parameter in head.parameters()
                if parameter.grad is not None
            ).sqrt()
        )
        for head in model.residual_experts.velocity_heads
    ]
    model.zero_grad(set_to_none=True)
    st_output = model(
        batch,
        action_condition_mode="nominal",
        route_mode="st_gumbel",
        gumbel_temperature=0.7,
        flow_seed=8,
        compute_teacher_targets=False,
    )
    st_output["motion_actions"].float().square().mean().backward()
    st_router_action_gradient = float(
        sum(
            parameter.grad.detach().float().square().sum()
            for parameter in model.router.parameters()
            if parameter.grad is not None
        ).sqrt()
    )
    assert st_router_action_gradient > 0.0
    model.zero_grad(set_to_none=True)
    predicted_output = model(
        batch,
        action_condition_mode="nominal",
        route_mode="predicted",
        flow_seed=9,
        compute_teacher_targets=False,
    )
    soft_output = model(
        batch,
        action_condition_mode="nominal",
        route_mode="soft",
        flow_seed=11,
        compute_teacher_targets=False,
    )
    predicted_repeat = model(
        batch,
        action_condition_mode="nominal",
        route_mode="predicted",
        flow_seed=9,
        compute_teacher_targets=False,
    )
    deployment_batch = {
        key: value
        for key, value in batch.items()
        if key
        not in {
            "target_actions",
            "target_motion",
            "target_gripper",
            "current_latent_target",
            "future_latent_targets",
            "expert_skill_labels",
            "expert_skill_mask",
            "expert_label_source",
        }
    }
    deployment_prefixes, _ = model.predict_actions(deployment_batch, flow_seed=10)
    null_mask = output["route_indices"].eq(6).unsqueeze(-1)
    assert not bool((output["residual_motion"].ne(0) & null_mask).any())
    assert output["nominal_motion"].shape == (args.batch_size, 16, 6)
    assert output["gripper_logits"].shape == (args.batch_size, 16, 1)
    assert output["route_world_tokens"].shape == (args.batch_size, 16, 32)
    assert output["router_logits"].shape == (args.batch_size, 16, 7)
    assert output["actions"].shape == (args.batch_size, 16, 7)
    assert output["current_view_weights"].shape == (args.batch_size, 2)
    assert torch.allclose(output["current_view_weights"].sum(dim=-1), torch.ones(args.batch_size))
    assert bool(((output["gripper_actions"] == 0) | (output["gripper_actions"] == 1)).all())
    assert torch.equal(predicted_output["nominal_motion"], predicted_repeat["nominal_motion"])
    assert torch.equal(predicted_output["actions"], predicted_repeat["actions"])
    assert all(1 <= len(prefix) <= 8 for prefix in deployment_prefixes)
    expected_steps = execution_steps(torch.tensor([[0, 0, 0, 2, 2, 1, 1, 6]]), 8)
    assert expected_steps.tolist() == [3]
    confident_routes = torch.tensor([[0, 0, 0, 2, 2, 1, 1, 1]])
    confident_probabilities = torch.nn.functional.one_hot(confident_routes, 7).float()
    smooth_motion = torch.zeros(1, 8, 6)
    confident_execution = risk_gated_execution(
        confident_routes,
        confident_probabilities,
        smooth_motion,
        torch.zeros_like(smooth_motion),
    )
    assert confident_execution["execution_steps"].tolist() == [8]
    assert confident_execution["execution_crosses_predicted_boundary"].tolist() == [True]
    assert not any("transition_difference" in name or "delta_h" in name for name, _ in model.router.named_modules())

    print(
        json.dumps(
            {
                "nominal_motion": list(output["nominal_motion"].shape),
                "gripper_logits": list(output["gripper_logits"].shape),
                "route_world_tokens": list(output["route_world_tokens"].shape),
                "future_latents": list(output["future_latents"].shape),
                "router_logits": list(output["router_logits"].shape),
                "oracle_routes": list(output["route_indices"].shape),
                "st_routes": list(st_output["route_indices"].shape),
                "predicted_routes": list(predicted_output["route_indices"].shape),
                "soft_route_gates": list(soft_output["route_gates"].shape),
                "null_zero_violations": int(output["null_motion_zero_violation_count"]),
                "expert_gradient_norms": expert_gradient_norms,
                "st_router_action_gradient": st_router_action_gradient,
                "prefix_example": int(expected_steps.item()),
                "total_loss": float(losses["total_loss"].detach()),
                "world_parameters": model.world_model.parameter_count(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
