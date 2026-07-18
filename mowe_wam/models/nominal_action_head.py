"""Nominal 6D motion flow and independent binary gripper head."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.models.action_flow import ActionFlowSampler, ActionFlowTrunk
from mowe_wam.utils.optional import require_torch


class RegressionNominalActionHead(nn.Module if nn is not None else object):
    """Legacy deterministic 7D regression head retained as a baseline."""

    def __init__(
        self,
        context_dim: int,
        memory_dim: int = 512,
        hidden_dim: int = 512,
        action_dim: int = 7,
        chunk_size: int = 8,
    ) -> None:
        require_torch()
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.net = nn.Sequential(
            nn.Linear(context_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, chunk_size * action_dim),
        )

    def forward(self, current_context, memory_context):
        output = self.net(torch.cat([current_context, memory_context], dim=-1))
        return output.view(output.shape[0], self.chunk_size, self.action_dim)


class NominalActionHead(nn.Module if nn is not None else object):
    """Generate normalized relative motion with flow and gripper with BCE logits."""

    def __init__(
        self,
        context_dim: int,
        memory_dim: int = 512,
        hidden_dim: int = 512,
        motion_dim: int = 6,
        chunk_size: int = 8,
        flow_depth: int = 3,
        dropout: float = 0.0,
        shared_trunk=None,
    ) -> None:
        require_torch()
        super().__init__()
        self.context_dim = int(context_dim)
        self.memory_dim = int(memory_dim)
        self.hidden_dim = int(hidden_dim)
        self.motion_dim = int(motion_dim)
        self.chunk_size = int(chunk_size)
        self.condition_encoder = nn.Sequential(
            nn.Linear(context_dim + memory_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.flow_trunk = shared_trunk or ActionFlowTrunk(
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            motion_dim=motion_dim,
            chunk_size=chunk_size,
            depth=flow_depth,
            dropout=dropout,
        )
        if self.flow_trunk.condition_dim != hidden_dim:
            raise ValueError("shared flow trunk condition_dim must equal nominal hidden_dim.")
        self.motion_head = nn.Linear(hidden_dim, motion_dim)
        self.gripper_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, chunk_size),
        )
        self.sampler = ActionFlowSampler(chunk_size=chunk_size, motion_dim=motion_dim)

    def encode_condition(self, context, memory_context):
        return self.condition_encoder(torch.cat([context, memory_context], dim=-1))

    def motion_velocity(self, noisy_motion, flow_time, context, memory_context, *, condition=None):
        condition = self.encode_condition(context, memory_context) if condition is None else condition
        return self.motion_head(self.flow_trunk(noisy_motion, flow_time, condition))

    def gripper_logits(self, context, memory_context, *, condition=None):
        condition = self.encode_condition(context, memory_context) if condition is None else condition
        return self.gripper_head(condition).view(condition.shape[0], self.chunk_size, 1)

    def sample(self, context, memory_context, *, seed=None, num_steps=4):
        condition = self.encode_condition(context, memory_context)

        def velocity(state, flow_time, _condition):
            return self.motion_head(self.flow_trunk(state, flow_time, _condition))

        nominal_motion = self.sampler.sample(
            velocity,
            condition,
            num_steps=num_steps,
            seed=seed,
        ).clamp(-1.0, 1.0)
        gripper_logits = self.gripper_logits(context, memory_context, condition=condition)
        gripper_probability = torch.sigmoid(gripper_logits)
        nominal_actions = torch.cat([nominal_motion, gripper_probability], dim=-1)
        return {
            "nominal_motion": nominal_motion,
            "gripper_logits": gripper_logits,
            "gripper_probability": gripper_probability,
            "nominal_actions": nominal_actions,
            "flow_condition": condition,
        }

    def forward(self, context, memory_context, *, seed=None, num_steps=4):
        return self.sample(context, memory_context, seed=seed, num_steps=num_steps)
