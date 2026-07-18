"""Residual-flow motor experts and the legacy regression baseline."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.models.action_flow import ActionFlowSampler, ActionFlowTrunk
from mowe_wam.utils.optional import require_torch


class ResidualFlowExperts(nn.Module if nn is not None else object):
    """Six light motor heads over one shared 6D flow trunk and solver."""

    def __init__(
        self,
        condition_dim: int = 512,
        hidden_dim: int = 512,
        motion_dim: int = 6,
        chunk_size: int = 8,
        num_motor_experts: int = 6,
        num_routes: int = 7,
        null_route: int = 6,
        flow_depth: int = 3,
        dropout: float = 0.0,
        shared_trunk=None,
    ) -> None:
        require_torch()
        super().__init__()
        if num_motor_experts != 6 or num_routes != 7 or null_route != 6:
            raise ValueError("Expected six motor experts and route 6 as null_finish.")
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.motion_dim = int(motion_dim)
        self.chunk_size = int(chunk_size)
        self.num_motor_experts = int(num_motor_experts)
        self.num_routes = int(num_routes)
        self.null_route = int(null_route)
        self.flow_trunk = shared_trunk or ActionFlowTrunk(
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            motion_dim=motion_dim,
            chunk_size=chunk_size,
            depth=flow_depth,
            dropout=dropout,
        )
        if self.flow_trunk.condition_dim != condition_dim:
            raise ValueError("shared flow trunk condition_dim does not match expert condition_dim.")
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_motor_experts)
            ]
        )
        self.velocity_heads = nn.ModuleList(
            [nn.Linear(hidden_dim, motion_dim) for _ in range(num_motor_experts)]
        )
        self.sampler = ActionFlowSampler(chunk_size=chunk_size, motion_dim=motion_dim)

    def all_velocities(
        self,
        expert_context,
        nominal_motion,
        noisy_residuals,
        flow_time,
    ):
        shared = self.flow_trunk(
            noisy_residuals,
            flow_time,
            expert_context,
            token_condition=nominal_motion,
        )
        outputs = [head(shared + adapter(shared)) for adapter, head in zip(self.adapters, self.velocity_heads)]
        return torch.stack(outputs, dim=2)  # [B, T, six motor experts, 6]

    def velocity(
        self,
        expert_context,
        nominal_motion,
        noisy_residuals,
        flow_time,
        route_gates,
    ):
        if route_gates.shape != (
            noisy_residuals.shape[0],
            self.chunk_size,
            self.num_routes,
        ):
            raise ValueError(
                f"route_gates must have shape [B, {self.chunk_size}, {self.num_routes}]."
            )
        all_velocity = self.all_velocities(
            expert_context,
            nominal_motion,
            noisy_residuals,
            flow_time,
        )
        motor_gates = route_gates[..., : self.num_motor_experts]
        return (all_velocity * motor_gates.unsqueeze(-1)).sum(dim=2)

    def sample(
        self,
        expert_context,
        nominal_motion,
        route_gates,
        *,
        seed=None,
        num_steps=4,
    ):
        motor_gate = route_gates[..., : self.num_motor_experts].sum(dim=-1, keepdim=True)
        initial_noise = self.sampler.noise(expert_context, seed=seed)

        def velocity_fn(state, flow_time, condition):
            return self.velocity(condition, nominal_motion, state, flow_time, route_gates)

        residual = self.sampler.sample(
            velocity_fn,
            expert_context,
            num_steps=num_steps,
            initial_noise=initial_noise,
            state_gate=motor_gate,
        )
        # The multiplication above is structural, not a learned null residual.
        return motor_gate * residual


class RegressionResidualActionExperts(nn.Module if nn is not None else object):
    """Legacy chunk-level residual regressors retained as a paper baseline."""

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 7,
        chunk_size: int = 8,
        num_experts: int = 5,
    ) -> None:
        require_torch()
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, chunk_size * action_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, features, router_probs, topk_experts):
        if topk_experts.dim() != 2 or topk_experts.shape[0] != features.shape[0]:
            raise ValueError("topk_experts must have shape [B, selected_experts].")
        batch_size, selected_count = topk_experts.shape
        selected = features.new_zeros(batch_size, selected_count, self.chunk_size, self.action_dim)
        for expert_index, expert in enumerate(self.experts):
            rows, slots = (topk_experts == expert_index).nonzero(as_tuple=True)
            if rows.numel() == 0:
                continue
            values = expert(features.index_select(0, rows)).view(
                rows.numel(), self.chunk_size, self.action_dim
            )
            selected[rows, slots] = values
        weights = router_probs.gather(1, topk_experts)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        residual = (selected * weights[:, :, None, None]).sum(dim=1)
        dense_weights = torch.zeros_like(router_probs).scatter(1, topk_experts, weights)
        return {
            "action_residual": residual,
            "selected_residuals": selected,
            "expert_indices": topk_experts,
            "expert_weights": dense_weights,
            "topk_weights": weights,
        }


# Backwards-compatible import for old scripts; new code uses ResidualFlowExperts.
ResidualActionExperts = RegressionResidualActionExperts
