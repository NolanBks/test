"""Shared rectified-flow primitives for normalized LIBERO motion actions.

Only the first six relative-motion dimensions belong in this module.  The
absolute gripper bit is intentionally handled by a separate classifier.
"""

from __future__ import annotations

import math
from collections.abc import Callable

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


def _batch_reference(condition):
    if torch is not None and torch.is_tensor(condition):
        return condition
    if isinstance(condition, dict):
        for value in condition.values():
            if torch is not None and torch.is_tensor(value):
                return value
    raise TypeError("condition must be a Tensor or contain at least one Tensor.")


class SinusoidalFlowTimeEmbedding(nn.Module if nn is not None else object):
    def __init__(self, hidden_dim: int) -> None:
        require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)

    def forward(self, flow_time):
        values = flow_time.float().reshape(flow_time.shape[0], -1)[:, :1]
        half = self.hidden_dim // 2
        if half == 0:
            return values
        frequencies = torch.exp(
            torch.arange(half, device=values.device, dtype=values.dtype)
            * (-math.log(10_000.0) / max(half - 1, 1))
        )
        angles = values * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if embedding.shape[-1] < self.hidden_dim:
            embedding = torch.nn.functional.pad(embedding, (0, self.hidden_dim - embedding.shape[-1]))
        return embedding


class ActionFlowTrunk(nn.Module if nn is not None else object):
    """A shared per-token flow trunk used by nominal and residual heads."""

    def __init__(
        self,
        condition_dim: int,
        hidden_dim: int = 512,
        motion_dim: int = 6,
        chunk_size: int = 8,
        depth: int = 3,
        dropout: float = 0.0,
    ) -> None:
        require_torch()
        super().__init__()
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.motion_dim = int(motion_dim)
        self.chunk_size = int(chunk_size)
        self.state_projection = nn.Linear(motion_dim, hidden_dim)
        self.token_condition_projection = nn.Linear(motion_dim, hidden_dim)
        self.condition_projection = nn.Linear(condition_dim, hidden_dim)
        self.time_embedding = SinusoidalFlowTimeEmbedding(hidden_dim)
        self.time_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.position_embedding = nn.Embedding(chunk_size, hidden_dim)
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(int(depth))
            ]
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, noisy_motion, flow_time, condition, token_condition=None):
        expected = (self.chunk_size, self.motion_dim)
        if noisy_motion.shape[1:] != expected:
            raise ValueError(f"noisy_motion must have shape [B, {expected[0]}, {expected[1]}].")
        if condition.shape != (noisy_motion.shape[0], self.condition_dim):
            raise ValueError(
                f"condition must have shape [B, {self.condition_dim}], got {tuple(condition.shape)}."
            )
        time = self.time_projection(self.time_embedding(flow_time)).unsqueeze(1)
        position = self.position_embedding.weight[: self.chunk_size].unsqueeze(0)
        hidden = self.state_projection(noisy_motion) + self.condition_projection(condition).unsqueeze(1)
        hidden = hidden + time + position
        if token_condition is not None:
            if token_condition.shape != noisy_motion.shape:
                raise ValueError("token_condition must have the same shape as noisy_motion.")
            hidden = hidden + self.token_condition_projection(token_condition)
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output_norm(hidden)


class ActionFlowSampler:
    """Fixed-step Euler sampler for conditional rectified flow."""

    implementation_id = "rectified_flow_euler_v1"

    def __init__(self, chunk_size: int = 8, motion_dim: int = 6) -> None:
        require_torch()
        self.chunk_size = int(chunk_size)
        self.motion_dim = int(motion_dim)

    def noise(self, condition, *, seed: int | None = None):
        reference = _batch_reference(condition)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=reference.device)
            generator.manual_seed(int(seed))
        return torch.randn(
            (reference.shape[0], self.chunk_size, self.motion_dim),
            device=reference.device,
            dtype=reference.dtype,
            generator=generator,
        )

    def sample(
        self,
        velocity_fn: Callable,
        condition,
        *,
        num_steps: int = 4,
        seed: int | None = None,
        initial_noise=None,
        state_gate=None,
    ):
        if int(num_steps) < 1:
            raise ValueError("num_steps must be at least 1.")
        state = self.noise(condition, seed=seed) if initial_noise is None else initial_noise
        if state.shape[1:] != (self.chunk_size, self.motion_dim):
            raise ValueError("initial_noise has the wrong action shape.")
        if state_gate is not None:
            state = state_gate * state
        dt = 1.0 / float(num_steps)
        for index in range(int(num_steps)):
            flow_time = state.new_full((state.shape[0], 1), index / float(num_steps))
            velocity = velocity_fn(state, flow_time, condition)
            if velocity.shape != state.shape:
                raise ValueError("velocity_fn must return the same shape as the flow state.")
            state = state + dt * velocity
            if state_gate is not None:
                state = state_gate * state
        return state


def rectified_flow_path(target_motion, *, noise=None, flow_time=None):
    """Sample ``x_s=(1-s)eps+s*x`` and its constant velocity target."""

    torch_mod = require_torch()
    if target_motion.ndim != 3 or target_motion.shape[-1] != 6:
        raise ValueError("target_motion must have shape [B, T, 6].")
    if noise is None:
        noise = torch_mod.randn_like(target_motion)
    if flow_time is None:
        flow_time = torch_mod.rand(
            (target_motion.shape[0], 1, 1), device=target_motion.device, dtype=target_motion.dtype
        )
    elif flow_time.ndim == 2:
        flow_time = flow_time.unsqueeze(-1)
    noisy = (1.0 - flow_time) * noise + flow_time * target_motion
    return {
        "noisy_motion": noisy,
        "target_velocity": target_motion - noise,
        "flow_time": flow_time.squeeze(-1),
        "noise": noise,
    }
