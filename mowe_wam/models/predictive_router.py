"""Future-predictive expert router with memory-aware FiLM conditioning."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


class PredictiveExpertRouter(nn.Module if nn is not None else object):
    """Route experts using current state, predicted transition, and event memory.

    Future transition and memory produce FiLM parameters for the state branch,
    preventing the high-dimensional VLA feature from trivially drowning out the
    low-dimensional predictive evidence.
    """

    def __init__(
        self,
        hidden_dim: int,
        predicate_dim: int,
        memory_context_dim: int,
        num_experts: int,
        top_k: int = 2,
        state_dim: int = 256,
        transition_dim: int = 128,
        previous_expert_dim: int = 32,
        switch_penalty: float = 0.05,
    ) -> None:
        require_torch()
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts.")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.switch_penalty = float(switch_penalty)
        self.state_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, state_dim), nn.GELU())
        transition_input_dim = int(predicate_dim) + 3
        self.transition_proj = nn.Sequential(nn.Linear(transition_input_dim, transition_dim), nn.GELU())
        self.memory_proj = nn.Sequential(nn.Linear(memory_context_dim, transition_dim), nn.GELU())
        self.film = nn.Linear(transition_dim * 2, state_dim * 2)
        self.previous_expert_embedding = nn.Embedding(self.num_experts + 1, previous_expert_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim + transition_dim * 2 + previous_expert_dim),
            nn.Linear(state_dim + transition_dim * 2 + previous_expert_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, self.num_experts),
        )

    def forward(
        self,
        current_features,
        future_predicates,
        progress_delta,
        future_risk,
        future_recovery,
        memory_context,
        previous_expert=None,
    ):
        batch_size = current_features.shape[0]
        transition_inputs = torch.cat([future_predicates, progress_delta, future_risk, future_recovery], dim=-1)
        transition = self.transition_proj(transition_inputs)
        memory = self.memory_proj(memory_context)
        gamma, beta = self.film(torch.cat([transition, memory], dim=-1)).chunk(2, dim=-1)
        state = self.state_proj(current_features)
        state = state * (1.0 + torch.tanh(gamma)) + beta

        if previous_expert is None:
            previous = torch.full((batch_size,), self.num_experts, dtype=torch.long, device=current_features.device)
        else:
            previous = previous_expert.to(device=current_features.device, dtype=torch.long).view(-1)
            previous = torch.where((previous >= 0) & (previous < self.num_experts), previous, torch.full_like(previous, self.num_experts))
        previous_embedding = self.previous_expert_embedding(previous)
        logits = self.net(torch.cat([state, transition, memory, previous_embedding], dim=-1))
        if self.switch_penalty > 0.0:
            valid_previous = previous < self.num_experts
            non_previous = torch.ones_like(logits)
            non_previous.scatter_(1, previous.clamp_max(self.num_experts - 1).unsqueeze(1), 0.0)
            logits = logits - self.switch_penalty * non_previous * valid_previous.unsqueeze(1)
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=self.top_k, dim=-1).indices
        return {
            "router_logits": logits,
            "router_probs": probs,
            "topk_experts": topk,
            "router_switch_penalty": logits.new_tensor(self.switch_penalty),
        }
