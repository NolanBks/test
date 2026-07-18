"""Predicate-conditioned sparse expert router."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


class ExpertRouter(nn.Module if nn is not None else object):
    def __init__(self, hidden_dim: int, predicate_dim: int, num_experts: int, top_k: int = 2) -> None:
        require_torch()
        super().__init__()
        if top_k > num_experts:
            raise ValueError("top_k cannot exceed num_experts.")
        self.num_experts = num_experts
        self.top_k = top_k
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + predicate_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, features, predicates):
        router_input = torch.cat([features, predicates], dim=-1)
        logits = self.net(router_input)
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=self.top_k, dim=-1).indices
        return {"router_logits": logits, "router_probs": probs, "topk_experts": topk}
