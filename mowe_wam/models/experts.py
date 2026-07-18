"""Action experts for sparse MoE decoding."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


class MoEActionExperts(nn.Module if nn is not None else object):
    def __init__(self, hidden_dim: int, action_dim: int, chunk_size: int, num_experts: int) -> None:
        require_torch()
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, chunk_size * action_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, features, router_probs, topk_experts=None):
        if topk_experts is None:
            topk_experts = torch.arange(self.num_experts, device=features.device).expand(features.shape[0], -1)
        if topk_experts.dim() != 2 or topk_experts.shape[0] != features.shape[0]:
            raise ValueError("topk_experts must have shape [batch_size, num_selected_experts].")
        if topk_experts.shape[1] < 1 or topk_experts.shape[1] > self.num_experts:
            raise ValueError("topk_experts must select between one and num_experts experts.")
        if (topk_experts < 0).any() or (topk_experts >= self.num_experts).any():
            raise ValueError("topk_experts contains an out-of-range expert index.")

        # Execute only the selected expert/sample pairs.  This is genuinely
        # sparse at Top-k < num_experts while retaining a dense fallback when
        # a caller intentionally omits ``topk_experts``.
        batch_size, num_selected = topk_experts.shape
        selected_actions = features.new_zeros(batch_size, num_selected, self.chunk_size, self.action_dim)
        for expert_idx, expert in enumerate(self.experts):
            selected_rows, selected_slots = (topk_experts == expert_idx).nonzero(as_tuple=True)
            if selected_rows.numel() == 0:
                continue
            expert_output = expert(features.index_select(0, selected_rows)).view(
                selected_rows.numel(), self.chunk_size, self.action_dim
            )
            selected_actions[selected_rows, selected_slots] = expert_output

        topk_weights = torch.gather(router_probs, dim=1, index=topk_experts)
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        expert_weights = torch.zeros_like(router_probs)
        expert_weights.scatter_(dim=1, index=topk_experts, src=topk_weights)
        actions = (selected_actions * topk_weights[:, :, None, None]).sum(dim=1)
        return {
            "actions": actions,
            "expert_actions": selected_actions,
            "expert_indices": topk_experts,
            "expert_weights": expert_weights,
            "topk_weights": topk_weights,
        }
