"""Future-grounded temporal skill routers."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


class FutureGroundedRouter(nn.Module if nn is not None else object):
    """Predict one of six motor skills or the null bypass at each action step."""

    def __init__(
        self,
        world_dim: int = 512,
        memory_dim: int = 512,
        latent_dim: int = 384,
        route_world_dim: int = 128,
        action_dim: int = 7,
        hidden_dim: int = 256,
        num_routes: int = 7,
        chunk_size: int = 8,
        null_route: int = 6,
        use_uncertainty: bool = False,
    ) -> None:
        require_torch()
        super().__init__()
        if num_routes != 7:
            raise ValueError("The first implementation requires six motor routes plus null_finish.")
        if not 0 <= null_route < num_routes:
            raise ValueError("null_route must index one of num_routes.")
        self.num_routes = int(num_routes)
        self.chunk_size = int(chunk_size)
        self.null_route = int(null_route)
        self.use_uncertainty = bool(use_uncertainty)

        # The three terms below are the intentionally simple q_j definition.
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.world_proj = nn.Linear(route_world_dim, hidden_dim)
        self.position_embedding = nn.Embedding(chunk_size, hidden_dim)

        self.world_belief_projection = nn.Linear(world_dim, hidden_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.future_projection = nn.Linear(latent_dim, hidden_dim)
        self.delta_projection = nn.Linear(latent_dim, hidden_dim)
        self.uncertainty_projection = nn.Linear(1, hidden_dim) if use_uncertainty else None
        self.route_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_routes),
        )

    def _oracle_gates(self, labels, mask, dtype):
        if labels is None:
            raise ValueError("route_mode='oracle' requires oracle_labels.")
        if labels.shape != (labels.shape[0], self.chunk_size):
            raise ValueError(f"oracle_labels must have shape [B, {self.chunk_size}].")
        valid = labels.ge(0) & labels.lt(self.num_routes)
        if mask is not None:
            valid = valid & mask.bool()
        safe = torch.where(valid, labels, labels.new_full(labels.shape, self.null_route))
        return torch.nn.functional.one_hot(safe.long(), self.num_routes).to(dtype=dtype), valid

    def forward(
        self,
        world_belief,
        future_latents,
        delta_latents,
        route_world_tokens,
        memory_context,
        nominal_action_tokens,
        uncertainty=None,
        route_mode: str = "predicted",
        *,
        oracle_labels=None,
        oracle_mask=None,
        gumbel_temperature: float = 1.0,
    ):
        expected_tokens = (nominal_action_tokens.shape[0], self.chunk_size)
        if nominal_action_tokens.shape[:2] != expected_tokens:
            raise ValueError(f"nominal_action_tokens must have {self.chunk_size} positions.")
        if route_world_tokens.shape[:2] != expected_tokens:
            raise ValueError(f"route_world_tokens must have shape [B, {self.chunk_size}, D].")

        # q_j = ActionMLP(A0[j]) + WorldProjection(h_(j+1)) + PositionEmbedding(j)
        action_query = self.action_mlp(nominal_action_tokens)
        world_token_query = self.world_proj(route_world_tokens)
        position_query = self.position_embedding.weight[: self.chunk_size].unsqueeze(0)
        queries = action_query + world_token_query + position_query
        future_summary = future_latents.mean(dim=(1, 2))
        delta_summary = delta_latents.mean(dim=(1, 2))
        world_belief_branch = self.world_belief_projection(world_belief)
        memory_branch = self.memory_projection(memory_context)
        future_branch = self.future_projection(future_summary)
        delta_branch = self.delta_projection(delta_summary)
        global_context = world_belief_branch + memory_branch + future_branch + delta_branch
        uncertainty_branch = None
        if self.use_uncertainty:
            if uncertainty is None:
                raise ValueError("Router configured with uncertainty requires uncertainty input.")
            uncertainty_branch = self.uncertainty_projection(
                uncertainty.mean(dim=1, keepdim=True)
            )
            global_context = global_context + uncertainty_branch
        logits = self.route_head(queries + global_context.unsqueeze(1))
        probabilities = torch.softmax(logits, dim=-1)

        if route_mode == "oracle":
            gates, valid_oracle_mask = self._oracle_gates(
                oracle_labels, oracle_mask, probabilities.dtype
            )
        elif route_mode == "st_gumbel":
            gates = torch.nn.functional.gumbel_softmax(
                logits,
                tau=max(float(gumbel_temperature), 1e-4),
                hard=True,
                dim=-1,
            )
            valid_oracle_mask = None
        elif route_mode == "predicted":
            indices = logits.argmax(dim=-1)
            gates = torch.nn.functional.one_hot(indices, self.num_routes).to(probabilities.dtype)
            valid_oracle_mask = None
        elif route_mode == "soft":
            gates = probabilities
            valid_oracle_mask = None
        else:
            raise ValueError("route_mode must be oracle, st_gumbel, predicted, or soft.")

        route_indices = gates.argmax(dim=-1)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        output = {
            "router_logits": logits,
            "router_probs": probabilities,
            "route_gates": gates,
            "route_indices": route_indices,
            "current_skill": route_indices[:, 0],
            "router_entropy": entropy,
            "route_source": route_mode,
            "router_queries": queries,
            # Activation norms make branch bypass/collapse visible without
            # requiring additional counterfactual forward passes every step.
            "router_branch_norms": {
                "action_query": action_query.float().norm(dim=-1).mean(),
                "world_token_query": world_token_query.float().norm(dim=-1).mean(),
                "position_query": position_query.float().norm(dim=-1).mean(),
                "world_belief": world_belief_branch.float().norm(dim=-1).mean(),
                "memory": memory_branch.float().norm(dim=-1).mean(),
                "future": future_branch.float().norm(dim=-1).mean(),
                "delta": delta_branch.float().norm(dim=-1).mean(),
            },
        }
        if uncertainty_branch is not None:
            output["router_branch_norms"]["uncertainty"] = (
                uncertainty_branch.float().norm(dim=-1).mean()
            )
        if valid_oracle_mask is not None:
            output["valid_oracle_mask"] = valid_oracle_mask
        return output


class LegacyFutureGroundedRouter(nn.Module if nn is not None else object):
    """Original chunk-level Top-k router retained for regression baselines."""

    def __init__(
        self,
        world_dim: int = 512,
        memory_dim: int = 512,
        latent_dim: int = 384,
        hidden_dim: int = 512,
        num_experts: int = 5,
        top_k: int = 2,
        use_uncertainty: bool = False,
    ) -> None:
        require_torch()
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must be between 1 and num_experts.")
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.use_uncertainty = bool(use_uncertainty)
        self.future_projection = nn.Linear(latent_dim, hidden_dim // 2)
        self.delta_projection = nn.Linear(latent_dim, hidden_dim // 2)
        input_dim = world_dim + memory_dim + hidden_dim + (1 if use_uncertainty else 0)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(
        self,
        world_belief,
        predicted_future_latents,
        predicted_delta_latents,
        memory_context,
        predicted_uncertainty=None,
        hard_topk: bool = True,
        temperature: float = 1.0,
    ):
        future_summary = predicted_future_latents.mean(dim=(1, 2))
        delta_summary = predicted_delta_latents.mean(dim=(1, 2))
        parts = [
            world_belief,
            memory_context,
            self.future_projection(future_summary),
            self.delta_projection(delta_summary),
        ]
        if self.use_uncertainty:
            if predicted_uncertainty is None:
                raise ValueError("Router configured for uncertainty but WAM omitted it.")
            parts.append(predicted_uncertainty.mean(dim=1, keepdim=True))
        logits = self.net(torch.cat(parts, dim=-1))
        probabilities = torch.softmax(logits / max(float(temperature), 1e-4), dim=-1)
        if hard_topk:
            topk_experts = probabilities.topk(self.top_k, dim=-1).indices
        else:
            topk_experts = torch.arange(self.num_experts, device=logits.device).expand(logits.shape[0], -1)
        topk_weights = probabilities.gather(1, topk_experts)
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        return {
            "router_logits": logits,
            "router_probs": probabilities,
            "topk_experts": topk_experts,
            "topk_weights": topk_weights,
            "router_entropy": entropy,
            "hard_topk": hard_topk,
        }
