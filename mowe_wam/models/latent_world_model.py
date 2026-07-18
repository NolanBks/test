"""Nominal-action-conditioned causal latent world models."""

from __future__ import annotations

from collections.abc import Sequence

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


class LatentWorldModel(nn.Module if nn is not None else object):
    """Predict stepwise world tokens and spatial targets from an action prefix."""

    def __init__(
        self,
        context_dim: int,
        memory_dim: int = 512,
        action_dim: int = 7,
        action_chunk_size: int = 8,
        future_horizons: Sequence[int] = (1, 4, 8),
        hidden_dim: int = 512,
        route_world_dim: int = 128,
        layers: int = 6,
        heads: int = 8,
        mlp_ratio: int = 4,
        target_tokens: int = 16,
        target_dim: int = 384,
        dropout: float = 0.0,
        predict_uncertainty: bool = False,
        max_sequence_tokens: int = 32,
    ) -> None:
        require_torch()
        super().__init__()
        self.action_chunk_size = int(action_chunk_size)
        self.future_horizons = tuple(int(item) for item in future_horizons)
        self.hidden_dim = int(hidden_dim)
        self.route_world_dim = int(route_world_dim)
        self.target_tokens = int(target_tokens)
        self.target_dim = int(target_dim)
        self.predict_uncertainty = bool(predict_uncertainty)
        if not self.future_horizons or any(
            item < 1 or item > self.action_chunk_size for item in self.future_horizons
        ):
            raise ValueError("future_horizons must be within the action chunk.")

        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_sequence_tokens, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * int(mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.route_world_head = nn.Linear(hidden_dim, route_world_dim)
        self.future_head = nn.Linear(hidden_dim, target_tokens * target_dim)
        self.delta_head = nn.Linear(hidden_dim, target_tokens * target_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1) if predict_uncertainty else None

    def _attention_mask(self, context_tokens: int, device):
        total = context_tokens + self.action_chunk_size
        mask = torch.ones((total, total), dtype=torch.bool, device=device)
        mask[:context_tokens, :context_tokens] = False
        for index in range(self.action_chunk_size):
            row = context_tokens + index
            mask[row, : context_tokens + index + 1] = False
        return mask

    def forward(
        self,
        current_context,
        short_memory=None,
        long_memory=None,
        action_condition=None,
        horizon_mask=None,
        *,
        short_memory_mask=None,
        long_memory_mask=None,
        short_memory_tokens=None,
        long_memory_tokens=None,
    ):
        del horizon_mask  # Targets use the mask; causal prediction is always produced for all horizons.
        short_memory = short_memory if short_memory is not None else short_memory_tokens
        long_memory = long_memory if long_memory is not None else long_memory_tokens
        if short_memory is None or long_memory is None or action_condition is None:
            raise ValueError("short_memory, long_memory, and action_condition are required.")
        expected = (self.action_chunk_size, self.action_projection.in_features)
        if action_condition.shape[1:] != expected:
            raise ValueError(f"action_condition must have shape [B, {expected[0]}, {expected[1]}].")

        batch_size = current_context.shape[0]
        current = self.context_projection(current_context).unsqueeze(1)
        short = self.memory_projection(short_memory)
        long = self.memory_projection(long_memory)
        context = torch.cat([current, short, long], dim=1)
        actions = self.action_projection(action_condition)
        tokens = torch.cat([context, actions], dim=1)
        if tokens.shape[1] > self.position_embeddings.shape[1]:
            raise ValueError("WAM sequence is longer than max_sequence_tokens.")
        tokens = tokens + self.position_embeddings[:, : tokens.shape[1]]

        if short_memory_mask is None:
            short_memory_mask = torch.ones(short.shape[:2], dtype=torch.bool, device=short.device)
        if long_memory_mask is None:
            long_memory_mask = torch.ones(long.shape[:2], dtype=torch.bool, device=long.device)
        valid = torch.cat(
            [
                torch.ones((batch_size, 1), dtype=torch.bool, device=tokens.device),
                short_memory_mask.bool(),
                long_memory_mask.bool(),
                torch.ones((batch_size, self.action_chunk_size), dtype=torch.bool, device=tokens.device),
            ],
            dim=1,
        )
        encoded = self.transformer(
            tokens,
            mask=self._attention_mask(context.shape[1], tokens.device),
            src_key_padding_mask=~valid,
        )
        action_hidden = self.output_norm(encoded[:, context.shape[1] :])
        horizon_indices = torch.as_tensor(
            [item - 1 for item in self.future_horizons], device=action_hidden.device, dtype=torch.long
        )
        horizon_hidden = action_hidden.index_select(1, horizon_indices)
        target_shape = (batch_size, len(self.future_horizons), self.target_tokens, self.target_dim)
        future = self.future_head(horizon_hidden).view(target_shape)
        delta = self.delta_head(horizon_hidden).view(target_shape)
        output = {
            "world_belief": action_hidden.mean(dim=1),
            "route_world_tokens": self.route_world_head(action_hidden),
            "future_latents": future,
            "delta_latents": delta,
            "predicted_future_latents": future,
            "predicted_delta_latents": delta,
            "horizon_hidden": horizon_hidden,
        }
        if self.uncertainty_head is not None:
            log_variance = self.uncertainty_head(horizon_hidden).squeeze(-1)
            output["log_variance"] = log_variance
            output["predicted_uncertainty"] = log_variance
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class LegacyLatentWorldModel(nn.Module if nn is not None else object):
    """Original horizon-query model retained for regression baseline configs."""

    def __init__(
        self,
        context_dim: int,
        action_dim: int = 7,
        action_chunk_size: int = 8,
        future_horizons: Sequence[int] = (1, 4, 8),
        hidden_dim: int = 512,
        layers: int = 6,
        heads: int = 8,
        mlp_ratio: int = 4,
        target_tokens: int = 16,
        target_dim: int = 384,
        dropout: float = 0.0,
        predict_uncertainty: bool = False,
        max_sequence_tokens: int = 32,
    ) -> None:
        require_torch()
        super().__init__()
        self.action_chunk_size = int(action_chunk_size)
        self.future_horizons = tuple(int(item) for item in future_horizons)
        self.hidden_dim = int(hidden_dim)
        self.target_tokens = int(target_tokens)
        self.target_dim = int(target_dim)
        self.predict_uncertainty = bool(predict_uncertainty)
        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.horizon_embeddings = nn.Parameter(torch.zeros(1, len(self.future_horizons), hidden_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, max_sequence_tokens, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            heads,
            hidden_dim * int(mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.future_head = nn.Linear(hidden_dim, target_tokens * target_dim)
        self.delta_head = nn.Linear(hidden_dim, target_tokens * target_dim)
        self.uncertainty_head = nn.Linear(hidden_dim, 1) if predict_uncertainty else None

    def _attention_mask(self, context_tokens: int, device):
        action_tokens = self.action_chunk_size
        horizon_tokens = len(self.future_horizons)
        total = context_tokens + action_tokens + horizon_tokens
        mask = torch.ones((total, total), dtype=torch.bool, device=device)
        mask[:context_tokens, :context_tokens] = False
        for index in range(action_tokens):
            row = context_tokens + index
            mask[row, :context_tokens] = False
            mask[row, context_tokens : context_tokens + index + 1] = False
        for index, horizon in enumerate(self.future_horizons):
            row = context_tokens + action_tokens + index
            prefix = min(action_tokens, horizon)
            mask[row, :context_tokens] = False
            mask[row, context_tokens : context_tokens + prefix] = False
            mask[row, row] = False
        return mask

    def forward(
        self,
        current_context,
        short_memory_tokens,
        short_memory_mask,
        long_memory_tokens,
        long_memory_mask,
        action_condition,
    ):
        current_token = self.context_projection(current_context).unsqueeze(1)
        context = torch.cat([current_token, short_memory_tokens, long_memory_tokens], dim=1)
        action_tokens = self.action_projection(action_condition)
        horizon_tokens = self.horizon_embeddings.expand(current_context.shape[0], -1, -1)
        tokens = torch.cat([context, action_tokens, horizon_tokens], dim=1)
        tokens = tokens + self.position_embeddings[:, : tokens.shape[1]]
        context_mask = torch.cat(
            [
                torch.ones((current_context.shape[0], 1), device=current_context.device, dtype=torch.bool),
                short_memory_mask.bool(),
                long_memory_mask.bool(),
            ],
            dim=1,
        )
        valid_mask = torch.cat(
            [
                context_mask,
                torch.ones(
                    (current_context.shape[0], self.action_chunk_size + len(self.future_horizons)),
                    device=current_context.device,
                    dtype=torch.bool,
                ),
            ],
            dim=1,
        )
        encoded = self.transformer(
            tokens,
            mask=self._attention_mask(context.shape[1], tokens.device),
            src_key_padding_mask=~valid_mask,
        )
        horizon_hidden = self.output_norm(encoded[:, -len(self.future_horizons) :])
        shape = (current_context.shape[0], len(self.future_horizons), self.target_tokens, self.target_dim)
        future = self.future_head(horizon_hidden).view(shape)
        delta = self.delta_head(horizon_hidden).view(shape)
        output = {
            "world_belief": horizon_hidden.mean(dim=1),
            "horizon_hidden": horizon_hidden,
            "predicted_future_latents": future,
            "predicted_delta_latents": delta,
        }
        if self.uncertainty_head is not None:
            output["predicted_uncertainty"] = self.uncertainty_head(horizon_hidden).squeeze(-1)
        return output

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
