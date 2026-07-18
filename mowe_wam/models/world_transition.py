"""Temporal prediction of the next control-relevant physical regime."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


class WorldTransitionHead(nn.Module if nn is not None else object):
    """Predict future predicates, progress change, risk, and recovery.

    The head uses current VLA features plus compact action/predicate history.
    It intentionally predicts a control-relevant state transition, not video.
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        predicate_dim: int,
        memory_context_dim: int,
        temporal_dim: int = 512,
        temporal_layers: int = 2,
        temporal_heads: int = 8,
        temporal_ff_dim: int = 1024,
        max_history_steps: int = 4,
    ) -> None:
        require_torch()
        super().__init__()
        if temporal_dim % temporal_heads != 0:
            raise ValueError("temporal_dim must be divisible by temporal_heads.")
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.predicate_dim = int(predicate_dim)
        self.memory_context_dim = int(memory_context_dim)
        self.temporal_dim = int(temporal_dim)
        self.max_history_steps = int(max_history_steps)

        self.current_proj = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.Linear(self.hidden_dim, self.temporal_dim))
        self.action_proj = nn.Linear(self.action_dim, self.temporal_dim)
        self.predicate_proj = nn.Linear(self.predicate_dim, self.temporal_dim)
        self.memory_proj = nn.Linear(self.memory_context_dim, self.temporal_dim)
        self.position = nn.Parameter(torch.zeros(1, self.max_history_steps, self.temporal_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.temporal_dim,
            nhead=int(temporal_heads),
            dim_feedforward=int(temporal_ff_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(temporal_layers))
        self.fuse = nn.Sequential(
            nn.LayerNorm(self.temporal_dim * 3),
            nn.Linear(self.temporal_dim * 3, self.temporal_dim),
            nn.GELU(),
            nn.Linear(self.temporal_dim, self.temporal_dim),
            nn.GELU(),
        )
        self.future_predicate_head = nn.Linear(self.temporal_dim, self.predicate_dim)
        self.progress_delta_head = nn.Linear(self.temporal_dim, 1)
        self.future_risk_head = nn.Linear(self.temporal_dim, 1)
        self.future_recovery_head = nn.Linear(self.temporal_dim, 1)

    def _causal_mask(self, steps: int, device):
        return torch.triu(torch.ones(steps, steps, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, current_features, history_actions, history_predicates, memory_context):
        if current_features.dim() != 2:
            raise ValueError("current_features must have shape [B, hidden_dim].")
        if history_actions.dim() != 3 or history_predicates.dim() != 3:
            raise ValueError("History inputs must have shape [B, K, dim].")
        if history_actions.shape[:2] != history_predicates.shape[:2]:
            raise ValueError("Action and predicate history must share batch and time dimensions.")
        if history_actions.shape[-1] != self.action_dim or history_predicates.shape[-1] != self.predicate_dim:
            raise ValueError("Unexpected action or predicate history dimension.")
        if memory_context.shape[-1] != self.memory_context_dim:
            raise ValueError("Unexpected memory context dimension.")

        steps = history_actions.shape[1]
        if steps < 1 or steps > self.max_history_steps:
            raise ValueError(f"Expected 1..{self.max_history_steps} history steps, got {steps}.")
        history_tokens = self.action_proj(history_actions) + self.predicate_proj(history_predicates) + self.position[:, :steps]
        history_tokens = self.history_encoder(history_tokens, mask=self._causal_mask(steps, history_tokens.device))
        history_feature = history_tokens[:, -1]
        current_feature = self.current_proj(current_features)
        memory_feature = self.memory_proj(memory_context)
        fused = self.fuse(torch.cat([current_feature, history_feature, memory_feature], dim=-1))
        future_predicate_logits = self.future_predicate_head(fused)
        future_risk_logits = self.future_risk_head(fused)
        future_recovery_logits = self.future_recovery_head(fused)
        return {
            "transition_features": fused,
            "future_predicate_logits": future_predicate_logits,
            "future_predicates": torch.sigmoid(future_predicate_logits),
            "progress_delta": self.progress_delta_head(fused),
            "future_risk_logits": future_risk_logits,
            "future_risk": torch.sigmoid(future_risk_logits),
            "future_recovery_logits": future_recovery_logits,
            "future_recovery": torch.sigmoid(future_recovery_logits),
        }
