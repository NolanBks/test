"""Task-conditioned lightweight fusion for separately encoded camera views."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


class LanguageConditionedViewFusion(nn.Module if nn is not None else object):
    """Fuse ordered view features with one language-conditioned scalar per view."""

    def __init__(
        self,
        feature_dim: int,
        language_dim: int | None = None,
        hidden_dim: int = 128,
        num_views: int = 2,
        view_order: tuple[str, ...] = ("primary", "wrist"),
    ) -> None:
        require_torch()
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.language_dim = int(language_dim or feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_views = int(num_views)
        self.view_order = tuple(str(value) for value in view_order)
        if self.num_views < 1 or len(self.view_order) != self.num_views:
            raise ValueError("view_order length must equal num_views and be non-empty.")
        self.visual_norm = nn.LayerNorm(self.feature_dim)
        self.language_norm = nn.LayerNorm(self.language_dim)
        self.visual_projection = nn.Linear(self.feature_dim, self.hidden_dim, bias=False)
        self.language_projection = nn.Linear(self.language_dim, self.hidden_dim, bias=False)
        self.view_embeddings = nn.Parameter(torch.zeros(self.num_views, self.hidden_dim))
        self.score_head = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.normal_(self.view_embeddings, std=0.02)
        # Neutral start: both valid views initially receive equal weight.
        nn.init.zeros_(self.score_head.weight)

    def forward(self, view_features, language, view_mask=None):
        """Fuse ``[B, ..., V, D]`` features using language ``[B, L]``."""

        if view_features.ndim < 3:
            raise ValueError("view_features must have shape [B, ..., V, D].")
        if view_features.shape[-2:] != (self.num_views, self.feature_dim):
            raise ValueError(
                f"Expected trailing view shape [{self.num_views}, {self.feature_dim}], "
                f"got {tuple(view_features.shape[-2:])}."
            )
        if language.ndim != 2 or language.shape[0] != view_features.shape[0]:
            raise ValueError("language must have shape [B, L] with the same batch size as the views.")
        target_dtype = self.visual_projection.weight.dtype
        view_features = view_features.to(dtype=target_dtype)
        language = language.to(device=view_features.device, dtype=target_dtype)
        language_hidden = self.language_projection(self.language_norm(language))
        for _ in range(view_features.ndim - 3):
            language_hidden = language_hidden.unsqueeze(1)
        language_hidden = language_hidden.unsqueeze(-2)
        embedding_shape = (1,) * (view_features.ndim - 2) + self.view_embeddings.shape
        hidden = torch.tanh(
            self.visual_projection(self.visual_norm(view_features))
            + language_hidden
            + self.view_embeddings.view(embedding_shape)
        )
        scores = self.score_head(hidden).squeeze(-1)
        if view_mask is not None:
            view_mask = view_mask.to(device=scores.device, dtype=torch.bool)
            if view_mask.shape != scores.shape:
                raise ValueError(f"view_mask shape {tuple(view_mask.shape)} != scores {tuple(scores.shape)}.")
            if bool((~view_mask.any(dim=-1)).any()):
                raise ValueError("Every item/timestep must contain at least one valid view.")
            scores = scores.masked_fill(~view_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        fused = (weights.unsqueeze(-1) * view_features).sum(dim=-2)
        float_weights = weights.float()
        entropy = -(float_weights * float_weights.clamp_min(1e-8).log()).sum(dim=-1)
        return {"fused": fused, "weights": weights, "entropy": entropy, "scores": scores}
