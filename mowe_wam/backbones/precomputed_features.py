"""Frozen-backbone boundary for the memory-mapped MoWE feature store."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


def _torch_dtype(name: str):
    torch_mod = require_torch()
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_mod.float16
    if normalized in {"fp32", "float32", "float"}:
        return torch_mod.float32
    raise ValueError(f"Unsupported precomputed feature dtype: {name!r}.")


class PrecomputedFeatureBackbone(nn.Module if nn is not None else object):
    """Expose cached pre-action OpenVLA features through the backbone API.

    This module intentionally owns no OpenVLA model or processor.  It is a
    small device/dtype boundary that preserves the exact context dictionary
    consumed by ``FlowWAMSkillPolicy``.
    """

    def __init__(
        self,
        hidden_dim: int,
        *,
        device: str | None = None,
        dtype: str = "bf16",
        num_images_in_input: int = 2,
    ) -> None:
        torch_mod = require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if int(num_images_in_input) != 2:
            raise ValueError("MoWE precomputed features require two ordered views.")
        self.num_images_in_input = 2
        self.device = torch_mod.device(device or ("cuda" if torch_mod.cuda.is_available() else "cpu"))
        self.dtype = _torch_dtype(dtype)
        self.freeze_backbone = True
        self.feature_source = "pre_action_context_cache"
        self.processor = None
        self.resize_resolution = (0, 0)

    def keep_frozen_backbone_eval(self) -> None:
        return None

    def trainable_parameters(self):
        return iter(())

    def extract_context_features(self, batch: dict):
        forbidden = {"labels", "input_ids", "target_actions", "expert_skill_labels"} & set(batch)
        if forbidden:
            raise ValueError(
                "Precomputed context boundary received training targets: "
                f"{sorted(forbidden)}"
            )
        required = {
            "current_visual_views",
            "history_visual_views",
            "long_history_visual_views",
            "precomputed_language",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"Missing precomputed context fields: {missing}")

        current = batch["current_visual_views"].to(device=self.device, dtype=self.dtype)
        history = batch["history_visual_views"].to(device=self.device, dtype=self.dtype)
        long_history = batch["long_history_visual_views"].to(
            device=self.device, dtype=self.dtype
        )
        language = batch["precomputed_language"].to(device=self.device, dtype=self.dtype)
        if current.ndim != 3 or current.shape[1:] != (2, self.hidden_dim):
            raise ValueError(
                "current_visual_views must have shape "
                f"[B, 2, {self.hidden_dim}], got {tuple(current.shape)}."
            )
        if history.ndim != 4 or history.shape[0] != current.shape[0] or history.shape[2:] != (
            2,
            self.hidden_dim,
        ):
            raise ValueError("history_visual_views must have shape [B, K, 2, D].")
        if long_history.ndim != 4 or long_history.shape[0] != current.shape[0] or long_history.shape[2:] != (
            2,
            self.hidden_dim,
        ):
            raise ValueError("long_history_visual_views must have shape [B, M, 2, D].")
        if language.shape != (current.shape[0], self.hidden_dim):
            raise ValueError(
                f"precomputed_language must have shape [B, {self.hidden_dim}]."
            )
        language_tokens = language.unsqueeze(1)
        language_mask = torch.ones(
            (language.shape[0], 1), dtype=torch.bool, device=self.device
        )
        return {
            "current_visual_tokens": current.mean(dim=1).unsqueeze(1),
            "current_visual_views": current,
            "history_visual_views": history,
            "long_history_visual_views": long_history,
            "current_visual": current.mean(dim=1),
            "history_visual": history.mean(dim=2),
            "long_history_visual": long_history.mean(dim=2),
            "language_tokens": language_tokens,
            "language_mask": language_mask,
            "language": language,
        }
