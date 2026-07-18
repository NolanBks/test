"""Minimal multi-scale latent memory for long-horizon policy context."""

from __future__ import annotations

from dataclasses import dataclass, field

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


def _masked_mean(tokens, mask):
    weights = mask.to(dtype=tokens.dtype).unsqueeze(-1)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class MultiScaleMemoryEncoder(nn.Module if nn is not None else object):
    """Encode dense recent history and sparse episode-prefix landmarks."""

    def __init__(
        self,
        visual_dim: int,
        language_dim: int,
        action_dim: int = 7,
        hidden_dim: int = 512,
        max_short_tokens: int = 8,
        max_long_tokens: int = 4,
        heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        require_torch()
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.visual_projection = nn.Linear(visual_dim, hidden_dim)
        self.language_projection = nn.Linear(language_dim, hidden_dim)
        self.action_projection = nn.Linear(action_dim, hidden_dim)
        self.short_positions = nn.Parameter(torch.zeros(1, max_short_tokens, hidden_dim))
        self.long_positions = nn.Parameter(torch.zeros(1, max_long_tokens + 1, hidden_dim))
        short_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        long_layer = nn.TransformerEncoderLayer(
            hidden_dim, heads, hidden_dim * 4, dropout=dropout, batch_first=True, norm_first=True
        )
        self.short_encoder = nn.TransformerEncoder(short_layer, num_layers=2, enable_nested_tensor=False)
        self.long_encoder = nn.TransformerEncoder(long_layer, num_layers=1, enable_nested_tensor=False)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        history_visual,
        current_visual,
        history_actions,
        history_mask,
        long_history_visual,
        long_history_actions,
        long_history_mask,
        language,
    ):
        batch_size = current_visual.shape[0]
        short_visual = torch.cat([history_visual, current_visual.unsqueeze(1)], dim=1)
        current_action_pad = history_actions.new_zeros((batch_size, 1, history_actions.shape[-1]))
        short_actions = torch.cat([history_actions, current_action_pad], dim=1)
        if short_visual.shape[1] > self.short_positions.shape[1]:
            raise ValueError("Short memory is longer than configured max_short_tokens.")
        short_tokens = (
            self.visual_projection(short_visual)
            + self.action_projection(short_actions)
            + self.short_positions[:, : short_visual.shape[1]]
        )
        short_tokens = self.short_encoder(short_tokens, src_key_padding_mask=~history_mask.bool())

        long_tokens = self.visual_projection(long_history_visual) + self.action_projection(long_history_actions)
        language_token = self.language_projection(language).unsqueeze(1)
        long_tokens = torch.cat([long_tokens, language_token], dim=1)
        long_mask = torch.cat(
            [long_history_mask.bool(), torch.ones((batch_size, 1), device=language.device, dtype=torch.bool)], dim=1
        )
        if long_tokens.shape[1] > self.long_positions.shape[1]:
            raise ValueError("Long memory is longer than configured max_long_tokens.")
        long_tokens = long_tokens + self.long_positions[:, : long_tokens.shape[1]]
        long_tokens = self.long_encoder(long_tokens, src_key_padding_mask=~long_mask)

        short_summary = _masked_mean(short_tokens, history_mask.bool())
        long_summary = _masked_mean(long_tokens, long_mask)
        memory_context = self.output_norm(short_summary + long_summary)
        return {
            "short_memory_tokens": short_tokens,
            "short_memory_mask": history_mask.bool(),
            "long_memory_tokens": long_tokens,
            "long_memory_mask": long_mask,
            "memory_context": memory_context,
            "short_memory_summary": short_summary,
            "long_memory_summary": long_summary,
        }


@dataclass
class OnlineMemoryState:
    """Per-environment episode buffer with training-identical memory indices."""

    history_length: int = 8
    long_memory_slots: int = 4
    max_episode_steps: int = 600
    primary_images: list = field(default_factory=list)
    wrist_images: list = field(default_factory=list)
    actions: list = field(default_factory=list)

    @property
    def images(self):
        """Backward-compatible observability alias for the primary stream."""

        return self.primary_images

    def reset(self) -> None:
        self.primary_images.clear()
        self.wrist_images.clear()
        self.actions.clear()

    def append(self, primary_image, wrist_image, previous_action=None) -> None:
        """Append paired current views and the action executed after the prior pair."""

        if self.primary_images and previous_action is None:
            raise ValueError("Every observation after the first requires previous_action.")
        if not self.primary_images and previous_action is not None:
            raise ValueError("The first observation cannot have a previous_action.")
        if len(self.primary_images) >= self.max_episode_steps:
            raise RuntimeError(
                f"Online episode exceeded max_episode_steps={self.max_episode_steps}; reset the memory state."
            )
        if getattr(primary_image, "shape", None) != getattr(wrist_image, "shape", None):
            raise ValueError("Primary and wrist processed images must have identical shapes.")
        if previous_action is not None:
            self.actions.append(previous_action.detach().cpu() if hasattr(previous_action, "detach") else previous_action)
        self.primary_images.append(
            primary_image.detach().cpu() if hasattr(primary_image, "detach") else primary_image
        )
        self.wrist_images.append(
            wrist_image.detach().cpu() if hasattr(wrist_image, "detach") else wrist_image
        )

    def tensors(self, action_dim: int = 7) -> dict:
        """Materialize chronological padded tensors with the training mask semantics."""

        torch_mod = require_torch()
        if not self.primary_images:
            raise RuntimeError("OnlineMemoryState is empty; append the current observation first.")
        current_primary = self.primary_images[-1]
        current_wrist = self.wrist_images[-1]
        history_slots = self.history_length - 1
        current_index = len(self.primary_images) - 1
        history_indices = list(range(max(0, current_index - history_slots), current_index))
        history_primary = [self.primary_images[index] for index in history_indices]
        history_wrist = [self.wrist_images[index] for index in history_indices]
        actions = [self.actions[index] for index in history_indices]
        history_pad = history_slots - len(history_primary)
        zero_primary = torch_mod.zeros_like(current_primary)
        zero_wrist = torch_mod.zeros_like(current_wrist)
        zero_action = current_primary.new_zeros((action_dim,), dtype=torch_mod.float32)

        short_start = max(0, current_index - history_slots)
        if short_start <= self.long_memory_slots:
            long_indices = list(range(short_start))
        elif self.long_memory_slots == 1:
            long_indices = [0]
        else:
            long_indices = [
                (slot * (short_start - 1)) // (self.long_memory_slots - 1)
                for slot in range(self.long_memory_slots)
            ]
        long_pad = self.long_memory_slots - len(long_indices)
        return {
            "history_pixel_values_primary": torch_mod.stack(
                [zero_primary.clone() for _ in range(history_pad)] + history_primary, dim=0
            ),
            "history_pixel_values_wrist": torch_mod.stack(
                [zero_wrist.clone() for _ in range(history_pad)] + history_wrist, dim=0
            ),
            "pixel_values_primary": current_primary,
            "pixel_values_wrist": current_wrist,
            "history_actions": torch_mod.stack(
                [zero_action.clone() for _ in range(history_pad)]
                + [value if value is not None else zero_action.clone() for value in actions],
                dim=0,
            ),
            "history_mask": torch_mod.tensor(
                [False] * history_pad + [True] * len(history_primary) + [True], dtype=torch_mod.bool
            ),
            "long_history_pixel_values_primary": torch_mod.stack(
                [zero_primary.clone() for _ in range(long_pad)]
                + [self.primary_images[index] for index in long_indices],
                dim=0,
            ),
            "long_history_pixel_values_wrist": torch_mod.stack(
                [zero_wrist.clone() for _ in range(long_pad)]
                + [self.wrist_images[index] for index in long_indices],
                dim=0,
            ),
            "long_history_actions": torch_mod.stack(
                [zero_action.clone() for _ in range(long_pad)]
                + [self.actions[index] for index in long_indices],
                dim=0,
            ),
            "long_history_mask": torch_mod.tensor(
                [False] * long_pad + [True] * len(long_indices), dtype=torch_mod.bool
            ),
        }
