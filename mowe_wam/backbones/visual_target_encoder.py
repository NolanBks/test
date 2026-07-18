"""Frozen visual teacher used only to build latent-WAM training targets."""

from __future__ import annotations

import hashlib
import json

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch


TEACHER_TRANSFORM_ID = "upstream_rlds_resize_then_frozen_dinov2_processor_v1"


def teacher_transform_metadata(checkpoint: str, image_resolution, spatial_grid: int) -> dict:
    contract = {
        "transform_id": TEACHER_TRANSFORM_ID,
        "teacher_checkpoint": str(checkpoint),
        "image_resolution": [int(value) for value in image_resolution],
        "spatial_grid": int(spatial_grid),
        "input_contract": "uint8_rgb_chw",
    }
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":"))
    return {**contract, "transform_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


class VisualTargetEncoder(nn.Module if nn is not None else object):
    """DINOv2 teacher with fixed 4x4 spatial pooling."""

    def __init__(
        self,
        checkpoint: str = "facebook/dinov2-small",
        spatial_grid: int = 4,
        target_dim: int = 384,
        num_spatial_tokens: int = 16,
        device: str | None = None,
        dtype: str = "bf16",
    ) -> None:
        require_torch()
        super().__init__()
        if spatial_grid < 1:
            raise ValueError("spatial_grid must be positive.")
        try:
            from transformers import AutoImageProcessor, AutoModel
        except ModuleNotFoundError as exc:
            raise RuntimeError("transformers is required for the frozen DINOv2 visual teacher.") from exc

        self.checkpoint = checkpoint
        self.spatial_grid = int(spatial_grid)
        self.spatial_tokens = self.spatial_grid**2
        if self.spatial_tokens != int(num_spatial_tokens):
            raise ValueError(
                f"spatial_grid={self.spatial_grid} produces {self.spatial_tokens} tokens, "
                f"not configured num_spatial_tokens={num_spatial_tokens}."
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        try:
            self.dtype = dtype_map[dtype.lower()]
        except KeyError as exc:
            raise ValueError(f"Unsupported teacher dtype: {dtype}") from exc
        self.processor = AutoImageProcessor.from_pretrained(checkpoint)
        self.model = AutoModel.from_pretrained(checkpoint, torch_dtype=self.dtype).to(self.device)
        self.target_dim = int(getattr(self.model.config, "hidden_size", 384))
        if self.target_dim != int(target_dim):
            raise ValueError(
                f"Teacher {checkpoint} hidden size is {self.target_dim}, not configured target_dim={target_dim}; "
                "the target path intentionally has no learnable projector."
            )
        self.freeze()

    def freeze(self) -> None:
        super().train(False)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.eval()
        return self

    def encode(self, raw_pixel_values):
        """Encode uint8/float RGB tensors shaped ``[B, 3, H, W]``."""

        if raw_pixel_values.dim() != 4 or raw_pixel_values.shape[1] != 3:
            raise ValueError("raw_pixel_values must have shape [B, 3, H, W].")
        images = [image.permute(1, 2, 0).detach().cpu().numpy() for image in raw_pixel_values]
        with torch.no_grad():
            inputs = self.processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            output = self.model(pixel_values=pixels, return_dict=True)
            tokens = output.last_hidden_state
            patch_count = tokens.shape[1] - 1
            side = int(patch_count**0.5)
            if side * side == patch_count:
                tokens = tokens[:, 1:]
            else:
                side = int(tokens.shape[1] ** 0.5)
                if side * side != tokens.shape[1]:
                    raise RuntimeError(f"Cannot reshape {tokens.shape[1]} DINO tokens into a spatial grid.")
            feature_map = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], side, side)
            pooled = torch.nn.functional.adaptive_avg_pool2d(
                feature_map, (self.spatial_grid, self.spatial_grid)
            )
        return pooled.flatten(2).transpose(1, 2).float()

    def forward(self, raw_pixel_values):
        return self.encode(raw_pixel_values)
