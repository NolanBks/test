"""Original OpenVLA context adapter using the OFT-compatible multi-image loader."""

from __future__ import annotations

import hashlib
import sys
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch
from mowe_wam.backbones.openvla_identity import (
    ORIGINAL_OPENVLA_REPO_ID,
    validate_openvla_identity,
    validate_original_openvla_reference,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_openvla_path(openvla_root: str | Path) -> Path:
    root = Path(openvla_root)
    if not root.is_absolute():
        root = _repo_root() / root
    if not root.exists():
        raise FileNotFoundError(f"OpenVLA-OFT checkout not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _torch_dtype(name: str):
    torch_mod = require_torch()
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_mod.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_mod.float16
    if normalized in {"fp32", "float32"}:
        return torch_mod.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _load_component_state_dict(path: Path):
    """Load an upstream component checkpoint and remove an optional DDP prefix."""

    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError(f"Expected a state dict in proprio checkpoint: {path}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def _resolve_proprio_checkpoint(path: str | Path | None, vla_path: str) -> Path:
    """Resolve a local proprio component checkpoint without modifying upstream code."""

    candidate = Path(path) if path else Path(vla_path)
    if candidate.is_file():
        return candidate
    if candidate.is_dir():
        matches = sorted(candidate.glob("proprio_projector*checkpoint*.pt"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) == 0:
            raise FileNotFoundError(
                f"use_proprio=true requires a proprio-projector checkpoint. None found in {candidate}; "
                "pass backbone.proprio_checkpoint explicitly."
            )
        raise ValueError(f"Expected one proprio-projector checkpoint in {candidate}, found: {matches}")
    raise FileNotFoundError(
        "use_proprio=true requires a local backbone.proprio_checkpoint when vla_path is a Hub identifier."
    )


class OpenVLAOFTAdapter(nn.Module if nn is not None else object):
    """Frozen original OpenVLA wrapper with legacy and non-leaky context APIs.

    The class name remains for import compatibility.  The mainline weights are
    required to come from ``openvla/openvla-7b``; ``external/openvla-oft`` is
    used only for its compatible model registration and ordered multi-image
    implementation.
    """

    def __init__(
        self,
        vla_path: str = "openvla/openvla-7b",
        checkpoint: str | None = None,
        repo_id: str = ORIGINAL_OPENVLA_REPO_ID,
        revision: str | None = None,
        identity: dict[str, Any] | None = None,
        require_original_backbone: bool = True,
        openvla_root: str | Path = "external/openvla-oft",
        device: str | None = None,
        dtype: str = "bf16",
        freeze_backbone: bool = True,
        feature_source: str = "pre_action_context",
        use_l1_regression: bool = True,
        use_proprio: bool = False,
        proprio_checkpoint: str | None = None,
        proprio_dim: int = 8,
        train_proprio_projector: bool = False,
        num_images_in_input: int = 1,
        language_cache_size: int = 1024,
        visual_cache_size: int = 8192,
    ) -> None:
        require_torch()
        super().__init__()
        self.openvla_root = _ensure_openvla_path(openvla_root)
        self.vla_path = checkpoint or vla_path
        self.repo_id = str(repo_id)
        self.revision = revision
        self.backbone_identity = None
        if require_original_backbone:
            if identity is not None:
                self.backbone_identity = validate_openvla_identity(identity)
                if self.backbone_identity["repo_id"] != self.repo_id:
                    raise ValueError("Adapter repo_id differs from the supplied backbone identity.")
                self.revision = str(self.backbone_identity["revision"])
            validate_original_openvla_reference(
                self.vla_path,
                repo_id=self.repo_id,
                revision=self.revision,
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = _torch_dtype(dtype)
        self.freeze_backbone = freeze_backbone
        self.feature_source = feature_source
        self.use_l1_regression = use_l1_regression
        self.use_proprio = use_proprio
        self.proprio_dim = int(proprio_dim)
        self.train_proprio_projector = bool(train_proprio_projector)
        self.num_images_in_input = num_images_in_input
        self.language_cache_size = max(0, int(language_cache_size))
        self.visual_cache_size = max(0, int(visual_cache_size))
        self._language_cache: OrderedDict[str, Any] = OrderedDict()
        self._visual_cache: OrderedDict[str, Any] = OrderedDict()

        try:
            from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

            from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
            from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
            from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The OpenVLA-compatible loader dependencies are required for real training. "
                "Install the pinned external/openvla-oft runtime before constructing the adapter."
            ) from exc

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # ``openvla/openvla-7b`` snapshots advertise their original single-image
        # implementation through ``auto_map``.  We deliberately register the
        # local OFT-compatible classes above and must keep remote code disabled,
        # otherwise Transformers bypasses that registration and silently loads
        # the snapshot implementation without its ordered multi-image API.
        load_kwargs = {"trust_remote_code": False}
        if not Path(str(self.vla_path)).expanduser().exists():
            load_kwargs["revision"] = self.revision
        self.processor = AutoProcessor.from_pretrained(self.vla_path, **load_kwargs)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.vla_path,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            **load_kwargs,
        ).to(self.device)

        vision_backbone = getattr(self.model, "vision_backbone", None)
        if vision_backbone is not None and hasattr(vision_backbone, "set_num_images_in_input"):
            vision_backbone.set_num_images_in_input(self.num_images_in_input)
        elif self.num_images_in_input != 1:
            raise RuntimeError(
                "This OpenVLA-OFT checkpoint does not expose multi-image configuration; "
                "set num_images_in_input=1 or use a compatible fused-vision backbone."
            )

        self.proprio_projector = None
        if self.use_proprio:
            from prismatic.models.projectors import ProprioProjector

            checkpoint_path = _resolve_proprio_checkpoint(proprio_checkpoint, self.vla_path)
            self.proprio_projector = ProprioProjector(llm_dim=int(self.model.llm_dim), proprio_dim=self.proprio_dim)
            self.proprio_projector.load_state_dict(_load_component_state_dict(checkpoint_path))
            self.proprio_projector.to(device=self.device, dtype=self.dtype)
            if not self.train_proprio_projector:
                self.proprio_projector.eval()
                for param in self.proprio_projector.parameters():
                    param.requires_grad_(False)

        if self.freeze_backbone:
            self.freeze()

        self.hidden_dim = int(getattr(self.model, "llm_dim", getattr(self.model.config, "hidden_size", 4096)))
        self.resize_resolution = tuple(getattr(self.model.config, "image_sizes", (224, 224)))

    def freeze(self) -> None:
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)
        if self.proprio_projector is not None and not self.train_proprio_projector:
            self.proprio_projector.eval()
            for param in self.proprio_projector.parameters():
                param.requires_grad_(False)

    def keep_frozen_backbone_eval(self) -> None:
        """Keep OpenVLA deterministic while allowing an opted-in proprio adapter to train."""

        if not self.freeze_backbone:
            return
        self.model.eval()
        if self.proprio_projector is not None:
            self.proprio_projector.train(self.train_proprio_projector)
    def trainable_parameters(self):
        return (param for param in self.parameters() if param.requires_grad)

    def prepare_inputs(self, batch: dict[str, Any]) -> dict[str, Any]:
        inputs = {
            "input_ids": batch["input_ids"].to(self.device),
            "attention_mask": batch["attention_mask"].to(self.device),
            "pixel_values": batch["pixel_values"].to(self.device, dtype=self.dtype),
            "labels": batch.get("labels"),
            "output_hidden_states": True,
            "return_dict": True,
        }
        if inputs["labels"] is not None:
            inputs["labels"] = inputs["labels"].to(self.device)
        if self.use_proprio and batch.get("proprio") is not None:
            proprio = batch["proprio"].to(self.device, dtype=self.dtype)
            if proprio.shape[-1] != self.proprio_dim:
                raise ValueError(f"Expected proprio dimension {self.proprio_dim}, got {proprio.shape[-1]}.")
            inputs["proprio"] = proprio
            inputs["proprio_projector"] = self.proprio_projector
        return inputs

    def _num_patches(self) -> int:
        vision_backbone = getattr(self.model, "vision_backbone", None)
        if vision_backbone is None:
            return 0
        num_patches = vision_backbone.get_num_patches() * vision_backbone.get_num_images_in_input()
        if self.proprio_projector is not None:
            num_patches += 1
        return int(num_patches)

    def _action_hidden_states(self, output, batch: dict[str, Any]):
        labels = batch.get("labels")
        hidden = output.hidden_states[-1]
        if labels is None:
            return hidden[:, -1:, :]

        try:
            from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
            from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

            token_ids = labels[:, 1:].to(hidden.device)
            action_mask = get_current_action_mask(token_ids) | get_next_actions_mask(token_ids)
            text_hidden = hidden[:, self._num_patches() : -1]
            batch_size = labels.shape[0]
            return text_hidden[action_mask].reshape(batch_size, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
        except Exception:
            return hidden[:, -1:, :]

    def extract_features(self, batch: dict[str, Any]):
        """Legacy action-hidden feature path.

        New latent-WAM code must call :meth:`extract_context_features` instead.
        This method remains available only for the predicate baselines.
        """

        inputs = self.prepare_inputs(batch)
        grad_context = nullcontext() if any(param.requires_grad for param in self.parameters()) else torch.no_grad()
        with grad_context:
            output = self.model(**inputs)

        if self.feature_source == "last_action_hidden":
            action_hidden = self._action_hidden_states(output, batch)
            return action_hidden.mean(dim=1)
        if self.feature_source == "last_token":
            return output.hidden_states[-1][:, -1, :]
        if self.feature_source == "mean_text":
            return output.hidden_states[-1][:, self._num_patches() :, :].mean(dim=1)
        raise ValueError(f"Unsupported feature_source: {self.feature_source}")

    def encode_image_tokens(self, pixel_values):
        """Encode images into projected OpenVLA visual tokens without action text."""

        if pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must have shape [B, C, H, W], got {tuple(pixel_values.shape)}.")
        pixels = pixel_values.to(self.device, dtype=self.dtype)
        grad_context = nullcontext() if any(param.requires_grad for param in self.model.parameters()) else torch.no_grad()
        with grad_context:
            process_vision = getattr(self.model, "_process_vision_features", None)
            if process_vision is not None:
                tokens = process_vision(pixels, language_embeddings=None, use_film=False)
            else:
                vision_backbone = getattr(self.model, "vision_backbone", None)
                projector = getattr(self.model, "projector", None)
                if vision_backbone is None or projector is None:
                    raise RuntimeError("OpenVLA checkpoint does not expose its vision backbone and projector.")
                tokens = projector(vision_backbone(pixels))
        return tokens

    def encode_images(self, pixel_values, *, return_tokens: bool = True):
        """Public non-leaky visual API used by the latent-WAM variant."""

        tokens = self.encode_image_tokens(pixel_values)
        return tokens if return_tokens else tokens.mean(dim=1)

    @staticmethod
    def _visual_cache_key(pixel_values) -> str:
        """Hash the exact processed pixels, so augmentation changes never alias."""

        value = pixel_values.detach().to(device="cpu", dtype=torch.float32).contiguous()
        digest = hashlib.sha256()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def encode_pooled_images(self, pixel_values):
        """Encode pooled frozen visual features with a per-frame LRU.

        Overlapping trajectory windows and variable-prefix replanning repeatedly
        reference the same processed frames.  Caching only the pooled frozen
        feature keeps the cache bounded and never stores action-conditioned
        language hidden states.
        """

        if pixel_values.dim() != 4:
            raise ValueError(
                f"pixel_values must have shape [B, C, H, W], got {tuple(pixel_values.shape)}."
            )
        if not self.freeze_backbone or self.visual_cache_size == 0:
            return self.encode_image_tokens(pixel_values).mean(dim=1)

        keys = [self._visual_cache_key(value) for value in pixel_values]
        resolved = {
            key: self._visual_cache[key]
            for key in dict.fromkeys(keys)
            if key in self._visual_cache
        }
        missing_indices = []
        seen_missing = set()
        for index, key in enumerate(keys):
            if key not in self._visual_cache and key not in seen_missing:
                missing_indices.append(index)
                seen_missing.add(key)
        if missing_indices:
            missing_pixels = torch.stack([pixel_values[index] for index in missing_indices], dim=0)
            missing_features = self.encode_image_tokens(missing_pixels).mean(dim=1)
            for index, feature in zip(missing_indices, missing_features):
                value = feature.detach().cpu()
                resolved[keys[index]] = value
                self._visual_cache[keys[index]] = value
                while len(self._visual_cache) > self.visual_cache_size:
                    self._visual_cache.popitem(last=False)

        values = []
        for key in keys:
            value = resolved[key]
            if key in self._visual_cache:
                self._visual_cache.pop(key)
                self._visual_cache[key] = value
            values.append(value.to(device=self.device, dtype=self.dtype))
        return torch.stack(values, dim=0)

    def _pool_ordered_view_tokens(self, tokens):
        if tokens.dim() != 3 or tokens.shape[1] % self.num_images_in_input != 0:
            raise RuntimeError(
                "OpenVLA visual token count must divide evenly across the configured ordered views."
            )
        tokens_per_view = tokens.shape[1] // self.num_images_in_input
        return tokens.reshape(
            tokens.shape[0], self.num_images_in_input, tokens_per_view, tokens.shape[-1]
        ).mean(dim=2)

    def encode_pooled_views(self, primary_pixel_values, wrist_pixel_values):
        """Encode primary/wrist frames together and retain one pooled vector per view.

        OpenVLA-OFT's fused vision backbone expects independently processed
        images concatenated on the channel axis.  Its output patch tokens remain
        ordered by view, so they are split before pooling rather than collapsed
        into an opaque two-view average.
        """

        if self.num_images_in_input != 2:
            raise ValueError("encode_pooled_views requires num_images_in_input=2.")
        if primary_pixel_values.shape != wrist_pixel_values.shape or primary_pixel_values.dim() != 4:
            raise ValueError("primary and wrist tensors must share shape [B, C, H, W].")
        combined = torch.cat([primary_pixel_values, wrist_pixel_values], dim=1)
        if not self.freeze_backbone or self.visual_cache_size == 0:
            return self._pool_ordered_view_tokens(self.encode_image_tokens(combined))

        keys = [self._visual_cache_key(value) for value in combined]
        resolved = {
            key: self._visual_cache[key]
            for key in dict.fromkeys(keys)
            if key in self._visual_cache
        }
        missing_indices = []
        seen_missing = set()
        for index, key in enumerate(keys):
            if key not in self._visual_cache and key not in seen_missing:
                missing_indices.append(index)
                seen_missing.add(key)
        if missing_indices:
            missing_pixels = torch.stack([combined[index] for index in missing_indices], dim=0)
            missing_features = self._pool_ordered_view_tokens(self.encode_image_tokens(missing_pixels))
            for index, feature in zip(missing_indices, missing_features):
                value = feature.detach().cpu()
                resolved[keys[index]] = value
                self._visual_cache[keys[index]] = value
                while len(self._visual_cache) > self.visual_cache_size:
                    self._visual_cache.popitem(last=False)
        values = []
        for key in keys:
            value = resolved[key]
            if key in self._visual_cache:
                self._visual_cache.pop(key)
                self._visual_cache[key] = value
            values.append(value.to(device=self.device, dtype=self.dtype))
        return torch.stack(values, dim=0)

    def encode_language_tokens(self, language: list[str] | tuple[str, ...]):
        """Encode instruction-only text; no action answer or labels are constructed."""

        if not isinstance(language, (list, tuple)) or not language:
            raise ValueError("language must be a non-empty list or tuple of instruction strings.")
        instructions = [str(value) for value in language]
        if not self.freeze_backbone or self.language_cache_size == 0:
            return self._encode_language_prompts(instructions)

        missing = []
        for instruction in dict.fromkeys(instructions):
            if instruction not in self._language_cache:
                missing.append(instruction)
        if missing:
            tokens, masks = self._encode_language_prompts(missing)
            for index, instruction in enumerate(missing):
                value = tokens[index, masks[index]].detach().cpu()
                self._language_cache[instruction] = value
                while len(self._language_cache) > self.language_cache_size:
                    self._language_cache.popitem(last=False)

        cached = []
        for instruction in instructions:
            value = self._language_cache.pop(instruction)
            self._language_cache[instruction] = value
            cached.append(value.to(device=self.device, dtype=self.dtype))
        max_length = max(value.shape[0] for value in cached)
        output = cached[0].new_zeros((len(cached), max_length, cached[0].shape[-1]))
        mask = torch.zeros((len(cached), max_length), device=self.device, dtype=torch.bool)
        for index, value in enumerate(cached):
            output[index, : value.shape[0]] = value
            mask[index, : value.shape[0]] = True
        return output, mask

    @staticmethod
    def format_instruction_prompt(instruction: str) -> str:
        """Match the upstream LIBERO pre-action prompt without adding action-answer tokens."""

        text = " ".join(str(instruction).strip().lower().split()).rstrip(" .?!")
        if not text:
            raise ValueError("instruction cannot be empty.")
        return f"In: What action should the robot take to {text}?\nOut:"

    def _encode_language_prompts(self, instructions: list[str]):
        prompts = [self.format_instruction_prompt(value) for value in instructions]
        tokenized = self.processor.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)
        grad_context = nullcontext() if any(param.requires_grad for param in self.model.parameters()) else torch.no_grad()
        with grad_context:
            output = self.model.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
        return output.hidden_states[-1], attention_mask.bool()

    def encode_language(self, language_or_input_ids, attention_mask=None):
        """Public instruction encoder; token IDs are allowed only without labels."""

        if torch.is_tensor(language_or_input_ids):
            if attention_mask is None:
                attention_mask = torch.ones_like(language_or_input_ids)
            ids = language_or_input_ids.to(self.device)
            mask = attention_mask.to(self.device)
            grad_context = (
                nullcontext() if any(param.requires_grad for param in self.model.parameters()) else torch.no_grad()
            )
            with grad_context:
                output = self.model.language_model(
                    input_ids=ids,
                    attention_mask=mask,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
            return output.hidden_states[-1], mask.bool()
        return self.encode_language_tokens(language_or_input_ids)

    @staticmethod
    def _masked_token_mean(tokens, mask=None):
        if mask is None:
            return tokens.mean(dim=1)
        weights = mask.to(device=tokens.device, dtype=tokens.dtype).unsqueeze(-1)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def extract_context_features(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Return visual/language context that cannot observe target actions.

        Required image tensors are already processed by the OpenVLA image
        processor.  ``labels`` and tokenized policy prompts are rejected so the
        new world model cannot silently regress action-conditioned VLA hidden
        states.
        """

        if "labels" in batch or "input_ids" in batch:
            raise ValueError(
                "The latent-WAM context path rejects labels/input_ids. Pass raw language strings and image tensors "
                "so target actions cannot leak into OpenVLA context features."
            )
        required = {
            "pixel_values_primary",
            "pixel_values_wrist",
            "history_pixel_values_primary",
            "history_pixel_values_wrist",
            "long_history_pixel_values_primary",
            "long_history_pixel_values_wrist",
            "language",
        }
        missing = sorted(required - set(batch))
        if missing:
            raise KeyError(f"Missing latent-WAM context fields: {missing}")

        current_views = self.encode_pooled_views(
            batch["pixel_values_primary"], batch["pixel_values_wrist"]
        )
        batch_size = current_views.shape[0]

        history_primary = batch["history_pixel_values_primary"]
        history_wrist = batch["history_pixel_values_wrist"]
        if (
            history_primary.shape != history_wrist.shape
            or history_primary.dim() != 5
            or history_primary.shape[0] != batch_size
        ):
            raise ValueError("paired history views must share shape [B, K-1, C, H, W].")
        history_steps = history_primary.shape[1]
        if history_steps:
            history_views = self.encode_pooled_views(
                history_primary.flatten(0, 1), history_wrist.flatten(0, 1)
            ).reshape(
                batch_size, history_steps, 2, -1
            )
        else:
            history_views = current_views.new_empty((batch_size, 0, 2, current_views.shape[-1]))

        long_primary = batch["long_history_pixel_values_primary"]
        long_wrist = batch["long_history_pixel_values_wrist"]
        if (
            long_primary.shape != long_wrist.shape
            or long_primary.dim() != 5
            or long_primary.shape[0] != batch_size
        ):
            raise ValueError("paired long-history views must share shape [B, M, C, H, W].")
        long_steps = long_primary.shape[1]
        if long_steps:
            long_views = self.encode_pooled_views(
                long_primary.flatten(0, 1), long_wrist.flatten(0, 1)
            ).reshape(
                batch_size, long_steps, 2, -1
            )
        else:
            long_views = current_views.new_empty((batch_size, 0, 2, current_views.shape[-1]))

        language_tokens, language_mask = self.encode_language_tokens(batch["language"])
        language = self._masked_token_mean(language_tokens, language_mask)
        return {
            "current_visual_tokens": current_views.mean(dim=1).unsqueeze(1),
            "current_visual_views": current_views,
            "history_visual_views": history_views,
            "long_history_visual_views": long_views,
            # Legacy latent-WAM consumers remain operational with a fixed mean;
            # the Flow-WAM path uses its trainable language-conditioned fusion.
            "current_visual": current_views.mean(dim=1),
            "history_visual": history_views.mean(dim=2),
            "long_history_visual": long_views.mean(dim=2),
            "language_tokens": language_tokens,
            "language_mask": language_mask,
            "language": language,
        }


# Preferred mainline name; keep the historical class import stable.
OpenVLAContextAdapter = OpenVLAOFTAdapter
