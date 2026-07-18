"""Episode-window LIBERO RLDS data path for the latent WAM architecture.

This module has no dependency on predicates or simulator state.  Every sample
is built from one episode; the only optional offline supervision is the
training-only, annotation-derived coarse skill label removed before model input.
"""

from __future__ import annotations

import hashlib
import random
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

try:
    import torch

    _IterableDatasetBase = torch.utils.data.IterableDataset
except ModuleNotFoundError:
    torch = None
    _IterableDatasetBase = object

from mowe_wam.utils.optional import require_torch
from mowe_wam.data.expert_skill_labels import UNKNOWN_LABEL
from mowe_wam.data.cot_skill_sidecar import (
    TensorFlowCotSkillOverlay,
    make_sidecar_episodic_dataset,
    split_cot_skill_marker,
    split_mowe_transport_markers,
)


LIBERO_SEQUENCE_DATASETS = (
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
)


def episode_partition(
    episode_id: str,
    *,
    validation_fraction: float = 0.05,
    split_seed: int = 17,
) -> str:
    """Assign a complete episode to a stable train/validation partition."""

    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if fraction == 0.0:
        return "train"
    digest = hashlib.sha256(f"{int(split_seed)}:{episode_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return "validation" if bucket < fraction else "train"


def rlds_manifest_fingerprint(dataset_root: str | Path, dataset_names: Sequence[str]) -> str:
    """Fingerprint RLDS manifests without hashing multi-gigabyte TFRecord contents."""

    root = Path(dataset_root).resolve()
    digest = hashlib.sha256()
    for dataset_name in dataset_names:
        version_root = root / str(dataset_name) / "1.0.0"
        if not version_root.exists():
            raise FileNotFoundError(f"RLDS dataset version directory not found: {version_root}")
        for path in sorted(version_root.iterdir()):
            authoritative_json = path.name in {"dataset_info.json", "features.json"}
            authoritative_record = ".tfrecord-" in path.name
            if not path.is_file() or not (authoritative_json or authoritative_record):
                continue
            stat = path.stat()
            digest.update(str(path.relative_to(root)).encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            if authoritative_json:
                digest.update(path.read_bytes())
    return digest.hexdigest()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_openvla_path(openvla_root: str | Path) -> Path:
    root = Path(openvla_root)
    if not root.is_absolute():
        root = _repo_root() / root
    if not root.exists():
        raise FileNotFoundError(f"OpenVLA-OFT checkout not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if hasattr(value, "item"):
        try:
            value = value.item()
            if isinstance(value, bytes):
                return value.decode("utf-8")
        except (TypeError, ValueError):
            pass
    return str(value)


def _raw_image_tensor(image: Any):
    """Return a stable uint8 CHW image for the frozen visual teacher."""

    torch_mod = require_torch()
    import numpy as np

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return torch_mod.from_numpy(array.copy()).permute(2, 0, 1).contiguous()


class LatentWAMRLDSBatchTransform:
    """Convert one upstream RLDS timestep without constructing action prompts.

    In particular, this transform never creates ``labels`` and never tokenizes
    the action target into ``input_ids``.  That makes accidental target-action
    leakage into the frozen OpenVLA context encoder structurally impossible.
    """

    def __init__(self, image_transform, use_proprio: bool = True) -> None:
        self.image_transform = image_transform
        self.use_proprio = bool(use_proprio)

    def __call__(self, rlds_batch: dict[str, Any]) -> dict[str, Any]:
        from PIL import Image

        torch_mod = require_torch()
        observation = rlds_batch["observation"]
        missing_views = sorted({"image_primary", "image_wrist"} - set(observation))
        if missing_views:
            raise KeyError(f"Dual-view Flow-WAM requires RLDS observation fields: {missing_views}")
        primary_array = observation["image_primary"][0]
        wrist_array = observation["image_wrist"][0]
        primary_image = Image.fromarray(primary_array).convert("RGB")
        wrist_image = Image.fromarray(wrist_array).convert("RGB")
        # RLDS/TFDS commonly exposes a read-only NumPy view.  Copy before
        # tensorization so downstream normalization never has undefined write
        # behaviour.
        import numpy as np

        actions = torch_mod.as_tensor(np.array(rlds_batch["action"], copy=True), dtype=torch_mod.float32)
        if actions.ndim != 2:
            raise ValueError(f"Expected chunked actions [T, A], got {tuple(actions.shape)}.")

        language, skill_label, marker_file_key, marker_traj_index = split_mowe_transport_markers(
            _decode_text(rlds_batch["task"]["language_instruction"])
        )
        sample = {
            "policy_pixel_values_primary": self.image_transform(primary_image),
            "policy_pixel_values_wrist": self.image_transform(wrist_image),
            # DINO future supervision intentionally remains primary-only.
            "raw_pixel_values": _raw_image_tensor(primary_array),
            "raw_wrist_pixel_values": _raw_image_tensor(wrist_array),
            "actions": actions,
            "language": language,
            "dataset_name": _decode_text(rlds_batch["dataset_name"]),
            "expert_skill_label": skill_label,
            "expert_label_source": "raw_annotation" if skill_label >= 0 else "unknown",
        }
        if "_mowe_source_traj_index" in rlds_batch:
            source_index = np.asarray(rlds_batch["_mowe_source_traj_index"]).reshape(-1)
            if len(source_index):
                sample["source_traj_index"] = int(source_index[0])
        if "_mowe_source_file_key" in rlds_batch:
            source_key = np.asarray(rlds_batch["_mowe_source_file_key"]).reshape(-1)
            if len(source_key):
                sample["source_file_key"] = _decode_text(source_key[0])
        if "source_file_key" not in sample and marker_file_key is not None:
            sample["source_file_key"] = marker_file_key
        if "source_traj_index" not in sample and marker_traj_index is not None:
            sample["source_traj_index"] = marker_traj_index
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            sample["proprio"] = torch_mod.as_tensor(
                np.array(rlds_batch["observation"]["proprio"][0], copy=True), dtype=torch_mod.float32
            ).reshape(-1)
        return sample


class LatentTeacherCacheBatchTransform:
    """Minimal transform for one-pass primary-camera teacher caching."""

    def __call__(self, rlds_batch: dict[str, Any]) -> dict[str, Any]:
        observation = rlds_batch["observation"]
        missing_views = sorted({"image_primary", "image_wrist"} - set(observation))
        if missing_views:
            raise KeyError(f"Teacher cache requires RLDS observation fields: {missing_views}")
        language, _ = split_cot_skill_marker(
            _decode_text(rlds_batch["task"]["language_instruction"])
        )
        return {
            "raw_pixel_values": _raw_image_tensor(observation["image_primary"][0]),
            "raw_wrist_pixel_values": _raw_image_tensor(observation["image_wrist"][0]),
            "language": language,
            "dataset_name": _decode_text(rlds_batch["dataset_name"]),
        }


def _episode_fingerprint(episode: Sequence[dict[str, Any]], dataset_name: str) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(dataset_name.encode("utf-8"))
    digest.update(str(len(episode)).encode("ascii"))
    if episode:
        digest.update(episode[0]["language"].encode("utf-8"))
        for index in sorted({0, len(episode) - 1}):
            for field in ("raw_pixel_values", "raw_wrist_pixel_values"):
                image = episode[index][field]
                digest.update(image.detach().cpu().numpy().tobytes()[:4096])
    return f"{dataset_name}:{digest.hexdigest()[:16]}"


def _sparse_prefix_indices(prefix_end: int, slots: int) -> list[int]:
    """Uniformly choose at most ``slots`` indices from ``[0, prefix_end)``."""

    if prefix_end <= 0 or slots <= 0:
        return []
    if prefix_end <= slots:
        return list(range(prefix_end))
    if slots == 1:
        return [0]
    # Integer arithmetic keeps this deterministic without a numpy dependency.
    return [(slot * (prefix_end - 1)) // (slots - 1) for slot in range(slots)]


def build_episode_windows(
    episode: Sequence[dict[str, Any]],
    history_length: int = 8,
    long_memory_slots: int = 4,
    future_horizons: Sequence[int] = (1, 4, 8),
    action_chunk_size: int = 8,
) -> Iterator[dict[str, Any]]:
    """Yield non-leaky same-episode windows from transformed RLDS steps."""

    torch_mod = require_torch()
    if history_length < 1:
        raise ValueError("history_length must be at least 1.")
    horizons = tuple(int(item) for item in future_horizons)
    if not horizons or any(item < 1 for item in horizons):
        raise ValueError("future_horizons must contain positive integers.")
    if not episode:
        return
    if int(action_chunk_size) < 1:
        raise ValueError("action_chunk_size must be positive.")

    dataset_name = str(episode[0]["dataset_name"])
    episode_id = _episode_fingerprint(episode, dataset_name)
    action_dim = int(episode[0]["actions"].shape[-1])
    history_slots = history_length - 1
    chunk_size = int(action_chunk_size)
    last_current = len(episode) - max(max(horizons), chunk_size - 1)

    for step_id in range(max(0, last_current)):
        history_indices = list(range(max(0, step_id - history_slots), step_id))
        history_pad = history_slots - len(history_indices)
        zero_primary = torch_mod.zeros_like(episode[step_id]["policy_pixel_values_primary"])
        zero_wrist = torch_mod.zeros_like(episode[step_id]["policy_pixel_values_wrist"])
        zero_action = torch_mod.zeros(action_dim, dtype=torch_mod.float32)

        history_primary = [zero_primary.clone() for _ in range(history_pad)] + [
            episode[index]["policy_pixel_values_primary"] for index in history_indices
        ]
        history_wrist = [zero_wrist.clone() for _ in range(history_pad)] + [
            episode[index]["policy_pixel_values_wrist"] for index in history_indices
        ]
        history_actions = [zero_action.clone() for _ in range(history_pad)] + [
            episode[index]["actions"][0] for index in history_indices
        ]
        history_mask = torch_mod.tensor(
            [False] * history_pad + [True] * len(history_indices) + [True], dtype=torch_mod.bool
        )

        short_start = max(0, step_id - history_slots)
        long_indices = _sparse_prefix_indices(short_start, long_memory_slots)
        long_pad = long_memory_slots - len(long_indices)
        long_primary = [zero_primary.clone() for _ in range(long_pad)] + [
            episode[index]["policy_pixel_values_primary"] for index in long_indices
        ]
        long_wrist = [zero_wrist.clone() for _ in range(long_pad)] + [
            episode[index]["policy_pixel_values_wrist"] for index in long_indices
        ]
        long_actions = [zero_action.clone() for _ in range(long_pad)] + [
            episode[index]["actions"][0] for index in long_indices
        ]

        future_indices = [step_id + horizon for horizon in horizons]
        current = episode[step_id]
        target_actions = torch_mod.stack(
            [episode[step_id + offset]["actions"][0].float() for offset in range(chunk_size)],
            dim=0,
        )
        skill_labels = []
        skill_sources = []
        for offset in range(chunk_size):
            target_step = step_id + offset
            if target_step < len(episode):
                skill_labels.append(int(episode[target_step].get("expert_skill_label", UNKNOWN_LABEL)))
                skill_sources.append(str(episode[target_step].get("expert_label_source", "unknown")))
            else:
                skill_labels.append(UNKNOWN_LABEL)
                skill_sources.append("unknown")
        output = {
            "episode_id": episode_id,
            "step_id": step_id,
            "dataset_name": dataset_name,
            "language": current["language"],
            "history_pixel_values_primary": torch_mod.stack(history_primary, dim=0)
            if history_slots
            else zero_primary.new_empty((0, *zero_primary.shape)),
            "history_pixel_values_wrist": torch_mod.stack(history_wrist, dim=0)
            if history_slots
            else zero_wrist.new_empty((0, *zero_wrist.shape)),
            "pixel_values_primary": current["policy_pixel_values_primary"],
            "pixel_values_wrist": current["policy_pixel_values_wrist"],
            "history_mask": history_mask,
            "history_actions": torch_mod.stack(history_actions, dim=0)
            if history_slots
            else zero_action.new_empty((0, action_dim)),
            "long_history_pixel_values_primary": torch_mod.stack(long_primary, dim=0)
            if long_memory_slots
            else zero_primary.new_empty((0, *zero_primary.shape)),
            "long_history_pixel_values_wrist": torch_mod.stack(long_wrist, dim=0)
            if long_memory_slots
            else zero_wrist.new_empty((0, *zero_wrist.shape)),
            "long_history_actions": torch_mod.stack(long_actions, dim=0)
            if long_memory_slots
            else zero_action.new_empty((0, action_dim)),
            "long_history_mask": torch_mod.tensor(
                [False] * long_pad + [True] * len(long_indices), dtype=torch_mod.bool
            ),
            "current_raw_pixel_values": current["raw_pixel_values"],
            "future_raw_pixel_values": torch_mod.stack(
                [episode[index]["raw_pixel_values"] for index in future_indices], dim=0
            ),
            "future_horizons": torch_mod.tensor(horizons, dtype=torch_mod.long),
            "future_mask": torch_mod.ones(len(horizons), dtype=torch_mod.bool),
            "target_actions": target_actions,
            "target_motion": target_actions[..., :6],
            "target_gripper": target_actions[..., 6:7],
            "expert_skill_labels": torch_mod.tensor(skill_labels, dtype=torch_mod.long),
            "expert_skill_mask": torch_mod.tensor(
                [label != UNKNOWN_LABEL for label in skill_labels], dtype=torch_mod.bool
            ),
            "expert_label_source": skill_sources,
        }
        if "proprio" in current:
            output["proprio"] = current["proprio"].float()
        yield output


class LiberoSequenceDataset(_IterableDatasetBase):
    """Round-robin mixture of episodic LIBERO RLDS datasets."""

    def __init__(
        self,
        dataset_root: str | Path,
        processor: Any,
        dataset_names: Sequence[str] = LIBERO_SEQUENCE_DATASETS,
        history_length: int = 8,
        long_memory_slots: int = 4,
        future_horizons: Sequence[int] = (1, 4, 8),
        split: str = "train",
        resize_resolution: tuple[int, int] = (224, 224),
        image_aug: bool = False,
        use_proprio: bool = True,
        openvla_root: str | Path = "external/openvla-oft",
        limit: int | None = None,
        joint_action_normalization: bool = True,
        skill_sidecar_path: str | Path | None = None,
        assume_sidecar_timestep_aligned: bool = True,
        window_shuffle_buffer_size: int = 0,
        episode_partition_name: str = "all",
        validation_fraction: float = 0.0,
        split_seed: int = 17,
        cache_only: bool = False,
        action_chunk_size: int = 8,
        windows_per_episode: int | None = None,
        distributed_rank: int = 0,
        distributed_world_size: int = 1,
        tf_frame_parallel_calls: int | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        root = Path(dataset_root)
        if not root.exists():
            raise FileNotFoundError(f"LIBERO RLDS root not found: {root}")
        if processor is None:
            raise ValueError("LiberoSequenceDataset requires the OpenVLA processor.")
        names = tuple(str(name) for name in dataset_names)
        unknown = sorted(set(names) - set(LIBERO_SEQUENCE_DATASETS))
        if unknown:
            raise ValueError(f"Unsupported sequence datasets: {unknown}")
        if not names:
            raise ValueError("dataset_names cannot be empty.")

        _ensure_openvla_path(openvla_root)
        from prismatic.vla.datasets import EpisodicRLDSDataset

        skill_overlay = TensorFlowCotSkillOverlay(skill_sidecar_path) if skill_sidecar_path is not None else None
        self.distributed_rank = int(distributed_rank)
        self.distributed_world_size = int(distributed_world_size)
        if self.distributed_world_size < 1 or not 0 <= self.distributed_rank < self.distributed_world_size:
            raise ValueError(
                "Invalid distributed dataset identity: "
                f"rank={self.distributed_rank}, world_size={self.distributed_world_size}."
            )
        dataset_class = (
            make_sidecar_episodic_dataset(
                EpisodicRLDSDataset,
                skill_overlay,
                distributed_rank=self.distributed_rank,
                distributed_world_size=self.distributed_world_size,
                frame_num_parallel_calls=tf_frame_parallel_calls,
            )
            if skill_overlay is not None
            else EpisodicRLDSDataset
        )

        self.cache_only = bool(cache_only)
        transform = (
            LatentTeacherCacheBatchTransform()
            if self.cache_only
            else LatentWAMRLDSBatchTransform(
                processor.image_processor.apply_transform,
                use_proprio=use_proprio,
            )
        )
        self.datasets = [
            dataset_class(
                root,
                name,
                transform,
                resize_resolution=resize_resolution,
                shuffle_buffer_size=1,
                train=split == "train",
                image_aug=image_aug,
            )
            for name in names
        ]
        if skill_overlay is None and self.distributed_world_size > 1:
            # The current Flow-WAM training path always uses the exact sidecar
            # overlay above.  Preserve correct fallback behavior for unlabeled
            # inspection paths, although this later shard cannot avoid frame
            # transforms that the upstream base class already attached.
            for dataset in self.datasets:
                dataset.dataset = dataset.dataset.shard(
                    self.distributed_world_size,
                    self.distributed_rank,
                )
        self.dataset_names = names
        self.action_statistics = {
            name: dataset.dataset_statistics for name, dataset in zip(names, self.datasets)
        }
        self.joint_action_normalization = bool(joint_action_normalization)
        self.joint_action_statistics = self._build_joint_action_statistics()
        if skill_sidecar_path is not None and not assume_sidecar_timestep_aligned:
            raise ValueError(
                "The first skill-sidecar implementation requires the documented aligned-timestep assumption."
            )
        self.skill_sidecar = None
        self.skill_sidecar_metadata = skill_overlay.metadata if skill_overlay is not None else None
        self.history_length = int(history_length)
        self.long_memory_slots = int(long_memory_slots)
        self.future_horizons = tuple(int(item) for item in future_horizons)
        self.limit = limit
        self.window_shuffle_buffer_size = max(0, int(window_shuffle_buffer_size))
        self.episode_partition_name = str(episode_partition_name)
        if self.episode_partition_name not in {"all", "train", "validation"}:
            raise ValueError("episode_partition_name must be all, train, or validation.")
        self.validation_fraction = float(validation_fraction)
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in [0, 1).")
        if self.episode_partition_name == "validation" and self.validation_fraction == 0.0:
            raise ValueError("validation partition requires validation_fraction > 0.")
        self.split_seed = int(split_seed)
        self.action_chunk_size = int(action_chunk_size)
        self.windows_per_episode = (
            None if windows_per_episode is None else max(1, int(windows_per_episode))
        )

    def _build_joint_action_statistics(self) -> dict[str, Any]:
        import numpy as np

        lows, highs, masks = [], [], []
        for name in self.dataset_names:
            stats = self.action_statistics[name]
            action = stats.get("action", {})
            if "q01" not in action or "q99" not in action:
                raise KeyError(f"Dataset {name} does not expose q01/q99 action statistics.")
            lows.append(np.asarray(action["q01"], dtype=np.float32))
            highs.append(np.asarray(action["q99"], dtype=np.float32))
            masks.append(np.asarray(action.get("mask", np.ones_like(lows[-1])), dtype=bool))
        joint_low = np.stack(lows).min(axis=0)
        joint_high = np.stack(highs).max(axis=0)
        joint_mask = np.stack(masks).all(axis=0)
        if joint_mask.shape[0] != 7:
            raise ValueError(f"Expected 7D LIBERO action statistics, got {joint_mask.shape[0]} dimensions.")
        joint_mask[-1] = False
        return {
            "q01": joint_low,
            "q99": joint_high,
            "mask": joint_mask,
            "motion_dim": 6,
            "gripper_contract": "canonical_absolute_binary_no_normalization",
            "method": "four_suite_motion_q01_q99_envelope",
        }

    def _apply_joint_action_normalization(self, episode: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not self.joint_action_normalization or not episode:
            return episode
        torch_mod = require_torch()
        name = str(episode[0]["dataset_name"])
        source = self.action_statistics[name]["action"]
        actions = []
        source_low = torch_mod.as_tensor(source["q01"], dtype=torch_mod.float32)
        source_high = torch_mod.as_tensor(source["q99"], dtype=torch_mod.float32)
        source_mask = torch_mod.as_tensor(source.get("mask", [True] * len(source_low)), dtype=torch_mod.bool)
        source_mask[-1] = False
        joint_low = torch_mod.as_tensor(self.joint_action_statistics["q01"], dtype=torch_mod.float32)
        joint_high = torch_mod.as_tensor(self.joint_action_statistics["q99"], dtype=torch_mod.float32)
        joint_mask = torch_mod.as_tensor(self.joint_action_statistics["mask"], dtype=torch_mod.bool)
        normalize_mask = source_mask & joint_mask
        for step in episode:
            normalized = step["actions"].float()
            raw = torch_mod.where(
                source_mask,
                (normalized + 1.0) * 0.5 * (source_high - source_low) + source_low,
                normalized,
            )
            joint = torch_mod.where(
                normalize_mask,
                (2.0 * (raw - joint_low) / (joint_high - joint_low).clamp_min(1e-8) - 1.0).clamp(-1.0, 1.0),
                raw,
            )
            output = dict(step)
            output["actions"] = joint
            actions.append(output)
        return actions

    def _iter_episodes(self):
        torch_mod = require_torch()
        worker = torch_mod.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        num_workers = 1 if worker is None else worker.num_workers
        iterators = [iter(dataset) for dataset in self.datasets]
        active = list(range(len(iterators)))
        episode_counts = [0 for _ in iterators]

        while active:
            next_active = []
            for dataset_index in active:
                try:
                    episode = next(iterators[dataset_index])
                except StopIteration:
                    continue
                episode_index = episode_counts[dataset_index]
                episode_counts[dataset_index] += 1
                next_active.append(dataset_index)
                if episode_index % num_workers != worker_id:
                    continue
                dataset_name = str(episode[0]["dataset_name"]) if episode else self.dataset_names[dataset_index]
                episode_id = _episode_fingerprint(episode, dataset_name)
                assigned = episode_partition(
                    episode_id,
                    validation_fraction=self.validation_fraction,
                    split_seed=self.split_seed,
                )
                if self.episode_partition_name != "all" and assigned != self.episode_partition_name:
                    continue
                if not self.cache_only:
                    episode = self._apply_joint_action_normalization(episode)
                yield episode_id, episode
            active = next_active

    def iter_episode_timesteps(self):
        """Yield each primary-camera timestep once for teacher-cache construction."""

        for episode_id, episode in self._iter_episodes():
            for step_id, step in enumerate(episode):
                yield {
                    "episode_id": episode_id,
                    "step_id": step_id,
                    "raw_pixel_values": step["raw_pixel_values"],
                }

    def iter_transformed_episodes(self):
        """Yield normalized, sidecar-joined episodes for offline conversion.

        This is intentionally an offline-only API.  The feature-store training
        path never imports or constructs this RLDS dataset.
        """

        if self.cache_only:
            raise ValueError("iter_transformed_episodes requires cache_only=False.")
        yield from self._iter_episodes()

    def exclude_source_episodes_before_frame_transform(self, source_keys) -> None:
        """Skip already-published converter episodes before image decoding.

        This is intentionally unavailable for the unlabeled fallback loader,
        whose upstream pipeline has already attached frame transforms.
        """

        keys = {str(value) for value in source_keys}
        if not keys:
            return
        for dataset in self.datasets:
            setter = getattr(dataset, "set_excluded_source_episode_keys", None)
            if setter is None:
                raise RuntimeError(
                    "Pre-frame resume filtering requires the exact sidecar episodic loader."
                )
            setter(keys)

    def _iter_ordered(self):
        for _, episode in self._iter_episodes():
            windows = build_episode_windows(
                episode,
                history_length=self.history_length,
                long_memory_slots=self.long_memory_slots,
                future_horizons=self.future_horizons,
                action_chunk_size=self.action_chunk_size,
            )
            if self.windows_per_episode is None:
                yield from windows
                continue
            maximum_offset = max(max(self.future_horizons), self.action_chunk_size - 1)
            window_count = max(0, len(episode) - maximum_offset)
            selected = set(_sparse_prefix_indices(window_count, self.windows_per_episode))
            for index, sample in enumerate(windows):
                if index in selected:
                    yield sample

    def __iter__(self):
        ordered = self._iter_ordered()
        buffer_size = self.window_shuffle_buffer_size
        if self.limit is not None:
            buffer_size = min(buffer_size, max(1, int(self.limit)))
        if buffer_size <= 1:
            iterator = ordered
        else:
            torch_mod = require_torch()
            worker = torch_mod.utils.data.get_worker_info()
            rng = random if worker is None else random.Random(torch_mod.initial_seed())

            def shuffled():
                buffer = []
                for sample in ordered:
                    if len(buffer) < buffer_size:
                        buffer.append(sample)
                        continue
                    index = rng.randrange(len(buffer))
                    yield buffer[index]
                    buffer[index] = sample
                while buffer:
                    yield buffer.pop(rng.randrange(len(buffer)))

            iterator = shuffled()

        for yielded, sample in enumerate(iterator):
            if self.limit is not None and yielded >= self.limit:
                return
            yield sample
