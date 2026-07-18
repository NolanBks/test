"""Memory-mapped, episode-sharded training store for frozen Flow-WAM features.

The module deliberately has no TensorFlow, video, OpenVLA, or DINO imports so
the long-training hot path stays small and predictable.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import Counter, OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import torch

    _DatasetBase = torch.utils.data.Dataset
    _SamplerBase = torch.utils.data.Sampler
except ModuleNotFoundError:
    torch = None
    _DatasetBase = object
    _SamplerBase = object

from mowe_wam.utils.optional import require_torch


FEATURE_STORE_FORMAT = "mowe_feature_store_v1"
FEATURE_STORE_VERSION = 1
UNKNOWN_SKILL = -1


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for the MoWE feature store.") from exc
    return np


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8"
    )
    os.replace(temporary, path)


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _sparse_prefix_indices(prefix_end: int, slots: int) -> list[int]:
    if prefix_end <= 0 or slots <= 0:
        return []
    if prefix_end <= slots:
        return list(range(prefix_end))
    if slots == 1:
        return [0]
    return [(slot * (prefix_end - 1)) // (slots - 1) for slot in range(slots)]


def load_feature_store_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"MoWE feature-store manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FEATURE_STORE_FORMAT or int(payload.get("version", -1)) != 1:
        raise ValueError(f"Unsupported MoWE feature store: {path}")
    return payload


def _validate_episode_arrays(
    *,
    views,
    dino_tokens,
    actions,
    skills,
    expected_contract: dict[str, Any] | None,
) -> dict[str, Any]:
    np = _require_numpy()
    arrays = {
        "openvla_views": np.asarray(views),
        "dino_tokens": np.asarray(dino_tokens),
        "actions": np.asarray(actions),
        "skills": np.asarray(skills),
    }
    length = int(arrays["actions"].shape[0])
    if arrays["openvla_views"].ndim != 3 or arrays["openvla_views"].shape[:2] != (
        length,
        2,
    ):
        raise ValueError("openvla views must have shape [T, 2, D].")
    if arrays["dino_tokens"].ndim != 3 or arrays["dino_tokens"].shape[0] != length:
        raise ValueError("DINO tokens must have shape [T, S, D].")
    if arrays["actions"].shape != (length, 7):
        raise ValueError("actions must have shape [T, 7].")
    if arrays["skills"].shape != (length,):
        raise ValueError("skills must have shape [T].")
    if not np.isfinite(arrays["openvla_views"]).all():
        raise ValueError("OpenVLA features contain NaN/Inf.")
    if not np.isfinite(arrays["dino_tokens"]).all():
        raise ValueError("DINO targets contain NaN/Inf.")
    if not np.isfinite(arrays["actions"]).all():
        raise ValueError("Actions contain NaN/Inf.")
    if not np.isin(arrays["skills"], np.arange(-1, 7)).all():
        raise ValueError("skills must use unknown=-1 or one of seven route IDs.")
    if not np.isin(arrays["actions"][:, -1], [0.0, 1.0]).all():
        raise ValueError("Feature-store gripper actions must be canonical binary 0/1.")
    contract = {
        "openvla_view_shape": list(arrays["openvla_views"].shape[1:]),
        "dino_token_shape": list(arrays["dino_tokens"].shape[1:]),
        "action_shape": [7],
        "openvla_dtype": "float16",
        "dino_dtype": "float16",
        "action_dtype": "float32",
        "skill_dtype": "int8",
        "view_order": ["primary", "wrist"],
    }
    if expected_contract is not None and contract != expected_contract:
        raise ValueError(f"Episode feature contract changed: {contract} != {expected_contract}")
    return arrays | {"contract": contract, "length": length}


class MoWEFeatureStoreWriter:
    """Resumable episode-boundary writer with atomic shard publication."""

    def __init__(
        self,
        root: str | Path,
        *,
        source_contract: dict[str, Any],
        history_length: int = 8,
        long_memory_slots: int = 4,
        future_horizons: Sequence[int] = (1, 4, 8),
        action_chunk_size: int = 8,
        episodes_per_shard: int = 96,
    ) -> None:
        self.root = Path(root)
        self.shards_root = self.root / "shards"
        self.tasks_root = self.root / "tasks"
        self.staging_root = self.root / ".staging"
        self.root.mkdir(parents=True, exist_ok=True)
        self.shards_root.mkdir(exist_ok=True)
        self.tasks_root.mkdir(exist_ok=True)
        self.staging_root.mkdir(exist_ok=True)
        self.source_contract = _jsonable(dict(source_contract))
        self.history_length = int(history_length)
        self.long_memory_slots = int(long_memory_slots)
        self.future_horizons = tuple(int(value) for value in future_horizons)
        self.action_chunk_size = int(action_chunk_size)
        self.episodes_per_shard = max(1, int(episodes_per_shard))
        if self.history_length < 1 or not self.future_horizons or min(self.future_horizons) < 1:
            raise ValueError("Invalid feature-store window contract.")
        if self.action_chunk_size < 1:
            raise ValueError("action_chunk_size must be positive.")

        self.completed_episodes: dict[str, dict[str, Any]] = {}
        self.shards: list[dict[str, Any]] = []
        self.tasks: list[dict[str, Any]] = []
        self._task_by_language: dict[str, int] = {}
        self.feature_contract: dict[str, Any] | None = None
        self._pending: list[dict[str, Any]] = []
        self._validate_or_write_conversion_contract()
        self._load_resume_state()

    def _validate_or_write_conversion_contract(self) -> None:
        path = self.root / "conversion_contract.json"
        contract = {
            "format": FEATURE_STORE_FORMAT,
            "version": FEATURE_STORE_VERSION,
            "source_contract": self.source_contract,
            "window_contract": {
                "history_length": self.history_length,
                "long_memory_slots": self.long_memory_slots,
                "future_horizons": list(self.future_horizons),
                "action_chunk_size": self.action_chunk_size,
            },
        }
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != contract:
                raise ValueError("Cannot resume a feature store with a different conversion contract.")
        else:
            _atomic_json(path, contract)

    def _load_resume_state(self) -> None:
        manifest_path = self.root / "manifest.json"
        if manifest_path.exists():
            existing = load_feature_store_manifest(self.root)
            if existing.get("source_contract") != self.source_contract:
                raise ValueError("Cannot resume a feature store with a different source contract.")
            expected_window = {
                "history_length": self.history_length,
                "long_memory_slots": self.long_memory_slots,
                "future_horizons": list(self.future_horizons),
                "action_chunk_size": self.action_chunk_size,
            }
            if existing.get("window_contract") != expected_window:
                raise ValueError("Cannot resume a feature store with a different window contract.")
        tasks_path = self.root / "tasks.json"
        if tasks_path.exists():
            self.tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
            self._task_by_language = {
                str(item["language"]): int(item["task_id"]) for item in self.tasks
            }
        for shard_dir in sorted(self.shards_root.glob("shard-*")):
            metadata_path = shard_dir / "episodes.json"
            shard_manifest_path = shard_dir / "manifest.json"
            if not metadata_path.exists() or not shard_manifest_path.exists():
                continue
            shard_manifest = json.loads(shard_manifest_path.read_text(encoding="utf-8"))
            if self.feature_contract is None:
                self.feature_contract = shard_manifest["feature_contract"]
            elif self.feature_contract != shard_manifest["feature_contract"]:
                raise ValueError("Completed feature shards have incompatible contracts.")
            episodes = json.loads(metadata_path.read_text(encoding="utf-8"))
            for episode in episodes:
                self.completed_episodes[str(episode["episode_id"])] = episode
            self.shards.append(shard_manifest)
        pending_path = self.staging_root / "pending.json"
        if pending_path.exists():
            payload = json.loads(pending_path.read_text(encoding="utf-8"))
            pending_contract = payload.get("feature_contract")
            if pending_contract is not None:
                if self.feature_contract is None:
                    self.feature_contract = pending_contract
                elif self.feature_contract != pending_contract:
                    raise ValueError("Pending feature episodes have an incompatible contract.")
            for item in payload.get("episodes", []):
                staging = self.root / item["staging_file"]
                if str(item["episode_id"]) in self.completed_episodes:
                    staging.unlink(missing_ok=True)
                    continue
                if not staging.exists():
                    raise FileNotFoundError(
                        f"Pending feature episode is missing: {staging}"
                    )
                if item.get("staging_sha256") != _sha256(staging):
                    raise ValueError(
                        f"Pending feature episode checksum changed: {staging}"
                    )
                self._pending.append(item)
            self._write_pending()
        referenced = {
            str((self.root / item["staging_file"]).resolve()) for item in self._pending
        }
        for orphan in self.staging_root.glob("episode-*.npz"):
            if str(orphan.resolve()) not in referenced:
                orphan.unlink()

    def _write_pending(self) -> None:
        _atomic_json(
            self.staging_root / "pending.json",
            {
                "feature_contract": self.feature_contract,
                "episodes": self._pending,
            },
        )

    def has_episode(self, episode_id: str) -> bool:
        return str(episode_id) in self.completed_episodes or any(
            item["episode_id"] == str(episode_id) for item in self._pending
        )

    def source_episode_identities(self) -> set[tuple[str, int]]:
        """Return committed/staged source identities without importing RLDS."""

        identities = set()
        for item in [*self.completed_episodes.values(), *self._pending]:
            file_key = item.get("source_file_key")
            trajectory_index = item.get("source_traj_index")
            if file_key is not None and trajectory_index is not None:
                identities.add((str(file_key), int(trajectory_index)))
        return identities

    def add_task(self, language: str, language_feature) -> int:
        np = _require_numpy()
        normalized = str(language)
        if normalized in self._task_by_language:
            task_id = self._task_by_language[normalized]
            stored = np.load(self.tasks_root / f"task-{task_id:05d}.npy", mmap_mode="r")
            candidate = np.asarray(language_feature, dtype=np.float16).reshape(-1)
            if tuple(stored.shape) != tuple(candidate.shape):
                raise ValueError("Language feature dimension changed for an existing task.")
            if not np.allclose(stored, candidate, rtol=1e-3, atol=1e-3):
                raise ValueError("Language feature changed for an existing task.")
            return task_id
        feature = np.asarray(language_feature, dtype=np.float16).reshape(-1)
        if not np.isfinite(feature).all():
            raise ValueError("Language feature contains NaN/Inf.")
        if self.feature_contract is not None:
            expected = int(self.feature_contract["openvla_view_shape"][-1])
            if feature.shape != (expected,):
                raise ValueError(f"Language feature must have shape [{expected}].")
        task_id = len(self.tasks)
        temporary = self.tasks_root / f".task-{task_id:05d}.tmp.npy"
        final = self.tasks_root / f"task-{task_id:05d}.npy"
        np.save(temporary, feature, allow_pickle=False)
        os.replace(temporary, final)
        record = {
            "task_id": task_id,
            "language": normalized,
            "feature_file": str(final.relative_to(self.root)),
            "feature_sha256": _sha256(final),
        }
        self.tasks.append(record)
        self._task_by_language[normalized] = task_id
        _atomic_json(self.root / "tasks.json", self.tasks)
        return task_id

    def add_episode(
        self,
        *,
        episode_id: str,
        dataset_name: str,
        partition: str,
        language: str,
        language_feature,
        openvla_views,
        dino_tokens,
        actions,
        skills,
        source_traj_index: int | None = None,
        source_file_key: str | None = None,
    ) -> bool:
        np = _require_numpy()
        episode_id = str(episode_id)
        if self.has_episode(episode_id):
            return False
        if partition not in {"train", "validation"}:
            raise ValueError("partition must be train or validation.")
        validated = _validate_episode_arrays(
            views=openvla_views,
            dino_tokens=dino_tokens,
            actions=actions,
            skills=skills,
            expected_contract=self.feature_contract,
        )
        if self.feature_contract is None:
            self.feature_contract = validated["contract"]
        task_id = self.add_task(language, language_feature)
        pending_index = len(self._pending)
        temporary = self.staging_root / f"episode-{len(self.completed_episodes) + pending_index:06d}.npz"
        np.savez(
            temporary,
            openvla_views=validated["openvla_views"].astype(np.float16, copy=False),
            dino_tokens=validated["dino_tokens"].astype(np.float16, copy=False),
            actions=validated["actions"].astype(np.float32, copy=False),
            skills=validated["skills"].astype(np.int8, copy=False),
        )
        window_count = max(
            0,
            int(validated["length"])
            - max(max(self.future_horizons), self.action_chunk_size - 1),
        )
        target_skill_counts: Counter[int] = Counter()
        for start in range(window_count):
            target_skill_counts.update(
                int(value)
                for value in validated["skills"][start : start + self.action_chunk_size]
            )
        self._pending.append(
            {
                "episode_id": episode_id,
                "dataset_name": str(dataset_name),
                "partition": partition,
                "language": str(language),
                "task_id": task_id,
                "length": int(validated["length"]),
                "window_count": window_count,
                "target_skill_counts": {
                    str(key): int(value)
                    for key, value in sorted(target_skill_counts.items())
                },
                "source_traj_index": source_traj_index,
                "source_file_key": source_file_key,
                "staging_file": str(temporary.relative_to(self.root)),
                "staging_sha256": _sha256(temporary),
            }
        )
        self._write_pending()
        if len(self._pending) >= self.episodes_per_shard:
            self.flush()
        return True

    def flush(self) -> None:
        if not self._pending:
            return
        np = _require_numpy()
        shard_index = len(self.shards)
        final_dir = self.shards_root / f"shard-{shard_index:03d}"
        temporary_dir = self.shards_root / f".shard-{shard_index:03d}.tmp"
        if temporary_dir.exists():
            shutil.rmtree(temporary_dir)
        temporary_dir.mkdir(parents=True)
        total_frames = sum(int(item["length"]) for item in self._pending)
        view_shape = tuple(self.feature_contract["openvla_view_shape"])
        dino_shape = tuple(self.feature_contract["dino_token_shape"])
        outputs = {
            "openvla_views": np.lib.format.open_memmap(
                temporary_dir / "openvla_views.npy",
                mode="w+",
                dtype=np.float16,
                shape=(total_frames, *view_shape),
            ),
            "dino_tokens": np.lib.format.open_memmap(
                temporary_dir / "dino_tokens.npy",
                mode="w+",
                dtype=np.float16,
                shape=(total_frames, *dino_shape),
            ),
            "actions": np.lib.format.open_memmap(
                temporary_dir / "actions.npy",
                mode="w+",
                dtype=np.float32,
                shape=(total_frames, 7),
            ),
            "skills": np.lib.format.open_memmap(
                temporary_dir / "skills.npy",
                mode="w+",
                dtype=np.int8,
                shape=(total_frames,),
            ),
        }
        episodes = []
        offset = 0
        for pending in self._pending:
            with np.load(self.root / pending["staging_file"], allow_pickle=False) as payload:
                length = int(pending["length"])
                for name in outputs:
                    outputs[name][offset : offset + length] = payload[name]
            record = {
                key: value
                for key, value in pending.items()
                if key not in {"staging_file", "staging_sha256"}
            }
            record.update(
                {
                    "episode_index": len(self.completed_episodes) + len(episodes),
                    "shard_id": shard_index,
                    "shard_offset": offset,
                    "window_count": int(pending["window_count"]),
                }
            )
            episodes.append(record)
            offset += length
        for output in outputs.values():
            output.flush()
        del outputs
        _atomic_json(temporary_dir / "episodes.json", episodes)
        files = {}
        for path in sorted(temporary_dir.glob("*.npy")):
            files[path.name] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        shard_manifest = {
            "shard_id": shard_index,
            "path": str(final_dir.relative_to(self.root)),
            "episode_count": len(episodes),
            "frame_count": total_frames,
            "feature_contract": self.feature_contract,
            "files": files,
        }
        _atomic_json(temporary_dir / "manifest.json", shard_manifest)
        if final_dir.exists():
            raise FileExistsError(f"Refusing to replace completed shard: {final_dir}")
        os.replace(temporary_dir, final_dir)
        for pending in self._pending:
            (self.root / pending["staging_file"]).unlink(missing_ok=True)
        for episode in episodes:
            self.completed_episodes[episode["episode_id"]] = episode
        self.shards.append(shard_manifest)
        self._pending.clear()
        self._write_pending()

    def finalize(self) -> dict[str, Any]:
        np = _require_numpy()
        self.flush()
        episodes = sorted(self.completed_episodes.values(), key=lambda item: int(item["episode_index"]))
        for expected, episode in enumerate(episodes):
            if int(episode["episode_index"]) != expected:
                raise ValueError("Feature-store episode indices are not contiguous.")
        episodes_path = self.root / "episodes.jsonl"
        temporary_episodes = episodes_path.with_suffix(".jsonl.tmp")
        with temporary_episodes.open("w", encoding="utf-8") as handle:
            for episode in episodes:
                handle.write(json.dumps(episode, sort_keys=True) + "\n")
        os.replace(temporary_episodes, episodes_path)
        window_count = sum(int(item["window_count"]) for item in episodes)
        temporary_windows = self.root / "windows.tmp.npy"
        windows = np.lib.format.open_memmap(
            temporary_windows, mode="w+", dtype=np.int32, shape=(window_count, 2)
        )
        cursor = 0
        for episode in episodes:
            count = int(episode["window_count"])
            if count:
                windows[cursor : cursor + count, 0] = int(episode["episode_index"])
                windows[cursor : cursor + count, 1] = np.arange(count, dtype=np.int32)
                cursor += count
        windows.flush()
        del windows
        os.replace(temporary_windows, self.root / "windows.npy")
        actual_counts = {
            "episode_count": len(episodes),
            "frame_count": sum(int(item["length"]) for item in episodes),
            "window_count": window_count,
        }
        expected_counts = self.source_contract.get("expected_counts")
        if expected_counts is None:
            completion_contract = {
                "expected_counts": None,
                "actual_counts": actual_counts,
                "counts_match": None,
            }
            counts_match = True
        else:
            normalized_expected = {
                name: int(expected_counts[name])
                for name in ("episode_count", "frame_count", "window_count")
            }
            counts_match = normalized_expected == actual_counts
            completion_contract = {
                "expected_counts": normalized_expected,
                "actual_counts": actual_counts,
                "counts_match": counts_match,
            }
        formal_training_ready = bool(
            self.source_contract.get("formal_training_ready", True)
        ) and bool(counts_match)
        manifest = {
            "format": FEATURE_STORE_FORMAT,
            "version": FEATURE_STORE_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_contract": self.source_contract,
            "completion_contract": completion_contract,
            "formal_training_ready": formal_training_ready,
            "feature_contract": self.feature_contract,
            "window_contract": {
                "history_length": self.history_length,
                "long_memory_slots": self.long_memory_slots,
                "future_horizons": list(self.future_horizons),
                "action_chunk_size": self.action_chunk_size,
            },
            "episode_count": actual_counts["episode_count"],
            "frame_count": actual_counts["frame_count"],
            "window_count": window_count,
            "partition_counts": {
                name: sum(1 for item in episodes if item["partition"] == name)
                for name in ("train", "validation")
            },
            "tasks": self.tasks,
            "shards": self.shards,
            "metadata_files": {
                "conversion_contract": {
                    "path": "conversion_contract.json",
                    "sha256": _sha256(self.root / "conversion_contract.json"),
                },
                "episodes": {"path": "episodes.jsonl", "sha256": _sha256(episodes_path)},
                "windows": {"path": "windows.npy", "sha256": _sha256(self.root / "windows.npy")},
                "tasks": {"path": "tasks.json", "sha256": _sha256(self.root / "tasks.json")},
            },
        }
        _atomic_json(self.root / "manifest.json", manifest)
        (self.staging_root / "pending.json").unlink(missing_ok=True)
        try:
            self.staging_root.rmdir()
        except OSError:
            pass
        return manifest


class _ShardLRU:
    def __init__(self, root: Path, max_open: int) -> None:
        self.root = root
        self.max_open = max(1, int(max_open))
        self._open: OrderedDict[int, dict[str, Any]] = OrderedDict()

    @staticmethod
    def _close(arrays: dict[str, Any]) -> None:
        for array in arrays.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()

    def get(self, shard_id: int) -> dict[str, Any]:
        np = _require_numpy()
        shard_id = int(shard_id)
        if shard_id in self._open:
            arrays = self._open.pop(shard_id)
            self._open[shard_id] = arrays
            return arrays
        directory = self.root / "shards" / f"shard-{shard_id:03d}"
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("openvla_views", "dino_tokens", "actions", "skills")
        }
        self._open[shard_id] = arrays
        while len(self._open) > self.max_open:
            _, evicted = self._open.popitem(last=False)
            self._close(evicted)
        return arrays

    def close(self) -> None:
        while self._open:
            _, arrays = self._open.popitem(last=False)
            self._close(arrays)


class MoWEFeatureWindowDataset(_DatasetBase):
    """Map-style episode windows with bounded read-only memory-map handles."""

    def __init__(
        self,
        root: str | Path,
        *,
        partition: str = "train",
        max_open_feature_shards: int = 2,
        verify_metadata_checksums: bool = True,
    ) -> None:
        require_torch()
        np = _require_numpy()
        self.root = Path(root)
        self.manifest = load_feature_store_manifest(self.root)
        if partition not in {"all", "train", "validation"}:
            raise ValueError("partition must be all, train, or validation.")
        self.partition = partition
        if verify_metadata_checksums:
            for record in self.manifest.get("metadata_files", {}).values():
                path = self.root / record["path"]
                if _sha256(path) != record["sha256"]:
                    raise ValueError(f"Feature-store metadata checksum mismatch: {path}")
        self.episodes = [
            json.loads(line)
            for line in (self.root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(self.episodes) != int(self.manifest["episode_count"]):
            raise ValueError("Feature-store episode count does not match manifest.")
        self._all_windows = np.load(self.root / "windows.npy", mmap_mode="r", allow_pickle=False)
        if self._all_windows.ndim != 2 or self._all_windows.shape[1] != 2:
            raise ValueError("windows.npy must have shape [N, 2].")
        allowed = {
            int(item["episode_index"])
            for item in self.episodes
            if partition == "all" or item["partition"] == partition
        }
        if partition == "all":
            self._window_positions = np.arange(len(self._all_windows), dtype=np.int64)
        else:
            self._window_positions = np.flatnonzero(
                np.isin(self._all_windows[:, 0], np.fromiter(sorted(allowed), dtype=np.int32))
            ).astype(np.int64, copy=False)
        self._episode_by_index = {int(item["episode_index"]): item for item in self.episodes}
        self._shards = _ShardLRU(self.root, max_open_feature_shards)
        self._language_features: dict[int, Any] = {}
        self.window_contract = self.manifest["window_contract"]
        self.feature_contract = self.manifest["feature_contract"]

    def __len__(self) -> int:
        return int(len(self._window_positions))

    @property
    def partition_episode_indices(self) -> list[int]:
        return sorted({int(self._all_windows[position, 0]) for position in self._window_positions})

    def window_positions_by_episode(self) -> dict[int, list[int]]:
        output: dict[int, list[int]] = defaultdict(list)
        for local_index, absolute_position in enumerate(self._window_positions):
            output[int(self._all_windows[absolute_position, 0])].append(local_index)
        return dict(output)

    def skill_counts_for_window_positions(self, positions: Iterable[int]) -> dict[str, int]:
        """Count target-chunk labels without materializing visual or language features."""

        counts: Counter[int] = Counter()
        chunk_size = int(self.window_contract["action_chunk_size"])
        for position in positions:
            absolute_position = int(self._window_positions[int(position)])
            episode_index, step_id = (
                int(value) for value in self._all_windows[absolute_position]
            )
            episode = self._episode_by_index[episode_index]
            arrays = self._shards.get(int(episode["shard_id"]))
            start = int(episode["shard_offset"]) + step_id
            counts.update(
                int(value) for value in arrays["skills"][start : start + chunk_size]
            )
        return {str(key): int(value) for key, value in sorted(counts.items())}

    def partition_target_skill_counts(self) -> dict[str, int]:
        """Aggregate stored episode histograms, scanning only legacy episodes if needed."""

        counts: Counter[int] = Counter()
        positions_by_episode = None
        for episode_index in self.partition_episode_indices:
            stored = self._episode_by_index[episode_index].get("target_skill_counts")
            if stored is None:
                if positions_by_episode is None:
                    positions_by_episode = self.window_positions_by_episode()
                stored = self.skill_counts_for_window_positions(
                    positions_by_episode[episode_index]
                )
            counts.update({int(skill): int(count) for skill, count in stored.items()})
        return {str(key): int(value) for key, value in sorted(counts.items())}

    def _language_feature(self, task_id: int):
        np = _require_numpy()
        task_id = int(task_id)
        if task_id not in self._language_features:
            self._language_features[task_id] = np.load(
                self.root / "tasks" / f"task-{task_id:05d}.npy",
                mmap_mode="r",
                allow_pickle=False,
            )
        return self._language_features[task_id]

    @staticmethod
    def _copy(array):
        np = _require_numpy()
        return np.array(array, copy=True)

    def __getitem__(self, index: int) -> dict[str, Any]:
        torch_mod = require_torch()
        np = _require_numpy()
        absolute_position = int(self._window_positions[int(index)])
        episode_index, step_id = (int(value) for value in self._all_windows[absolute_position])
        episode = self._episode_by_index[episode_index]
        arrays = self._shards.get(int(episode["shard_id"]))
        base = int(episode["shard_offset"])
        step = base + step_id
        history_length = int(self.window_contract["history_length"])
        history_slots = history_length - 1
        long_slots = int(self.window_contract["long_memory_slots"])
        horizons = tuple(int(value) for value in self.window_contract["future_horizons"])
        chunk_size = int(self.window_contract["action_chunk_size"])
        view_shape = tuple(self.feature_contract["openvla_view_shape"])

        history_indices = list(range(max(0, step_id - history_slots), step_id))
        history_pad = history_slots - len(history_indices)
        history_views = np.zeros((history_slots, *view_shape), dtype=np.float16)
        history_actions = np.zeros((history_slots, 7), dtype=np.float32)
        if history_indices:
            source = np.asarray(history_indices, dtype=np.int64) + base
            history_views[history_pad:] = arrays["openvla_views"][source]
            history_actions[history_pad:] = arrays["actions"][source]
        short_start = max(0, step_id - history_slots)
        long_indices = _sparse_prefix_indices(short_start, long_slots)
        long_pad = long_slots - len(long_indices)
        long_views = np.zeros((long_slots, *view_shape), dtype=np.float16)
        long_actions = np.zeros((long_slots, 7), dtype=np.float32)
        if long_indices:
            source = np.asarray(long_indices, dtype=np.int64) + base
            long_views[long_pad:] = arrays["openvla_views"][source]
            long_actions[long_pad:] = arrays["actions"][source]

        target_slice = slice(step, step + chunk_size)
        target_actions = self._copy(arrays["actions"][target_slice]).astype(np.float32, copy=False)
        skill_labels = self._copy(arrays["skills"][target_slice]).astype(np.int64, copy=False)
        future_indices = np.asarray([step + horizon for horizon in horizons], dtype=np.int64)
        current_target = self._copy(arrays["dino_tokens"][step])
        future_targets = self._copy(arrays["dino_tokens"][future_indices])
        current_views = self._copy(arrays["openvla_views"][step])
        language_feature = self._copy(self._language_feature(int(episode["task_id"])))
        return {
            "episode_id": str(episode["episode_id"]),
            "step_id": step_id,
            "dataset_name": str(episode["dataset_name"]),
            "language": str(episode["language"]),
            "current_visual_views": torch_mod.from_numpy(current_views),
            "history_visual_views": torch_mod.from_numpy(history_views),
            "long_history_visual_views": torch_mod.from_numpy(long_views),
            "precomputed_language": torch_mod.from_numpy(language_feature),
            "history_actions": torch_mod.from_numpy(history_actions),
            "history_mask": torch_mod.tensor(
                [False] * history_pad + [True] * len(history_indices) + [True],
                dtype=torch_mod.bool,
            ),
            "long_history_actions": torch_mod.from_numpy(long_actions),
            "long_history_mask": torch_mod.tensor(
                [False] * long_pad + [True] * len(long_indices), dtype=torch_mod.bool
            ),
            "current_latent_target": torch_mod.from_numpy(current_target),
            "future_latent_targets": torch_mod.from_numpy(future_targets),
            "future_horizons": torch_mod.tensor(horizons, dtype=torch_mod.long),
            "future_mask": torch_mod.ones(len(horizons), dtype=torch_mod.bool),
            "target_actions": torch_mod.from_numpy(target_actions),
            "target_motion": torch_mod.from_numpy(target_actions[:, :6].copy()),
            "target_gripper": torch_mod.from_numpy(target_actions[:, 6:7].copy()),
            "expert_skill_labels": torch_mod.from_numpy(skill_labels),
            "expert_skill_mask": torch_mod.from_numpy(skill_labels != UNKNOWN_SKILL),
            "expert_label_source": [
                "sidecar" if int(value) != UNKNOWN_SKILL else "unknown" for value in skill_labels
            ],
            "source_file_key": episode.get("source_file_key"),
            "source_traj_index": episode.get("source_traj_index"),
        }

    def close(self) -> None:
        self._shards.close()
        mmap = getattr(self._all_windows, "_mmap", None)
        if mmap is not None:
            mmap.close()
        for array in self._language_features.values():
            mmap = getattr(array, "_mmap", None)
            if mmap is not None:
                mmap.close()
        self._language_features.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class EpisodeAwareDistributedSampler(_SamplerBase):
    """Assign complete episodes to ranks and resume an exact local cursor."""

    def __init__(
        self,
        dataset: MoWEFeatureWindowDataset,
        *,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 7,
        shuffle: bool = True,
        shuffle_block_size: int = 256,
    ) -> None:
        require_torch()
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.shuffle_block_size = int(shuffle_block_size)
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError("Invalid sampler rank/world_size.")
        if self.shuffle_block_size < 1:
            raise ValueError("shuffle_block_size must be positive.")
        self.order_strategy = "shard_aware_block_shuffle_v1"
        self.epoch = 0
        self.cursor = 0
        self._positions_by_episode = dataset.window_positions_by_episode()
        self._skill_counts_by_episode = {}
        for episode_index, positions in self._positions_by_episode.items():
            stored = dataset._episode_by_index[episode_index].get("target_skill_counts")
            if stored is None:
                stored = dataset.skill_counts_for_window_positions(positions)
            self._skill_counts_by_episode[episode_index] = Counter(
                {int(skill): int(count) for skill, count in stored.items()}
            )
        self._global_skill_counts: Counter[int] = Counter()
        for counts in self._skill_counts_by_episode.values():
            self._global_skill_counts.update(counts)
        self.assignments = self._build_assignments()
        self.local_episode_indices = tuple(self.assignments[self.rank])
        self.local_positions = [
            position
            for episode_index in self.local_episode_indices
            for position in self._positions_by_episode[episode_index]
        ]
        assignment_contract = {
            "assignments": self.assignments,
            "order_strategy": self.order_strategy,
            "shuffle_block_size": self.shuffle_block_size,
        }
        encoded = json.dumps(assignment_contract, sort_keys=True, separators=(",", ":"))
        self.assignment_fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _build_assignments(self) -> list[list[int]]:
        by_suite: dict[str, list[int]] = defaultdict(list)
        for episode_index in self._positions_by_episode:
            episode = self.dataset._episode_by_index[episode_index]
            by_suite[str(episode["dataset_name"])].append(episode_index)
        assignments = [[] for _ in range(self.world_size)]
        total_load = [0] * self.world_size
        skill_load = [Counter() for _ in range(self.world_size)]
        global_window_count = sum(len(values) for values in self._positions_by_episode.values())
        for suite in sorted(by_suite):
            suite_load = [0] * self.world_size
            suite_window_count = sum(
                len(self._positions_by_episode[episode_index])
                for episode_index in by_suite[suite]
            )
            ordered = sorted(
                by_suite[suite],
                key=lambda episode_index: (
                    -sum(
                        count / max(self._global_skill_counts[skill], 1)
                        for skill, count in self._skill_counts_by_episode[episode_index].items()
                    ),
                    -len(self._positions_by_episode[episode_index]),
                    hashlib.sha256(f"{self.seed}:{episode_index}".encode()).hexdigest(),
                ),
            )
            for episode_index in ordered:
                episode_windows = len(self._positions_by_episode[episode_index])
                episode_skills = self._skill_counts_by_episode[episode_index]

                def projected_load(rank: int):
                    ratios = [
                        (total_load[rank] + episode_windows)
                        / max(global_window_count / self.world_size, 1.0),
                        (suite_load[rank] + episode_windows)
                        / max(suite_window_count / self.world_size, 1.0),
                    ]
                    ratios.extend(
                        (skill_load[rank][skill] + count)
                        / max(self._global_skill_counts[skill] / self.world_size, 1.0)
                        for skill, count in episode_skills.items()
                        if self._global_skill_counts[skill] > 0
                    )
                    return (max(ratios), sum(ratios), total_load[rank], rank)

                target = min(
                    range(self.world_size),
                    key=projected_load,
                )
                assignments[target].append(episode_index)
                suite_load[target] += episode_windows
                total_load[target] += episode_windows
                skill_load[target].update(episode_skills)
        return [sorted(values) for values in assignments]

    def _order(self) -> list[int]:
        if not self.shuffle:
            return list(self.local_positions)

        # Randomizing every window globally thrashes the small mmap shard LRU during
        # long runs. Randomize inside each episode, then globally shuffle shard-local
        # blocks: this keeps mixing while bounding shard switches to roughly one per
        # block rather than one per sample.
        positions_by_shard: dict[int, list[int]] = defaultdict(list)
        for episode_index in self.local_episode_indices:
            episode_positions = list(self._positions_by_episode[episode_index])
            random.Random(
                f"{self.seed}:{self.epoch}:episode:{episode_index}"
            ).shuffle(episode_positions)
            shard_id = int(self.dataset._episode_by_index[episode_index]["shard_id"])
            positions_by_shard[shard_id].extend(episode_positions)

        blocks: list[list[int]] = []
        for shard_id in sorted(positions_by_shard):
            positions = positions_by_shard[shard_id]
            blocks.extend(
                positions[start : start + self.shuffle_block_size]
                for start in range(0, len(positions), self.shuffle_block_size)
            )
        random.Random(self.seed + self.epoch).shuffle(blocks)
        return [position for block in blocks for position in block]

    def __iter__(self) -> Iterable[int]:
        order = self._order()
        if self.cursor >= len(order):
            self.epoch += 1
            self.cursor = 0
            order = self._order()
        while self.cursor < len(order):
            position = order[self.cursor]
            self.cursor += 1
            yield position

    def __len__(self) -> int:
        return len(self.local_positions)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "episode_aware_sampler_v1",
            "rank": self.rank,
            "world_size": self.world_size,
            "seed": self.seed,
            "order_strategy": self.order_strategy,
            "shuffle_block_size": self.shuffle_block_size,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "assignment_fingerprint": self.assignment_fingerprint,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != "episode_aware_sampler_v1":
            raise ValueError("Unsupported sampler checkpoint state.")
        expected = (
            self.rank,
            self.world_size,
            self.seed,
            self.order_strategy,
            self.shuffle_block_size,
            self.assignment_fingerprint,
        )
        observed = (
            int(state.get("rank", -1)),
            int(state.get("world_size", -1)),
            int(state.get("seed", -1)),
            state.get("order_strategy"),
            int(state.get("shuffle_block_size", -1)),
            state.get("assignment_fingerprint"),
        )
        if observed != expected:
            raise ValueError(f"Sampler resume contract changed: {observed} != {expected}")
        epoch = int(state.get("epoch", 0))
        cursor = int(state.get("cursor", 0))
        if epoch < 0 or not 0 <= cursor <= len(self.local_positions):
            raise ValueError("Invalid sampler epoch/cursor.")
        self.epoch = epoch
        self.cursor = cursor

    def assignment_report(self, *, include_skill_counts: bool = False) -> dict[str, Any]:
        suites: dict[str, int] = defaultdict(int)
        for episode_index in self.local_episode_indices:
            suite = str(self.dataset._episode_by_index[episode_index]["dataset_name"])
            suites[suite] += len(self._positions_by_episode[episode_index])
        report = {
            "rank": self.rank,
            "world_size": self.world_size,
            "episode_count": len(self.local_episode_indices),
            "window_count": len(self.local_positions),
            "suite_window_counts": dict(sorted(suites.items())),
            "episode_indices": list(self.local_episode_indices),
            "assignment_fingerprint": self.assignment_fingerprint,
            "order_strategy": self.order_strategy,
            "shuffle_block_size": self.shuffle_block_size,
        }
        if include_skill_counts:
            counts: Counter[int] = Counter()
            for episode_index in self.local_episode_indices:
                counts.update(self._skill_counts_by_episode[episode_index])
            report["target_skill_counts"] = {
                str(key): int(value) for key, value in sorted(counts.items())
            }
            report["assignment_strategy"] = (
                "deterministic_suite_window_target_skill_minimax_v1"
            )
        return report


def validate_episode_assignment_reports(
    dataset: MoWEFeatureWindowDataset,
    reports: Sequence[dict[str, Any]],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Prove runtime rank assignments are complete, disjoint, and skill preserving."""

    world_size = int(world_size)
    expected_ranks = set(range(world_size))
    observed_ranks = {int(report.get("rank", -1)) for report in reports}
    issues = []
    if len(reports) != world_size or observed_ranks != expected_ranks:
        issues.append("rank_union")
    owners: dict[int, int] = {}
    overlap = set()
    observed_skills: Counter[str] = Counter()
    for report in reports:
        rank = int(report.get("rank", -1))
        if int(report.get("world_size", -1)) != world_size:
            issues.append(f"world_size:{rank}")
        if int(report.get("episode_count", 0)) < 1:
            issues.append(f"empty_episode_rank:{rank}")
        if int(report.get("window_count", 0)) < 1:
            issues.append(f"empty_window_rank:{rank}")
        for episode_index in report.get("episode_indices", []):
            episode_index = int(episode_index)
            previous = owners.setdefault(episode_index, rank)
            if previous != rank:
                overlap.add(episode_index)
        observed_skills.update(
            {
                str(skill): int(count)
                for skill, count in report.get("target_skill_counts", {}).items()
            }
        )
    expected_episodes = set(dataset.partition_episode_indices)
    if set(owners) != expected_episodes:
        issues.append("episode_union")
    if overlap:
        issues.append("episode_overlap")
    fingerprints = {
        report.get("assignment_fingerprint") for report in reports
    }
    if None in fingerprints or len(fingerprints) != 1:
        issues.append("assignment_fingerprint")
    expected_skills = dataset.partition_target_skill_counts()
    if dict(sorted(observed_skills.items(), key=lambda item: int(item[0]))) != expected_skills:
        issues.append("target_skill_union")
    if sum(int(report.get("window_count", 0)) for report in reports) != len(dataset):
        issues.append("window_union")
    if issues:
        raise RuntimeError(
            "Feature-store runtime assignment contract failed: "
            f"{sorted(set(issues))}; overlap={sorted(overlap)[:8]}."
        )
    return {
        "world_size": world_size,
        "episode_union_complete": True,
        "episode_overlap": [],
        "window_union_complete": True,
        "target_skill_union_complete": True,
        "assignment_fingerprint": next(iter(fingerprints)),
        "episode_count": len(expected_episodes),
        "window_count": len(dataset),
        "target_skill_counts": expected_skills,
    }


def audit_feature_store(root: str | Path, *, verify_all_checksums: bool = False) -> dict[str, Any]:
    """Validate counts, shapes, episode boundaries, windows, and optional hashes."""

    np = _require_numpy()
    root = Path(root)
    manifest = load_feature_store_manifest(root)
    episodes = [
        json.loads(line)
        for line in (root / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    windows = np.load(root / "windows.npy", mmap_mode="r", allow_pickle=False)
    issues = []
    warnings = []
    completion_contract = manifest.get("completion_contract", {})
    if completion_contract.get("counts_match") is False:
        issues.append("expected_counts")
    if len(episodes) != int(manifest["episode_count"]):
        issues.append("episode_count")
    if len(windows) != int(manifest["window_count"]):
        issues.append("window_count")
    expected_windows = sum(int(item["window_count"]) for item in episodes)
    if expected_windows != len(windows):
        issues.append("episode_window_sum")
    if episodes and [int(item["episode_index"]) for item in episodes] != list(range(len(episodes))):
        issues.append("episode_indices")
    if len(windows):
        if int(windows[:, 0].min()) < 0 or int(windows[:, 0].max()) >= len(episodes):
            issues.append("window_episode_range")
        for episode in episodes:
            subset = windows[windows[:, 0] == int(episode["episode_index"]), 1]
            if len(subset) != int(episode["window_count"]) or (
                len(subset) and not np.array_equal(subset, np.arange(len(subset), dtype=np.int32))
            ):
                issues.append(f"window_contract:{episode['episode_id']}")
                break
    shard_frames = 0
    for shard in manifest["shards"]:
        directory = root / shard["path"]
        arrays = {
            name: np.load(directory / f"{name}.npy", mmap_mode="r", allow_pickle=False)
            for name in ("openvla_views", "dino_tokens", "actions", "skills")
        }
        frame_count = int(shard["frame_count"])
        shard_frames += frame_count
        if any(array.shape[0] != frame_count for array in arrays.values()):
            issues.append(f"shard_shape:{shard['shard_id']}")
        chunk_size = int(manifest["window_contract"]["action_chunk_size"])
        for episode in episodes:
            if int(episode["shard_id"]) != int(shard["shard_id"]):
                continue
            stored = episode.get("target_skill_counts")
            if stored is None:
                warnings.append(f"target_skill_counts_missing:{episode['episode_id']}")
            base = int(episode["shard_offset"])
            count = int(episode["window_count"])
            observed: Counter[int] = Counter()
            for start in range(count):
                observed.update(
                    int(value)
                    for value in arrays["skills"][
                        base + start : base + start + chunk_size
                    ]
                )
            normalized_observed = {
                str(key): int(value) for key, value in sorted(observed.items())
            }
            if stored is not None:
                normalized_stored = {
                    str(key): int(value)
                    for key, value in sorted(
                        stored.items(), key=lambda item: int(item[0])
                    )
                }
                if normalized_stored != normalized_observed:
                    issues.append(f"target_skill_counts:{episode['episode_id']}")
        if verify_all_checksums:
            for name, record in shard["files"].items():
                if _sha256(directory / name) != record["sha256"]:
                    issues.append(f"checksum:{shard['shard_id']}:{name}")
    if shard_frames != int(manifest["frame_count"]):
        issues.append("frame_count")
    return {
        "format": manifest["format"],
        "root": str(root.resolve()),
        "episode_count": len(episodes),
        "frame_count": shard_frames,
        "window_count": len(windows),
        "partition_counts": manifest["partition_counts"],
        "shard_count": len(manifest["shards"]),
        "feature_contract": manifest["feature_contract"],
        "source_contract": manifest["source_contract"],
        "completion_contract": completion_contract,
        "formal_training_ready": manifest.get(
            "formal_training_ready",
            manifest.get("source_contract", {}).get("formal_training_ready", True),
        ),
        "checksums_verified": bool(verify_all_checksums),
        "warnings": warnings,
        "issues": issues,
        "valid": not issues,
    }
