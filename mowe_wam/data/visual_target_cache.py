"""Sharded, metadata-validated cache for frozen visual teacher features."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

from mowe_wam.utils.optional import require_torch


CACHE_FORMAT = "latent_teacher_feature_cache_v2"
MANIFEST_NAME = "manifest.json"


def feature_cache_key(episode_id: str, step_id: int) -> str:
    return f"{episode_id}:{int(step_id)}"


class ShardedVisualTargetCacheWriter:
    """Write one teacher feature per episode timestep without retaining the corpus in RAM."""

    def __init__(
        self,
        output_dir: str | Path,
        metadata: dict[str, Any],
        *,
        shard_size: int = 4096,
    ) -> None:
        if int(shard_size) < 1:
            raise ValueError("shard_size must be positive.")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = self.output_dir / MANIFEST_NAME
        existing_shards = list(self.output_dir.glob("features-*.pt"))
        if manifest.exists() or existing_shards:
            raise FileExistsError(
                f"Teacher cache artifacts already exist in {self.output_dir}; choose a new output directory."
            )
        self.metadata = dict(metadata)
        self.shard_size = int(shard_size)
        self._pending: dict[str, Any] = {}
        self._index: dict[str, str] = {}
        self._shards: list[dict[str, Any]] = []
        self._closed = False

    def __contains__(self, key: str) -> bool:
        return key in self._index or key in self._pending

    @property
    def record_count(self) -> int:
        return len(self._index) + len(self._pending)

    def add(self, key: str, feature) -> bool:
        """Add a CPU float16 ``[spatial_tokens, target_dim]`` feature once."""

        torch = require_torch()
        if self._closed:
            raise RuntimeError("Cannot add to a closed teacher-cache writer.")
        key = str(key)
        if key in self:
            return False
        value = feature.detach().cpu().to(dtype=torch.float16).contiguous()
        if value.ndim != 2:
            raise ValueError("Cached teacher feature must have shape [spatial_tokens, target_dim].")
        self._pending[key] = value
        if len(self._pending) >= self.shard_size:
            self._flush()
        return True

    def _flush(self) -> None:
        if not self._pending:
            return
        torch = require_torch()
        shard_name = f"features-{len(self._shards):05d}.pt"
        torch.save({"format": CACHE_FORMAT, "features": self._pending}, self.output_dir / shard_name)
        for key in self._pending:
            self._index[key] = shard_name
        self._shards.append({"file": shard_name, "records": len(self._pending)})
        self._pending = {}

    def close(self) -> Path:
        if self._closed:
            return self.output_dir / MANIFEST_NAME
        self._flush()
        manifest = {
            "format": CACHE_FORMAT,
            "metadata": self.metadata,
            "record_count": len(self._index),
            "shard_size": self.shard_size,
            "shards": self._shards,
            "index": self._index,
        }
        path = self.output_dir / MANIFEST_NAME
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        self._closed = True
        return path


class ShardedVisualTargetCache:
    """Read feature shards lazily with a small process-local LRU."""

    def __init__(self, path: str | Path, *, max_open_shards: int = 2) -> None:
        root = Path(path)
        manifest_path = root / MANIFEST_NAME if root.is_dir() else root
        if not manifest_path.exists():
            raise FileNotFoundError(f"Teacher-cache manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != CACHE_FORMAT:
            raise ValueError(f"Unsupported sharded teacher-cache format: {manifest.get('format')}")
        if int(max_open_shards) < 1:
            raise ValueError("max_open_shards must be positive.")
        self.root = manifest_path.parent
        self.metadata = dict(manifest.get("metadata", {}))
        self.record_count = int(manifest.get("record_count", 0))
        self.index = {str(key): str(value) for key, value in manifest.get("index", {}).items()}
        if self.record_count != len(self.index):
            raise ValueError(
                f"Teacher-cache record_count={self.record_count} but index has {len(self.index)} keys."
            )
        self.max_open_shards = int(max_open_shards)
        self._loaded: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def _shard(self, name: str) -> dict[str, Any]:
        torch = require_torch()
        if name in self._loaded:
            values = self._loaded.pop(name)
            self._loaded[name] = values
            return values
        state = torch.load(self.root / name, map_location="cpu")
        if state.get("format") != CACHE_FORMAT or not isinstance(state.get("features"), dict):
            raise ValueError(f"Invalid teacher-cache shard: {self.root / name}")
        values = state["features"]
        self._loaded[name] = values
        while len(self._loaded) > self.max_open_shards:
            self._loaded.popitem(last=False)
        return values

    def get(self, key: str):
        key = str(key)
        shard_name = self.index.get(key)
        if shard_name is None:
            raise KeyError(f"Teacher cache miss for {key}")
        shard = self._shard(shard_name)
        if key not in shard:
            raise KeyError(f"Teacher-cache index points to {shard_name}, but {key} is absent.")
        return shard[key].float()

    def window(self, episode_id: str, step_id: int, horizons) -> tuple[Any, Any]:
        torch = require_torch()
        current = self.get(feature_cache_key(episode_id, step_id))
        future = torch.stack(
            [self.get(feature_cache_key(episode_id, int(step_id) + int(horizon))) for horizon in horizons],
            dim=0,
        )
        return current, future


def validate_visual_cache_metadata(actual: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = {
        key: {"cache": actual.get(key), "expected": value}
        for key, value in expected.items()
        if actual.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Teacher cache metadata mismatch: {mismatches}")
