"""Recoverable LeRobot-v3-style canonical archive for MoWE source episodes.

The canonical archive is an offline artifact. Training never imports PyArrow,
launches FFmpeg, or decodes these videos; the mmap feature store remains the
long-training hot path. A chunk manifest is published last and is therefore
the commit marker for its Parquet and MP4 files.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


CANONICAL_ARCHIVE_FORMAT = "mowe_lerobot_v3"
CANONICAL_ARCHIVE_VERSION = 1


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


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for canonical archive conversion.") from exc
    return np


def _require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Canonical archive conversion requires PyArrow. Install requirements-mowe.txt "
            "in the offline conversion environment; it is not needed for training."
        ) from exc
    return pa, pq


def _as_rgb_hwc_uint8(frame):
    np = _require_numpy()
    if hasattr(frame, "detach"):
        frame = frame.detach().cpu().numpy()
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"RGB frame must be rank 3, got {array.shape}.")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        raise ValueError(f"RGB frame must have three channels, got {array.shape}.")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and float(array.max(initial=0.0)) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _resolve_ffmpeg(explicit: str | None = None) -> str:
    if explicit:
        candidate = Path(explicit).expanduser()
        if not candidate.exists():
            raise FileNotFoundError(f"Configured FFmpeg executable does not exist: {candidate}")
        return str(candidate)
    discovered = shutil.which("ffmpeg")
    if discovered:
        return discovered
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Canonical MP4 conversion requires ffmpeg on PATH or the imageio-ffmpeg package."
        ) from exc
    return str(imageio_ffmpeg.get_ffmpeg_exe())


def canonical_conversion_environment(ffmpeg: str | None = None) -> dict[str, str]:
    """Fail fast before a long RLDS scan if offline conversion deps are absent."""

    pa, _ = _require_pyarrow()
    return {"pyarrow": str(pa.__version__), "ffmpeg": _resolve_ffmpeg(ffmpeg)}


class FFmpegVideoWriter:
    """Stream constant-resolution RGB frames to one H.264 MP4 file."""

    def __init__(
        self,
        path: str | Path,
        *,
        width: int,
        height: int,
        fps: float,
        ffmpeg: str | None = None,
        codec: str = "libx264",
        crf: int = 18,
        preset: str = "medium",
    ) -> None:
        if width < 1 or height < 1 or fps <= 0:
            raise ValueError("Video width, height, and fps must be positive.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.frame_count = 0
        command = [
            _resolve_ffmpeg(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            str(float(fps)),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            str(codec),
            "-preset",
            str(preset),
            "-crf",
            str(int(crf)),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def write(self, frame) -> None:
        array = _as_rgb_hwc_uint8(frame)
        if tuple(array.shape) != (self.height, self.width, 3):
            raise ValueError(
                f"Video frame shape changed: {array.shape} != {(self.height, self.width, 3)}"
            )
        if self._process.stdin is None:
            raise RuntimeError("FFmpeg stdin is unavailable.")
        try:
            self._process.stdin.write(array.tobytes())
        except BrokenPipeError as exc:
            error = self._process.stderr.read().decode("utf-8", errors="replace")
            self._process.stderr.close()
            raise RuntimeError(f"FFmpeg stopped while encoding {self.path}: {error}") from exc
        self.frame_count += 1

    def close(self) -> None:
        if self._process.stdin is not None and not self._process.stdin.closed:
            self._process.stdin.close()
        error = self._process.stderr.read().decode("utf-8", errors="replace")
        self._process.stderr.close()
        return_code = self._process.wait()
        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg failed for {self.path} with exit code {return_code}: {error}"
            )
        if not self.path.exists() or self.path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg produced an empty video: {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        if exc_type is None:
            self.close()
        else:
            if self._process.stdin is not None and not self._process.stdin.closed:
                self._process.stdin.close()
            self._process.kill()
            self._process.wait()
            self._process.stderr.close()


class MoWECanonicalArchiveWriter:
    """Episode-boundary, resumable Parquet+MP4 archive writer."""

    def __init__(
        self,
        root: str | Path,
        *,
        source_contract: dict[str, Any],
        fps: float,
        episodes_per_chunk: int = 32,
        ffmpeg: str | None = None,
        video_codec: str = "libx264",
        video_crf: int = 18,
        video_preset: str = "medium",
        video_writer_factory: Callable[..., Any] | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError("Canonical archive fps must be explicitly positive.")
        if episodes_per_chunk < 1:
            raise ValueError("episodes_per_chunk must be positive.")
        self.root = Path(root)
        self.meta_root = self.root / "meta"
        self.chunk_meta_root = self.meta_root / "chunks"
        self.staging_root = self.root / ".staging"
        self.root.mkdir(parents=True, exist_ok=True)
        self.meta_root.mkdir(exist_ok=True)
        self.chunk_meta_root.mkdir(exist_ok=True)
        self.staging_root.mkdir(exist_ok=True)
        self.source_contract = _jsonable(dict(source_contract))
        self.fps = float(fps)
        self.episodes_per_chunk = int(episodes_per_chunk)
        self.ffmpeg = ffmpeg
        self.video_codec = str(video_codec)
        self.video_crf = int(video_crf)
        self.video_preset = str(video_preset)
        self.video_writer_factory = video_writer_factory or FFmpegVideoWriter
        self.tasks: list[dict[str, Any]] = []
        self._task_by_language: dict[str, int] = {}
        self.chunks: list[dict[str, Any]] = []
        self.completed_episodes: dict[str, dict[str, Any]] = {}
        self.pending: list[dict[str, Any]] = []
        self.video_shapes: dict[str, tuple[int, int, int] | None] = {
            "primary": None,
            "wrist": None,
        }
        self._validate_or_write_contract()
        self._load_state()

    @property
    def video_shape(self) -> tuple[int, int, int] | None:
        """Legacy equal-camera shape, or ``None`` for heterogeneous cameras."""

        primary = self.video_shapes["primary"]
        wrist = self.video_shapes["wrist"]
        return primary if primary is not None and primary == wrist else None

    @staticmethod
    def _record_video_shapes(record: dict[str, Any]) -> dict[str, tuple[int, int, int]]:
        shapes = record.get("video_shapes")
        if isinstance(shapes, dict) and {"primary", "wrist"}.issubset(shapes):
            return {
                name: tuple(int(value) for value in shapes[name])
                for name in ("primary", "wrist")
            }
        legacy = record.get("video_shape")
        if legacy:
            shape = tuple(int(value) for value in legacy)
            return {"primary": shape, "wrist": shape}
        raise ValueError("Canonical record is missing per-camera video shapes.")

    def _bind_video_shapes(
        self, shapes: dict[str, tuple[int, int, int]], *, source: str
    ) -> None:
        for name in ("primary", "wrist"):
            observed = tuple(int(value) for value in shapes[name])
            if len(observed) != 3 or observed[-1] != 3 or min(observed) < 1:
                raise ValueError(f"Invalid {name} video shape in {source}: {observed}.")
            if observed[0] % 2 or observed[1] % 2:
                raise ValueError(
                    f"Canonical {name} video height/width must be even for yuv420p: {observed}."
                )
            expected = self.video_shapes[name]
            if expected is None:
                self.video_shapes[name] = observed
            elif expected != observed:
                raise ValueError(
                    f"Canonical {name} video shape changed in {source}: "
                    f"{observed} != {expected}."
                )

    def _validate_or_write_contract(self) -> None:
        contract = {
            "format": CANONICAL_ARCHIVE_FORMAT,
            "version": CANONICAL_ARCHIVE_VERSION,
            "source_contract": self.source_contract,
            "fps": self.fps,
            "episodes_per_chunk": self.episodes_per_chunk,
            "video": {
                "codec": self.video_codec,
                "crf": self.video_crf,
                "preset": self.video_preset,
                "pixel_format": "yuv420p",
                "archive_only_lossy": True,
            },
        }
        path = self.meta_root / "conversion_contract.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != contract:
                raise ValueError("Cannot resume canonical conversion with a different contract.")
        else:
            _atomic_json(path, contract)

    def _load_state(self) -> None:
        tasks_path = self.meta_root / "tasks.json"
        if tasks_path.exists():
            self.tasks = json.loads(tasks_path.read_text(encoding="utf-8"))
            self._task_by_language = {
                str(item["task"]): int(item["task_index"]) for item in self.tasks
            }
        for expected, path in enumerate(sorted(self.chunk_meta_root.glob("chunk-*.json"))):
            chunk = json.loads(path.read_text(encoding="utf-8"))
            if int(chunk.get("chunk_id", -1)) != expected:
                raise ValueError("Canonical chunk manifests are not contiguous.")
            self.chunks.append(chunk)
            for episode in chunk.get("episodes", []):
                self.completed_episodes[str(episode["episode_id"])] = episode
            self._bind_video_shapes(
                self._record_video_shapes(chunk), source=f"committed chunk {expected}"
            )
        pending_path = self.staging_root / "pending.json"
        if pending_path.exists():
            observed_pending = json.loads(pending_path.read_text(encoding="utf-8"))
            for item in observed_pending:
                staging = self.root / item["staging_file"]
                if str(item["episode_id"]) in self.completed_episodes:
                    # The process may have died after publishing the chunk
                    # commit marker but before clearing its staging records.
                    staging.unlink(missing_ok=True)
                    continue
                if not staging.exists():
                    raise FileNotFoundError(f"Canonical pending episode is missing: {staging}")
                if item.get("staging_sha256") != _sha256(staging):
                    raise ValueError(
                        f"Canonical pending episode checksum changed: {staging}"
                    )
                self._bind_video_shapes(
                    self._record_video_shapes(item),
                    source=f"pending episode {item['episode_id']}",
                )
                self.pending.append(item)
            self._write_pending()
        referenced = {str((self.root / item["staging_file"]).resolve()) for item in self.pending}
        for orphan in self.staging_root.glob("episode-*.npz"):
            if str(orphan.resolve()) not in referenced:
                orphan.unlink()

    def _write_pending(self) -> None:
        _atomic_json(self.staging_root / "pending.json", self.pending)

    def has_episode(self, episode_id: str) -> bool:
        episode_id = str(episode_id)
        return episode_id in self.completed_episodes or any(
            str(item["episode_id"]) == episode_id for item in self.pending
        )

    def source_episode_identities(self) -> set[tuple[str, int]]:
        """Return committed/staged source identities for pre-decode resume filtering."""

        identities = set()
        for item in [*self.completed_episodes.values(), *self.pending]:
            file_key = item.get("source_file_key")
            trajectory_index = item.get("source_traj_index")
            if file_key is not None and trajectory_index is not None:
                identities.add((str(file_key), int(trajectory_index)))
        return identities

    def _task_id(self, language: str) -> int:
        language = str(language)
        if language in self._task_by_language:
            return self._task_by_language[language]
        task_id = len(self.tasks)
        self.tasks.append({"task_index": task_id, "task": language})
        self._task_by_language[language] = task_id
        _atomic_json(self.meta_root / "tasks.json", self.tasks)
        return task_id

    def add_episode(
        self,
        *,
        episode_id: str,
        dataset_name: str,
        partition: str,
        language: str,
        actions,
        skills,
        primary_frames: Sequence[Any],
        wrist_frames: Sequence[Any],
        proprio=None,
        timestamps=None,
        source_traj_index: int | None = None,
        source_file_key: str | None = None,
    ) -> bool:
        np = _require_numpy()
        episode_id = str(episode_id)
        if self.has_episode(episode_id):
            return False
        if partition not in {"train", "validation"}:
            raise ValueError("partition must be train or validation.")
        action_array = np.asarray(actions, dtype=np.float32)
        skill_array = np.asarray(skills, dtype=np.int8)
        length = int(action_array.shape[0]) if action_array.ndim else 0
        if action_array.shape != (length, 7) or skill_array.shape != (length,):
            raise ValueError("Canonical actions/skills must have shapes [T,7] and [T].")
        if length < 1 or len(primary_frames) != length or len(wrist_frames) != length:
            raise ValueError("Canonical episode frame/action lengths must match and be non-empty.")
        if not np.isfinite(action_array).all() or not np.isin(skill_array, np.arange(-1, 7)).all():
            raise ValueError("Canonical actions must be finite and skills must be in [-1,6].")
        if not np.isin(action_array[:, -1], [0.0, 1.0]).all():
            raise ValueError("Canonical gripper actions must use shared binary 0/1 semantics.")
        primary = np.stack([_as_rgb_hwc_uint8(frame) for frame in primary_frames])
        wrist = np.stack([_as_rgb_hwc_uint8(frame) for frame in wrist_frames])
        if primary.ndim != 4 or wrist.ndim != 4:
            raise ValueError("Each canonical camera must have a constant [T,H,W,3] shape.")
        video_shapes = {
            "primary": tuple(int(value) for value in primary.shape[1:]),
            "wrist": tuple(int(value) for value in wrist.shape[1:]),
        }
        self._bind_video_shapes(video_shapes, source=f"episode {episode_id}")
        if proprio is None:
            proprio_array = np.empty((length, 0), dtype=np.float32)
        else:
            proprio_array = np.asarray(proprio, dtype=np.float32)
            if proprio_array.ndim != 2 or proprio_array.shape[0] != length:
                raise ValueError("Canonical proprio must have shape [T,D].")
            if not np.isfinite(proprio_array).all():
                raise ValueError("Canonical proprio contains NaN/Inf.")
        if timestamps is None:
            timestamp_array = np.arange(length, dtype=np.float64) / self.fps
        else:
            timestamp_array = np.asarray(timestamps, dtype=np.float64)
            if timestamp_array.shape != (length,) or not np.isfinite(timestamp_array).all():
                raise ValueError("Canonical timestamps must be finite [T].")
            if length > 1 and not bool(np.all(np.diff(timestamp_array) > 0)):
                raise ValueError("Canonical timestamps must be strictly increasing.")
        task_id = self._task_id(language)
        token = hashlib.sha256(episode_id.encode("utf-8")).hexdigest()[:20]
        staging = self.staging_root / f"episode-{token}.npz"
        temporary = self.staging_root / f"episode-{token}.tmp.npz"
        np.savez(
            temporary,
            actions=action_array,
            skills=skill_array,
            primary=primary,
            wrist=wrist,
            proprio=proprio_array,
            timestamps=timestamp_array,
        )
        os.replace(temporary, staging)
        record = {
            "episode_id": episode_id,
            "dataset_name": str(dataset_name),
            "partition": partition,
            "language": str(language),
            "task_id": task_id,
            "length": length,
            "video_shapes": {
                name: list(shape) for name, shape in video_shapes.items()
            },
            "proprio_dim": int(proprio_array.shape[1]),
            "source_traj_index": source_traj_index,
            "source_file_key": source_file_key,
            "staging_file": str(staging.relative_to(self.root)),
            "staging_sha256": _sha256(staging),
        }
        self.pending.append(record)
        self._write_pending()
        if len(self.pending) >= self.episodes_per_chunk:
            self.flush()
        return True

    def _chunk_paths(self, chunk_id: int) -> dict[str, Path]:
        chunk = chunk_id // 1000
        file_id = chunk_id % 1000
        return {
            "data": self.root / f"data/chunk-{chunk:03d}/file-{file_id:03d}.parquet",
            "episodes": self.root
            / f"meta/episodes/chunk-{chunk:03d}/file-{file_id:03d}.parquet",
            "primary_video": self.root
            / f"videos/primary/chunk-{chunk:03d}/file-{file_id:03d}.mp4",
            "wrist_video": self.root
            / f"videos/wrist/chunk-{chunk:03d}/file-{file_id:03d}.mp4",
        }

    @staticmethod
    def _temporary(path: Path) -> Path:
        # Keep the real extension last so FFmpeg can infer the MP4 container.
        return path.with_name(f".{path.stem}.tmp{path.suffix}")

    def flush(self) -> None:
        if not self.pending:
            return
        np = _require_numpy()
        pa, pq = _require_pyarrow()
        chunk_id = len(self.chunks)
        batch = list(self.pending[: self.episodes_per_chunk])
        paths = self._chunk_paths(chunk_id)
        temporary_paths = {name: self._temporary(path) for name, path in paths.items()}
        for path in [*paths.values(), *temporary_paths.values()]:
            path.parent.mkdir(parents=True, exist_ok=True)
        # Any payload without a committed chunk manifest is safe to replace.
        for path in [*paths.values(), *temporary_paths.values()]:
            path.unlink(missing_ok=True)

        frame_episode_index = []
        frame_index = []
        timestamps = []
        dataset_names = []
        task_ids = []
        actions = []
        skills = []
        skill_valid = []
        proprio_values = []
        proprio_valid = []
        source_traj_indices = []
        source_file_keys = []
        episode_rows = []
        committed_records = []
        action_sum = np.zeros(7, dtype=np.float64)
        action_sum_sq = np.zeros(7, dtype=np.float64)
        action_min = np.full(7, np.inf, dtype=np.float64)
        action_max = np.full(7, -np.inf, dtype=np.float64)
        skill_counts: Counter[int] = Counter()
        frame_cursor = 0
        episode_base = len(self.completed_episodes)
        global_frame_base = sum(int(chunk["frame_count"]) for chunk in self.chunks)
        if any(self.video_shapes[name] is None for name in ("primary", "wrist")):
            raise RuntimeError("Canonical camera shapes are unresolved before chunk flush.")

        def writer_options(camera: str) -> dict[str, Any]:
            height, width, _ = self.video_shapes[camera]
            return {
                "width": width,
                "height": height,
                "fps": self.fps,
                "ffmpeg": self.ffmpeg,
                "codec": self.video_codec,
                "crf": self.video_crf,
                "preset": self.video_preset,
            }

        try:
            with self.video_writer_factory(
                temporary_paths["primary_video"], **writer_options("primary")
            ) as primary_writer, self.video_writer_factory(
                temporary_paths["wrist_video"], **writer_options("wrist")
            ) as wrist_writer:
                for local_episode, pending in enumerate(batch):
                    episode_index = episode_base + local_episode
                    staging = self.root / pending["staging_file"]
                    with np.load(staging, allow_pickle=False) as payload:
                        episode_actions = np.asarray(payload["actions"], dtype=np.float32)
                        episode_skills = np.asarray(payload["skills"], dtype=np.int8)
                        episode_timestamps = np.asarray(payload["timestamps"], dtype=np.float64)
                        episode_proprio = np.asarray(payload["proprio"], dtype=np.float32)
                        length = int(pending["length"])
                        for frame in payload["primary"]:
                            primary_writer.write(frame)
                        for frame in payload["wrist"]:
                            wrist_writer.write(frame)
                    frame_episode_index.extend([episode_index] * length)
                    frame_index.extend(range(length))
                    timestamps.extend(episode_timestamps.tolist())
                    dataset_names.extend([pending["dataset_name"]] * length)
                    task_ids.extend([int(pending["task_id"])] * length)
                    actions.extend(episode_actions.tolist())
                    skills.extend(episode_skills.tolist())
                    skill_valid.extend((episode_skills >= 0).tolist())
                    proprio_values.extend(episode_proprio.tolist())
                    proprio_valid.extend([episode_proprio.shape[1] > 0] * length)
                    source_traj_indices.extend([pending.get("source_traj_index")] * length)
                    source_file_keys.extend([pending.get("source_file_key")] * length)
                    action_sum += episode_actions.sum(axis=0, dtype=np.float64)
                    action_sum_sq += np.square(episode_actions, dtype=np.float64).sum(axis=0)
                    action_min = np.minimum(action_min, episode_actions.min(axis=0))
                    action_max = np.maximum(action_max, episode_actions.max(axis=0))
                    skill_counts.update(int(value) for value in episode_skills)
                    committed = {
                        key: value
                        for key, value in pending.items()
                        if key
                        not in {
                            "staging_file",
                            "staging_sha256",
                            "video_shape",
                            "video_shapes",
                        }
                    }
                    committed.update(
                        {
                            "episode_index": episode_index,
                            "global_frame_start": global_frame_base + frame_cursor,
                            "global_frame_end": global_frame_base + frame_cursor + length,
                            "chunk_id": chunk_id,
                            "video_frame_start": frame_cursor,
                            "video_frame_end": frame_cursor + length,
                        }
                    )
                    committed_records.append(committed)
                    episode_rows.append(
                        {
                            **committed,
                            "data_file": str(paths["data"].relative_to(self.root)),
                            "primary_video_file": str(
                                paths["primary_video"].relative_to(self.root)
                            ),
                            "wrist_video_file": str(
                                paths["wrist_video"].relative_to(self.root)
                            ),
                            "source_manifest_fingerprint": self.source_contract.get(
                                "rlds_manifest_fingerprint"
                            )
                            or self.source_contract.get("dataset_fingerprint"),
                            "sidecar_fingerprint": self.source_contract.get(
                                "skill_sidecar_fingerprint"
                            ),
                        }
                    )
                    frame_cursor += length
            if primary_writer.frame_count != frame_cursor or wrist_writer.frame_count != frame_cursor:
                raise RuntimeError("Canonical video frame count differs from Parquet frame count.")

            frame_table = pa.table(
                {
                    "episode_index": pa.array(frame_episode_index, type=pa.int32()),
                    "frame_index": pa.array(frame_index, type=pa.int32()),
                    "timestamp": pa.array(timestamps, type=pa.float64()),
                    "dataset_name": pa.array(dataset_names, type=pa.string()),
                    "task_index": pa.array(task_ids, type=pa.int32()),
                    "action": pa.array(actions, type=pa.list_(pa.float32(), 7)),
                    "skill_id": pa.array(skills, type=pa.int8()),
                    "skill_valid": pa.array(skill_valid, type=pa.bool_()),
                    "proprio": pa.array(proprio_values, type=pa.list_(pa.float32())),
                    "proprio_valid": pa.array(proprio_valid, type=pa.bool_()),
                    "source_traj_index": pa.array(source_traj_indices, type=pa.int64()),
                    "source_file_key": pa.array(source_file_keys, type=pa.string()),
                }
            )
            episode_table = pa.Table.from_pylist(episode_rows)
            pq.write_table(frame_table, temporary_paths["data"], compression="zstd")
            pq.write_table(episode_table, temporary_paths["episodes"], compression="zstd")
            for name, path in temporary_paths.items():
                if not path.exists() or path.stat().st_size == 0:
                    raise RuntimeError(f"Canonical chunk produced an empty {name} file.")
            for name in ("data", "episodes", "primary_video", "wrist_video"):
                os.replace(temporary_paths[name], paths[name])
            files = {
                name: {
                    "path": str(path.relative_to(self.root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for name, path in paths.items()
            }
            manifest = {
                "format": "mowe_canonical_chunk_v1",
                "chunk_id": chunk_id,
                "episode_count": len(committed_records),
                "frame_count": frame_cursor,
                "video_shapes": {
                    name: list(self.video_shapes[name]) for name in ("primary", "wrist")
                },
                "video_shape": list(self.video_shape) if self.video_shape is not None else None,
                "fps": self.fps,
                "files": files,
                "episodes": committed_records,
                "statistics": {
                    "action_count": frame_cursor,
                    "action_sum": action_sum.tolist(),
                    "action_sum_sq": action_sum_sq.tolist(),
                    "action_min": action_min.tolist(),
                    "action_max": action_max.tolist(),
                    "skill_counts": {
                        str(key): int(value) for key, value in sorted(skill_counts.items())
                    },
                },
            }
            _atomic_json(self.chunk_meta_root / f"chunk-{chunk_id:06d}.json", manifest)
        finally:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)

        for pending in batch:
            (self.root / pending["staging_file"]).unlink(missing_ok=True)
        del self.pending[: len(batch)]
        self._write_pending()
        self.chunks.append(manifest)
        for record in committed_records:
            self.completed_episodes[str(record["episode_id"])] = record

    def _aggregate_statistics(self) -> dict[str, Any]:
        np = _require_numpy()
        count = sum(int(chunk["statistics"]["action_count"]) for chunk in self.chunks)
        if count < 1:
            return {"action_count": 0}
        total = sum(
            (np.asarray(chunk["statistics"]["action_sum"], dtype=np.float64) for chunk in self.chunks),
            start=np.zeros(7, dtype=np.float64),
        )
        total_sq = sum(
            (
                np.asarray(chunk["statistics"]["action_sum_sq"], dtype=np.float64)
                for chunk in self.chunks
            ),
            start=np.zeros(7, dtype=np.float64),
        )
        minimum = np.stack(
            [np.asarray(chunk["statistics"]["action_min"]) for chunk in self.chunks]
        ).min(axis=0)
        maximum = np.stack(
            [np.asarray(chunk["statistics"]["action_max"]) for chunk in self.chunks]
        ).max(axis=0)
        variance = np.maximum(total_sq / count - np.square(total / count), 0.0)
        skills: Counter[str] = Counter()
        for chunk in self.chunks:
            skills.update(chunk["statistics"].get("skill_counts", {}))
        return {
            "action_count": count,
            "action_mean": (total / count).tolist(),
            "action_std": np.sqrt(variance).tolist(),
            "action_min": minimum.tolist(),
            "action_max": maximum.tolist(),
            "source_action_statistics": self.source_contract.get("joint_action_statistics"),
            "skill_counts": {str(key): int(value) for key, value in sorted(skills.items())},
        }

    def finalize(self) -> dict[str, Any]:
        pa, pq = _require_pyarrow()
        while self.pending:
            self.flush()
        tasks_table = pa.Table.from_pylist(self.tasks)
        tasks_path = self.meta_root / "tasks.parquet"
        tasks_temporary = self._temporary(tasks_path)
        pq.write_table(tasks_table, tasks_temporary, compression="zstd")
        os.replace(tasks_temporary, tasks_path)
        stats = self._aggregate_statistics()
        _atomic_json(self.meta_root / "stats.json", stats)
        episode_count = sum(int(chunk["episode_count"]) for chunk in self.chunks)
        frame_count = sum(int(chunk["frame_count"]) for chunk in self.chunks)
        serialized_video_shapes = {
            name: list(self.video_shapes[name]) if self.video_shapes[name] is not None else None
            for name in ("primary", "wrist")
        }
        info = {
            "codebase_version": CANONICAL_ARCHIVE_VERSION,
            "format": CANONICAL_ARCHIVE_FORMAT,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fps": self.fps,
            "total_episodes": episode_count,
            "total_frames": frame_count,
            "total_tasks": len(self.tasks),
            "chunks_size": self.episodes_per_chunk,
            "video_shape": list(self.video_shape) if self.video_shape is not None else None,
            "video_shapes": serialized_video_shapes,
            "features": {
                "action": {"dtype": "float32", "shape": [7]},
                "skill_id": {"dtype": "int8", "shape": [1]},
                "primary": {
                    "dtype": "video",
                    "shape": serialized_video_shapes["primary"] or [],
                },
                "wrist": {
                    "dtype": "video",
                    "shape": serialized_video_shapes["wrist"] or [],
                },
            },
            "source_contract": self.source_contract,
        }
        _atomic_json(self.meta_root / "info.json", info)
        manifest = {
            "format": CANONICAL_ARCHIVE_FORMAT,
            "version": CANONICAL_ARCHIVE_VERSION,
            "episode_count": episode_count,
            "frame_count": frame_count,
            "task_count": len(self.tasks),
            "chunk_count": len(self.chunks),
            "fps": self.fps,
            "video_shape": list(self.video_shape) if self.video_shape is not None else None,
            "video_shapes": serialized_video_shapes,
            "source_contract": self.source_contract,
            "chunks": self.chunks,
            "metadata_files": {
                name: {
                    "path": str(path.relative_to(self.root)),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for name, path in {
                    "conversion_contract": self.meta_root / "conversion_contract.json",
                    "info": self.meta_root / "info.json",
                    "stats": self.meta_root / "stats.json",
                    "tasks": tasks_path,
                }.items()
            },
        }
        _atomic_json(self.root / "manifest.json", manifest)
        return manifest


def load_canonical_archive_manifest(root: str | Path) -> dict[str, Any]:
    path = Path(root) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Canonical archive manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != CANONICAL_ARCHIVE_FORMAT
        or int(manifest.get("version", -1)) != CANONICAL_ARCHIVE_VERSION
    ):
        raise ValueError(f"Unsupported canonical archive: {path}")
    return manifest


def audit_canonical_archive(
    root: str | Path, *, verify_checksums: bool = False
) -> dict[str, Any]:
    """Audit committed metadata, Parquet rows, episode offsets, and payloads."""

    _, pq = _require_pyarrow()
    root = Path(root)
    manifest = load_canonical_archive_manifest(root)
    issues: list[str] = []
    total_episodes = 0
    total_frames = 0
    expected_episode_index = 0
    try:
        expected_video_shapes = MoWECanonicalArchiveWriter._record_video_shapes(manifest)
    except ValueError:
        expected_video_shapes = None
        issues.append("video_shapes")
    for expected_chunk, chunk in enumerate(manifest.get("chunks", [])):
        if int(chunk.get("chunk_id", -1)) != expected_chunk:
            issues.append(f"chunk_id:{expected_chunk}")
        try:
            chunk_video_shapes = MoWECanonicalArchiveWriter._record_video_shapes(chunk)
        except ValueError:
            chunk_video_shapes = None
            issues.append(f"video_shapes:{expected_chunk}")
        if (
            expected_video_shapes is not None
            and chunk_video_shapes is not None
            and chunk_video_shapes != expected_video_shapes
        ):
            issues.append(f"video_shapes:{expected_chunk}")
        files = chunk.get("files", {})
        for name in ("data", "episodes", "primary_video", "wrist_video"):
            record = files.get(name, {})
            path = root / str(record.get("path", ""))
            if not path.is_file() or path.stat().st_size <= 0:
                issues.append(f"missing:{expected_chunk}:{name}")
                continue
            if verify_checksums and _sha256(path) != record.get("sha256"):
                issues.append(f"checksum:{expected_chunk}:{name}")
        data_path = root / files["data"]["path"]
        episodes_path = root / files["episodes"]["path"]
        if data_path.exists() and pq.read_metadata(data_path).num_rows != int(chunk["frame_count"]):
            issues.append(f"data_rows:{expected_chunk}")
        if episodes_path.exists() and pq.read_metadata(episodes_path).num_rows != int(
            chunk["episode_count"]
        ):
            issues.append(f"episode_rows:{expected_chunk}")
        for episode in chunk.get("episodes", []):
            if int(episode.get("episode_index", -1)) != expected_episode_index:
                issues.append(f"episode_index:{expected_episode_index}")
            if int(episode["video_frame_end"]) - int(episode["video_frame_start"]) != int(
                episode["length"]
            ):
                issues.append(f"video_offset:{episode.get('episode_id')}")
            expected_episode_index += 1
        total_episodes += int(chunk["episode_count"])
        total_frames += int(chunk["frame_count"])
    for name, record in manifest.get("metadata_files", {}).items():
        path = root / record["path"]
        if not path.exists():
            issues.append(f"metadata_missing:{name}")
        elif verify_checksums and _sha256(path) != record.get("sha256"):
            issues.append(f"metadata_checksum:{name}")
    if total_episodes != int(manifest.get("episode_count", -1)):
        issues.append("episode_count")
    if total_frames != int(manifest.get("frame_count", -1)):
        issues.append("frame_count")
    return {
        "format": "mowe_canonical_archive_audit_v1",
        "root": str(root.resolve()),
        "episode_count": total_episodes,
        "frame_count": total_frames,
        "chunk_count": len(manifest.get("chunks", [])),
        "verify_checksums": bool(verify_checksums),
        "issues": sorted(set(issues)),
        "passed": not issues,
    }
