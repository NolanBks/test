"""Strict readers and audits for CALVIN ABC language training data.

The module supports the official per-frame NPZ layout and the 512-shard
``calvin_abc`` RLDS train export. It has no simulator or CALVIN-package
dependency, allowing data gates to run before the evaluation environment is
installed.
"""

from __future__ import annotations

import hashlib
import io
import re
import struct
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mowe_wam.data.expert_skill_labels import (
    LABEL_VERSION,
    SKILL_NAMES,
    SKILL_TO_ID,
    UNKNOWN_LABEL,
    VERB_TO_SKILL,
    label_directive,
)


CALVIN_DATASET_CONTRACT = "official_calvin_abc_language_segments_v1"
CALVIN_RLDS_DATASET_CONTRACT = "calvin_abc_rlds_episode_v1"
CALVIN_RLDS_LABEL_VERSION = "calvin_language_motor_verb_v1"
_EPISODE_FILE = re.compile(r"^(?P<prefix>episode_)(?P<index>\d+)(?P<suffix>\.npz)$")
_RLDS_SHARD = re.compile(
    r"^calvin_abc-train\.tfrecord-(?P<index>\d{5})-of-(?P<count>\d{5})$"
)
_CALVIN_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_CALVIN_VERB_TO_SKILL = {
    **VERB_TO_SKILL,
    "collapse": "pick_grasp",
    "remove": "pick_grasp",
    "take": "pick_grasp",
    "unstack": "pick_grasp",
    "slide": "push_pull",
    "sweep": "push_pull",
    "store": "place_release",
    "toggle": "open_close",
}


def label_calvin_instruction(text: str) -> tuple[int, str, str]:
    """Map CALVIN paraphrases to the unchanged seven-route motor taxonomy."""

    directive = _normalize_text(text)
    words = _CALVIN_WORD.findall(directive.lower())
    if not words:
        return UNKNOWN_LABEL, "", directive
    candidates = words if words[0] in {"go", "in"} else words[:1]
    for word in candidates:
        skill = _CALVIN_VERB_TO_SKILL.get(word)
        if skill is not None:
            return SKILL_TO_ID[skill], word, directive
    return UNKNOWN_LABEL, words[0], directive


def _require_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("NumPy is required for CALVIN dataset audit.") from exc
    return np


def _normalize_text(value: Any) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if hasattr(value, "item") and not isinstance(value, str):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    text = " ".join(str(value).strip().split())
    if not text:
        raise ValueError("CALVIN language annotation cannot be empty.")
    return text


def _sha256(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def resolve_calvin_abc_training_root(root: str | Path) -> Path:
    """Resolve ``task_ABC_D/training`` and reject D/validation leakage."""

    root = Path(root).expanduser().resolve()
    if root.name == "task_ABC_D" and (root / "training").is_dir():
        root = root / "training"
    if root.name != "training" or root.parent.name != "task_ABC_D":
        raise ValueError(
            "CALVIN ABC converter accepts only the official task_ABC_D/training; "
            "validation/D and other split roots are forbidden."
        )
    if not root.is_dir():
        raise FileNotFoundError(f"CALVIN ABC training directory not found: {root}")
    return root


class CalvinLanguageSegmentDataset:
    """Iterate official language-annotated ABC segments without crossing goals."""

    required_frame_keys = ("rgb_static", "rgb_gripper", "robot_obs", "rel_actions")

    def __init__(
        self,
        root: str | Path,
        *,
        min_segment_length: int = 9,
        official_repo_commit: str = "fa03f01f19c65920e18cf37398a9ce859274af76",
    ) -> None:
        np = _require_numpy()
        self.root = resolve_calvin_abc_training_root(root)
        self.min_segment_length = int(min_segment_length)
        if self.min_segment_length < 1:
            raise ValueError("min_segment_length must be positive.")
        self.official_repo_commit = str(official_repo_commit)
        self.window_max_offset = self.min_segment_length - 1
        self.dataset_name = "calvin_abc_language_segments"
        self.annotation_path = self.root / "lang_annotations/auto_lang_ann.npy"
        if not self.annotation_path.is_file():
            raise FileNotFoundError(
                f"Official CALVIN language annotations not found: {self.annotation_path}"
            )
        annotation = np.load(self.annotation_path, allow_pickle=True).item()
        if not isinstance(annotation, dict):
            raise ValueError("CALVIN auto_lang_ann.npy must contain a dictionary.")
        try:
            spans = annotation["info"]["indx"]
            annotations = annotation["language"]["ann"]
            tasks = annotation["language"]["task"]
        except KeyError as exc:
            raise ValueError(f"CALVIN language annotation schema is missing {exc}.") from exc
        if not (len(spans) == len(annotations) == len(tasks)):
            raise ValueError("CALVIN annotation spans, language, and task IDs have different lengths.")
        self.segments = []
        previous_start = -1
        for index, (span, language, task) in enumerate(zip(spans, annotations, tasks)):
            values = np.asarray(span).reshape(-1)
            if len(values) != 2:
                raise ValueError(f"CALVIN annotation span {index} is not [start,end].")
            start, end = (int(value) for value in values)
            if start < 0 or end < start or start < previous_start:
                raise ValueError("CALVIN annotation spans must be sorted, non-negative, and inclusive.")
            previous_start = start
            text = _normalize_text(language)
            skill_id, leading_verb, directive = label_directive(text)
            self.segments.append(
                {
                    "segment_index": index,
                    "start_frame": start,
                    "end_frame": end,
                    "length": end - start + 1,
                    "language": text,
                    "task": _normalize_text(task),
                    "skill_id": int(skill_id),
                    "leading_verb": leading_verb,
                    "directive": directive,
                }
            )
        files = sorted(self.root.glob("episode_*.npz"))
        if not files:
            raise FileNotFoundError(f"No official CALVIN episode_*.npz frames under {self.root}")
        first_match = _EPISODE_FILE.match(files[0].name)
        if first_match is None:
            raise ValueError(f"Unsupported CALVIN frame filename: {files[0].name}")
        self._digits = len(first_match.group("index"))
        self._prefix = first_match.group("prefix")
        self._suffix = first_match.group("suffix")
        indices = []
        fingerprint = hashlib.sha256()
        fingerprint.update(self.annotation_path.read_bytes())
        for path in files:
            match = _EPISODE_FILE.match(path.name)
            if match is None or len(match.group("index")) != self._digits:
                raise ValueError(f"Inconsistent CALVIN frame filename: {path.name}")
            frame_index = int(match.group("index"))
            indices.append(frame_index)
            stat = path.stat()
            fingerprint.update(str(path.relative_to(self.root)).encode("utf-8"))
            fingerprint.update(str(stat.st_size).encode("ascii"))
        if indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("CALVIN episode frame files are not contiguous.")
        self.first_frame = indices[0]
        self.last_frame = indices[-1]
        for segment in self.segments:
            if segment["start_frame"] < self.first_frame or segment["end_frame"] > self.last_frame:
                raise ValueError(
                    f"CALVIN segment {segment['segment_index']} references unavailable frames."
                )
        self.dataset_fingerprint = fingerprint.hexdigest()
        self.annotation_fingerprint = _sha256(self.annotation_path)

    def frame_path(self, frame_index: int) -> Path:
        frame_index = int(frame_index)
        if not self.first_frame <= frame_index <= self.last_frame:
            raise IndexError(f"CALVIN frame {frame_index} is outside the training split.")
        path = self.root / f"{self._prefix}{frame_index:0{self._digits}d}{self._suffix}"
        if not path.is_file():
            raise FileNotFoundError(f"CALVIN frame is missing: {path}")
        return path

    def source_file_key(self, segment: dict[str, Any]) -> str:
        return (
            f"{self.frame_path(segment['start_frame'])}:"
            f"{self.frame_path(segment['end_frame']).name}"
        )

    def load_frame(self, frame_index: int) -> dict[str, Any]:
        np = _require_numpy()
        path = self.frame_path(frame_index)
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(set(self.required_frame_keys) - set(payload.files))
            if missing:
                raise KeyError(f"CALVIN frame {path.name} is missing keys: {missing}")
            output = {key: np.array(payload[key], copy=True) for key in self.required_frame_keys}
        if output["rgb_static"].dtype != np.uint8 or output["rgb_static"].shape != (200, 200, 3):
            raise ValueError("CALVIN rgb_static must be uint8 [200,200,3].")
        if output["rgb_gripper"].dtype != np.uint8 or output["rgb_gripper"].shape != (84, 84, 3):
            raise ValueError("CALVIN rgb_gripper must be uint8 [84,84,3].")
        if output["rel_actions"].shape != (7,) or not np.isfinite(output["rel_actions"]).all():
            raise ValueError("CALVIN rel_actions must be finite [7].")
        if output["robot_obs"].shape != (15,) or not np.isfinite(output["robot_obs"]).all():
            raise ValueError("CALVIN robot_obs must be finite [15].")
        if float(output["rel_actions"][6]) not in {-1.0, 1.0}:
            raise ValueError("CALVIN relative gripper must use official close=-1/open=1.")
        return output

    def load_action(self, frame_index: int):
        """Load only the compressed action member for the statistics pass."""

        np = _require_numpy()
        path = self.frame_path(frame_index)
        with np.load(path, allow_pickle=False) as payload:
            if "rel_actions" not in payload.files:
                raise KeyError(f"CALVIN frame {path.name} is missing rel_actions.")
            action = np.asarray(payload["rel_actions"], dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError("CALVIN rel_actions must be finite [7].")
        if float(action[6]) not in {-1.0, 1.0}:
            raise ValueError("CALVIN relative gripper must use official close=-1/open=1.")
        return action

    @property
    def valid_segments(self) -> list[dict[str, Any]]:
        return [
            segment
            for segment in self.segments
            if int(segment["length"]) >= self.min_segment_length
        ]

    def iter_segment_records(self, limit: int | None = None) -> Iterable[dict[str, Any]]:
        """Yield annotation metadata without opening any frame NPZ files."""

        if limit is not None and int(limit) < 1:
            raise ValueError("CALVIN segment limit must be positive.")
        for index, segment in enumerate(self.valid_segments):
            if limit is not None and index >= int(limit):
                break
            yield dict(segment)

    @staticmethod
    def segment_episode_id(segment: dict[str, Any]) -> str:
        return (
            f"calvin_abc:{int(segment['segment_index']):06d}:"
            f"{int(segment['start_frame'])}-{int(segment['end_frame'])}"
        )

    def load_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        """Materialize one previously validated language segment."""

        np = _require_numpy()
        index = int(segment["segment_index"])
        if index < 0 or index >= len(self.segments) or self.segments[index] != segment:
            raise ValueError("CALVIN segment record does not belong to this audited dataset.")
        frames = [
            self.load_frame(frame_index)
            for frame_index in range(
                int(segment["start_frame"]), int(segment["end_frame"]) + 1
            )
        ]
        return {
            **segment,
            "episode_id": self.segment_episode_id(segment),
            "rgb_static": np.stack([frame["rgb_static"] for frame in frames]),
            "rgb_gripper": np.stack([frame["rgb_gripper"] for frame in frames]),
            "robot_obs": np.stack([frame["robot_obs"] for frame in frames]).astype(
                np.float32, copy=False
            ),
            "rel_actions": np.stack([frame["rel_actions"] for frame in frames]).astype(
                np.float32, copy=False
            ),
            "skill_ids": np.full(
                int(segment["length"]), int(segment["skill_id"]), dtype=np.int8
            ),
        }

    def iter_segments(self, limit: int | None = None) -> Iterable[dict[str, Any]]:
        for segment in self.iter_segment_records(limit=limit):
            yield self.load_segment(segment)

    def action_statistics(self, limit_segments: int | None = None) -> dict[str, Any]:
        np = _require_numpy()
        segments = self.valid_segments[
            : None if limit_segments is None else int(limit_segments)
        ]
        frame_indices = sorted(
            {
                frame
                for segment in segments
                for frame in range(
                    int(segment["start_frame"]), int(segment["end_frame"]) + 1
                )
            }
        )
        if not frame_indices:
            raise ValueError("No CALVIN language frames are available for action statistics.")
        actions = np.stack([self.load_action(index) for index in frame_indices])
        motion = actions[:, :6].astype(np.float64)
        q01 = np.quantile(motion, 0.01, axis=0)
        q99 = np.quantile(motion, 0.99, axis=0)
        if not bool(np.all(q99 > q01)):
            raise ValueError("CALVIN ABC q01/q99 has a degenerate motion dimension.")
        gripper_values, gripper_counts = np.unique(actions[:, 6], return_counts=True)
        if set(float(value) for value in gripper_values) != {-1.0, 1.0}:
            raise ValueError("CALVIN ABC training actions must contain exactly gripper {-1,+1}.")
        return {
            "source": "CALVIN task_ABC_D/training language-annotated unique frames",
            "frame_count": len(frame_indices),
            "motion_q01": q01.tolist(),
            "motion_q99": q99.tolist(),
            "motion_min": motion.min(axis=0).tolist(),
            "motion_max": motion.max(axis=0).tolist(),
            "motion_mask": [True] * 6,
            "action_mode": "relative_cartesian",
            "rotation_representation": "euler_xyz",
            "gripper_open_value": 1.0,
            "gripper_closed_value": -1.0,
            "gripper_counts": {
                str(float(value)): int(count)
                for value, count in zip(gripper_values, gripper_counts)
            },
            "limit_segments": limit_segments,
        }

    def audit(self, *, limit_segments: int | None = None) -> dict[str, Any]:
        np = _require_numpy()
        segments = self.valid_segments[
            : None if limit_segments is None else int(limit_segments)
        ]
        skill_counts: Counter[int] = Counter()
        task_counts: Counter[str] = Counter()
        verb_counts: Counter[str] = Counter()
        for segment in segments:
            skill_counts[int(segment["skill_id"])] += int(segment["length"])
            task_counts[str(segment["task"])] += 1
            verb_counts[str(segment["leading_verb"])] += 1
        transitions = sum(int(segment["length"]) for segment in segments)
        valid_windows = sum(
            max(0, int(segment["length"]) - self.window_max_offset)
            for segment in segments
        )
        statistics = self.action_statistics(limit_segments=limit_segments)
        named_counts = {
            (SKILL_NAMES[index] if index >= 0 else "unknown"): int(skill_counts[index])
            for index in range(-1, len(SKILL_NAMES))
        }
        motor_counts = np.asarray([skill_counts[index] for index in range(6)], dtype=np.float64)
        if bool((motor_counts > 0).all()):
            weights = 1.0 / np.sqrt(motor_counts)
            weights = weights / weights.mean()
            class_weights = weights.tolist() + [1.0]
        else:
            class_weights = [1.0] * 7
        report = {
            "format": "mowe_calvin_abc_training_audit_v1",
            "dataset_contract": CALVIN_DATASET_CONTRACT,
            "root": str(self.root),
            "official_repo_commit": self.official_repo_commit,
            "dataset_fingerprint_sha256": self.dataset_fingerprint,
            "annotation_fingerprint_sha256": self.annotation_fingerprint,
            "annotation_segments_total": len(self.segments),
            "segments": len(segments),
            "dropped_short_segments": sum(
                int(segment["length"]) < self.min_segment_length for segment in self.segments
            ),
            "transitions": transitions,
            "window_max_offset": self.window_max_offset,
            "valid_windows": valid_windows,
            f"valid_windows_h{self.window_max_offset}": valid_windows,
            "label_version": LABEL_VERSION,
            "label_counts": named_counts,
            "unknown_ratio": named_counts["unknown"] / max(transitions, 1),
            "all_motor_classes_present": all(skill_counts[index] > 0 for index in range(6)),
            "leading_verb_counts": dict(sorted(verb_counts.items())),
            "task_segment_counts": dict(sorted(task_counts.items())),
            "action_statistics": statistics,
            "class_weights_inverse_sqrt": class_weights,
            "limit_segments": limit_segments,
            "train_eval_isolation": {
                "accepted_root": "task_ABC_D/training only",
                "evaluation_environment": "D",
                "evaluation_data_read": False,
            },
        }
        report["passed"] = (
            transitions > 0
            and valid_windows > 0
            and report["unknown_ratio"] < 0.25
            and report["all_motor_classes_present"]
            and statistics["frame_count"] > 0
        )
        return report

    def skill_config(self, audit_report: dict[str, Any], *, audit_path: str) -> dict[str, Any]:
        return {
            "format": "mowe_skill_experts_v1",
            "label_version": LABEL_VERSION,
            "source_path": str(self.annotation_path),
            "assume_sidecar_timestep_aligned": True,
            "join_key": "CALVIN auto_lang_ann.info.indx inclusive segment",
            "shuffle_files_before_join": False,
            "num_parallel_reads": 1,
            "boundary_label_policy": "direct_per_timestep_no_extra_mask",
            "unknown_label": UNKNOWN_LABEL,
            "null_route": SKILL_TO_ID["null_finish"],
            "skills": SKILL_TO_ID,
            "audit": {
                "report": str(audit_path),
                "dataset_manifest_fingerprint_sha256": self.dataset_fingerprint,
                "sidecar_fingerprint_sha256": self.annotation_fingerprint,
                "episodes": int(audit_report["segments"]),
                "transitions": int(audit_report["transitions"]),
                "valid_windows": int(audit_report["valid_windows"]),
                f"valid_windows_h{self.window_max_offset}": int(
                    audit_report["valid_windows"]
                ),
                "exact_episode_key_matches": int(audit_report["segments"]),
                "annotation_step_match_ratio": 1.0,
                "alignment_verified": False,
                "label_counts": audit_report["label_counts"],
            },
            "class_weights_inverse_sqrt": audit_report["class_weights_inverse_sqrt"],
        }


def _require_tensorflow():
    try:
        import tensorflow as tf
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to parse CALVIN RLDS TFRecord Examples."
        ) from exc
    return tf


def _iter_tfrecord_payloads(path: Path):
    """Yield record offset and payload without constructing a tf.data pipeline."""

    with path.open("rb") as handle:
        record_index = 0
        while True:
            offset = handle.tell()
            header = handle.read(12)
            if not header:
                return
            if len(header) != 12:
                raise ValueError(f"Truncated TFRecord header: {path}")
            length = struct.unpack("<Q", header[:8])[0]
            payload = handle.read(length)
            footer = handle.read(4)
            if len(payload) != length or len(footer) != 4:
                raise ValueError(f"Truncated TFRecord payload: {path} record {record_index}")
            yield record_index, offset, length, payload, header + payload + footer
            record_index += 1


class CalvinRLDSEpisodeDataset:
    """Strict reader for the 512-shard Open-X CALVIN ABC train RLDS export."""

    dataset_name = "calvin_abc_rlds"

    def __init__(
        self,
        root: str | Path,
        *,
        min_segment_length: int = 17,
        expected_shards: int = 512,
        official_repo_commit: str = "fa03f01f19c65920e18cf37398a9ce859274af76",
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"CALVIN RLDS root not found: {self.root}")
        self.min_segment_length = int(min_segment_length)
        if self.min_segment_length < 1:
            raise ValueError("min_segment_length must be positive.")
        self.window_max_offset = self.min_segment_length - 1
        self.official_repo_commit = str(official_repo_commit)
        self.shards = sorted(self.root.glob("calvin_abc-train.tfrecord-*-of-*"))
        if len(self.shards) != int(expected_shards):
            raise ValueError(
                f"CALVIN ABC RLDS requires {expected_shards} shards, found {len(self.shards)}."
            )
        indices = []
        declared_counts = set()
        fingerprint = hashlib.sha256()
        self._expected_checksums: dict[Path, str] = {}
        for path in self.shards:
            match = _RLDS_SHARD.match(path.name)
            if match is None:
                raise ValueError(f"Unexpected CALVIN RLDS shard name: {path.name}")
            indices.append(int(match.group("index")))
            declared_counts.add(int(match.group("count")))
            if path.stat().st_size < 16:
                raise ValueError(f"CALVIN RLDS shard is empty or truncated: {path}")
            checksum = self._metadata_checksum(path)
            if checksum is None:
                checksum = _sha256(path)
            self._expected_checksums[path] = checksum
            fingerprint.update(path.name.encode("utf-8"))
            fingerprint.update(str(path.stat().st_size).encode("ascii"))
            fingerprint.update(checksum.encode("ascii"))
        if declared_counts != {int(expected_shards)} or indices != list(
            range(int(expected_shards))
        ):
            raise ValueError("CALVIN ABC RLDS shard indices/count declaration is incomplete.")
        self.dataset_fingerprint = fingerprint.hexdigest()
        self.annotation_fingerprint = hashlib.sha256(
            f"calvin-rlds-language:{self.dataset_fingerprint}".encode("ascii")
        ).hexdigest()
        self._records: list[dict[str, Any]] | None = None
        self._audit_cache: dict[str, Any] | None = None

    def _metadata_checksum(self, shard: Path) -> str | None:
        metadata = (
            self.root
            / ".cache/huggingface/download"
            / f"{shard.name}.metadata"
        )
        if not metadata.is_file():
            return None
        lines = metadata.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2 or not re.fullmatch(r"[0-9a-f]{64}", lines[1].strip()):
            return None
        return lines[1].strip()

    def _parse_record(self, payload: bytes, *, materialize: bool) -> dict[str, Any]:
        tf = _require_tensorflow()
        np = _require_numpy()
        example = tf.train.Example.FromString(payload)
        features = example.features.feature
        required = {
            "episode_metadata/episode_id",
            "steps/action",
            "steps/language_instruction",
            "steps/metadata/episode_index",
            "steps/observation/rgb_gripper",
            "steps/observation/rgb_static",
            "steps/observation/state",
        }
        missing = sorted(required - set(features))
        if missing:
            raise KeyError(f"CALVIN RLDS record is missing fields: {missing}")
        languages = [
            _normalize_text(value)
            for value in features["steps/language_instruction"].bytes_list.value
        ]
        length = len(languages)
        if length < 1 or len(set(languages)) != 1:
            raise ValueError("Each CALVIN RLDS episode must have one repeated instruction.")
        episode_ids = list(features["episode_metadata/episode_id"].int64_list.value)
        episode_indices = list(features["steps/metadata/episode_index"].int64_list.value)
        if len(episode_ids) != 1 or len(episode_indices) != length:
            raise ValueError("CALVIN RLDS episode identity fields have invalid lengths.")
        episode_id = int(episode_ids[0])
        if any(int(value) != episode_id for value in episode_indices):
            raise ValueError("CALVIN RLDS per-step episode_index does not match episode_id.")
        actions = np.asarray(
            features["steps/action"].float_list.value, dtype=np.float32
        )
        states = np.asarray(
            features["steps/observation/state"].float_list.value, dtype=np.float32
        )
        if actions.size != length * 7 or states.size != length * 15:
            raise ValueError("CALVIN RLDS action/state arrays have invalid flattened shapes.")
        actions = actions.reshape(length, 7)
        states = states.reshape(length, 15)
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError("CALVIN RLDS action/state arrays must be finite.")
        if not np.isin(actions[:, 6], (-1.0, 1.0)).all():
            raise ValueError("CALVIN RLDS gripper actions must use {-1,+1}.")
        primary_values = features["steps/observation/rgb_static"].bytes_list.value
        wrist_values = features["steps/observation/rgb_gripper"].bytes_list.value
        if len(primary_values) != length or len(wrist_values) != length:
            raise ValueError("CALVIN RLDS camera arrays do not match episode length.")
        png_signature = b"\x89PNG\r\n\x1a\n"
        if not all(value.startswith(png_signature) for value in primary_values) or not all(
            value.startswith(png_signature) for value in wrist_values
        ):
            raise ValueError("CALVIN RLDS camera observations must be PNG encoded.")
        language = languages[0]
        skill_id, leading_verb, directive = label_calvin_instruction(language)
        output = {
            "episode_index": episode_id,
            "length": length,
            "language": language,
            "task": language,
            "skill_id": int(skill_id),
            "leading_verb": leading_verb,
            "directive": directive,
            "rel_actions": actions,
            "robot_obs": states,
        }
        if materialize:
            primary = list(primary_values)
            wrist = list(wrist_values)

            def decode(frames, expected_shape):
                from PIL import Image

                decoded = [
                    np.asarray(Image.open(io.BytesIO(value)).convert("RGB"), dtype=np.uint8)
                    for value in frames
                ]
                array = np.stack(decoded)
                if tuple(array.shape[1:]) != tuple(expected_shape):
                    raise ValueError(
                        f"CALVIN RLDS camera shape {array.shape[1:]} != {expected_shape}."
                    )
                return array

            output["rgb_static"] = decode(primary, (200, 200, 3))
            output["rgb_gripper"] = decode(wrist, (84, 84, 3))
            output["skill_ids"] = np.full(length, int(skill_id), dtype=np.int8)
        return output

    def _ensure_records(self) -> list[dict[str, Any]]:
        if self._records is not None:
            return self._records
        records = []
        source_episode_counts: Counter[int] = Counter()
        shard_checksums_verified = True
        for shard in self.shards:
            digest = hashlib.sha256()
            for record_index, offset, length, payload, encoded_record in _iter_tfrecord_payloads(shard):
                digest.update(encoded_record)
                record = self._parse_record(payload, materialize=False)
                episode_id = int(record["episode_index"])
                source_episode_counts[episode_id] += 1
                record.update(
                    {
                        "segment_index": len(records),
                        "shard": shard.name,
                        "record_index": record_index,
                        "record_offset": offset,
                        "record_length": length,
                    }
                )
                record.pop("robot_obs")
                records.append(record)
            shard_checksums_verified &= (
                digest.hexdigest() == self._expected_checksums[shard]
            )
        if not records:
            raise ValueError("CALVIN RLDS contains no episodes.")
        self._records = records
        self._source_episode_counts = source_episode_counts
        self._shard_checksums_verified = bool(shard_checksums_verified)
        return records

    @property
    def valid_segments(self) -> list[dict[str, Any]]:
        return [
            record
            for record in self._ensure_records()
            if int(record["length"]) >= self.min_segment_length
        ]

    def iter_segment_records(self, limit: int | None = None) -> Iterable[dict[str, Any]]:
        if limit is not None and int(limit) < 1:
            raise ValueError("CALVIN segment limit must be positive.")
        records = self.valid_segments[: None if limit is None else int(limit)]
        for record in records:
            yield dict(record)

    @staticmethod
    def segment_episode_id(segment: dict[str, Any]) -> str:
        return (
            f"calvin_abc_rlds:{int(segment['segment_index']):06d}:"
            f"source{int(segment['episode_index']):06d}"
        )

    def source_file_key(self, segment: dict[str, Any]) -> str:
        return f"{segment['shard']}:{int(segment['record_index'])}"

    def _load_record(self, segment: dict[str, Any], *, materialize: bool) -> dict[str, Any]:
        shard = self.root / str(segment["shard"])
        with shard.open("rb") as handle:
            handle.seek(int(segment["record_offset"]))
            header = handle.read(12)
            length = struct.unpack("<Q", header[:8])[0]
            if length != int(segment["record_length"]):
                raise ValueError("CALVIN RLDS record length changed after audit.")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError("CALVIN RLDS record is truncated after audit.")
        loaded = self._parse_record(payload, materialize=materialize)
        for key in ("episode_index", "length", "language"):
            if loaded[key] != segment[key]:
                raise ValueError(f"CALVIN RLDS record changed after audit: {key}")
        output = {
            **segment,
            **loaded,
            "episode_id": self.segment_episode_id(segment),
        }
        return output

    def load_segment(self, segment: dict[str, Any]) -> dict[str, Any]:
        return self._load_record(segment, materialize=True)

    def iter_segments(self, limit: int | None = None) -> Iterable[dict[str, Any]]:
        for record in self.iter_segment_records(limit=limit):
            yield self.load_segment(record)

    def action_statistics(self, limit_segments: int | None = None) -> dict[str, Any]:
        np = _require_numpy()
        records = list(self.iter_segment_records(limit=limit_segments))
        actions = [record["rel_actions"] for record in records]
        if not actions:
            raise ValueError("No CALVIN RLDS actions are available.")
        actions = np.concatenate(actions, axis=0)
        motion = actions[:, :6].astype(np.float64)
        q01 = np.quantile(motion, 0.01, axis=0)
        q99 = np.quantile(motion, 0.99, axis=0)
        if not bool(np.all(q99 > q01)):
            raise ValueError("CALVIN ABC q01/q99 has a degenerate motion dimension.")
        values, counts = np.unique(actions[:, 6], return_counts=True)
        if set(float(value) for value in values) != {-1.0, 1.0}:
            raise ValueError("CALVIN ABC train must contain both gripper {-1,+1} values.")
        return {
            "source": "CALVIN ABC RLDS train episodes only",
            "frame_count": int(len(actions)),
            "motion_q01": q01.tolist(),
            "motion_q99": q99.tolist(),
            "motion_min": motion.min(axis=0).tolist(),
            "motion_max": motion.max(axis=0).tolist(),
            "motion_mask": [True] * 6,
            "action_mode": "relative_cartesian",
            "rotation_representation": "euler_xyz",
            "gripper_open_value": 1.0,
            "gripper_closed_value": -1.0,
            "gripper_counts": {
                str(float(value)): int(count) for value, count in zip(values, counts)
            },
            "limit_segments": limit_segments,
        }

    def audit(self, *, limit_segments: int | None = None) -> dict[str, Any]:
        np = _require_numpy()
        records = list(self.iter_segment_records(limit=limit_segments))
        skill_counts: Counter[int] = Counter()
        verb_counts: Counter[str] = Counter()
        for record in records:
            skill_counts[int(record["skill_id"])] += int(record["length"])
            verb_counts[str(record["leading_verb"])] += 1
        transitions = sum(int(record["length"]) for record in records)
        valid_windows = sum(
            max(0, int(record["length"]) - self.window_max_offset)
            for record in records
        )
        statistics = self.action_statistics(limit_segments=limit_segments)
        named_counts = {
            (SKILL_NAMES[index] if index >= 0 else "unknown"): int(skill_counts[index])
            for index in range(-1, len(SKILL_NAMES))
        }
        motor_counts = np.asarray([skill_counts[index] for index in range(6)], dtype=np.float64)
        if bool((motor_counts > 0).all()):
            weights = 1.0 / np.sqrt(motor_counts)
            weights = weights / weights.mean()
            class_weights = weights.tolist() + [1.0]
        else:
            class_weights = [1.0] * 7
        report = {
            "format": "mowe_calvin_abc_training_audit_v2",
            "dataset_contract": CALVIN_RLDS_DATASET_CONTRACT,
            "root": str(self.root),
            "official_repo_commit": self.official_repo_commit,
            "dataset_fingerprint_sha256": self.dataset_fingerprint,
            "annotation_fingerprint_sha256": self.annotation_fingerprint,
            "shards_expected": len(self.shards),
            "shards_found": len(self.shards),
            "shard_checksums_verified": bool(self._shard_checksums_verified),
            "segments": len(records),
            "unique_source_episode_ids": len(self._source_episode_counts),
            "reused_source_episode_ids": sum(
                count > 1 for count in self._source_episode_counts.values()
            ),
            "dropped_short_segments": len(self._ensure_records()) - len(self.valid_segments),
            "transitions": transitions,
            "window_max_offset": self.window_max_offset,
            "valid_windows": valid_windows,
            f"valid_windows_h{self.window_max_offset}": valid_windows,
            "label_version": CALVIN_RLDS_LABEL_VERSION,
            "label_counts": named_counts,
            "unknown_ratio": named_counts["unknown"] / max(transitions, 1),
            "all_motor_classes_present": all(skill_counts[index] > 0 for index in range(6)),
            "leading_verb_counts": dict(sorted(verb_counts.items())),
            "action_statistics": statistics,
            "class_weights_inverse_sqrt": class_weights,
            "limit_segments": limit_segments,
            "train_eval_isolation": {
                "accepted_root": "calvin_abc train TFRecord shards only",
                "evaluation_environment": "D",
                "evaluation_data_read": False,
            },
        }
        report["passed"] = (
            transitions > 0
            and valid_windows > 0
            and report["unknown_ratio"] < 0.25
            and report["all_motor_classes_present"]
            and statistics["frame_count"] == transitions
            and report["shard_checksums_verified"]
        )
        self._audit_cache = report
        return report

    def skill_config(self, audit_report: dict[str, Any], *, audit_path: str) -> dict[str, Any]:
        return {
            "format": "mowe_skill_experts_v1",
            "label_version": CALVIN_RLDS_LABEL_VERSION,
            "source_path": str(self.root),
            "assume_sidecar_timestep_aligned": True,
            "join_key": (
                "RLDS shard + record_index + episode_metadata.episode_id + timestep"
            ),
            "shuffle_files_before_join": False,
            "num_parallel_reads": 1,
            "boundary_label_policy": "direct_per_timestep_no_extra_mask",
            "unknown_label": UNKNOWN_LABEL,
            "null_route": SKILL_TO_ID["null_finish"],
            "skills": SKILL_TO_ID,
            "audit": {
                "report": str(audit_path),
                "dataset_manifest_fingerprint_sha256": self.dataset_fingerprint,
                "sidecar_fingerprint_sha256": self.annotation_fingerprint,
                "episodes": int(audit_report["segments"]),
                "transitions": int(audit_report["transitions"]),
                "valid_windows": int(audit_report["valid_windows"]),
                f"valid_windows_h{self.window_max_offset}": int(
                    audit_report["valid_windows"]
                ),
                "exact_episode_key_matches": int(audit_report["segments"]),
                "annotation_step_match_ratio": 1.0,
                "alignment_verified": False,
                "label_counts": audit_report["label_counts"],
            },
            "class_weights_inverse_sqrt": audit_report["class_weights_inverse_sqrt"],
        }


def resolve_calvin_training_dataset(
    root: str | Path,
    *,
    dataset_format: str = "auto",
    min_segment_length: int = 17,
    official_repo_commit: str = "fa03f01f19c65920e18cf37398a9ce859274af76",
):
    root = Path(root).expanduser().resolve()
    if dataset_format not in {"auto", "official_npz", "rlds"}:
        raise ValueError(f"Unsupported CALVIN dataset format: {dataset_format}")
    has_rlds = bool(list(root.glob("calvin_abc-train.tfrecord-*-of-*")))
    if dataset_format == "rlds" or (dataset_format == "auto" and has_rlds):
        return CalvinRLDSEpisodeDataset(
            root,
            min_segment_length=min_segment_length,
            official_repo_commit=official_repo_commit,
        )
    return CalvinLanguageSegmentDataset(
        root,
        min_segment_length=min_segment_length,
        official_repo_commit=official_repo_commit,
    )
