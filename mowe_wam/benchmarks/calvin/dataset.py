"""Strict reader and audit for the official CALVIN ABC language training split.

The official dataset stores one ``episode_XXXXXXX.npz`` per interaction
timestep and language segments in ``lang_annotations/auto_lang_ann.npy``. This
module has no simulator or CALVIN-package dependency, allowing data gates to
run before the evaluation environment is installed.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from mowe_wam.data.expert_skill_labels import (
    LABEL_VERSION,
    SKILL_NAMES,
    SKILL_TO_ID,
    UNKNOWN_LABEL,
    label_directive,
)


CALVIN_DATASET_CONTRACT = "official_calvin_abc_language_segments_v1"
_EPISODE_FILE = re.compile(r"^(?P<prefix>episode_)(?P<index>\d+)(?P<suffix>\.npz)$")


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
        valid_windows = sum(max(0, int(segment["length"]) - 8) for segment in segments)
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
            "valid_windows_h8": valid_windows,
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
                "valid_windows_h8": int(audit_report["valid_windows_h8"]),
                "exact_episode_key_matches": int(audit_report["segments"]),
                "annotation_step_match_ratio": 1.0,
                "alignment_verified": False,
                "label_counts": audit_report["label_counts"],
            },
            "class_weights_inverse_sqrt": audit_report["class_weights_inverse_sqrt"],
        }
