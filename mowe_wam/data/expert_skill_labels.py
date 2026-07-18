"""Auditable leading-verb skill labels for the training-only CoT sidecar."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path


SKILL_NAMES = (
    "pick_grasp",
    "place_release",
    "move_transport",
    "open_close",
    "turn_rotate",
    "push_pull",
    "null_finish",
)
SKILL_TO_ID = {name: index for index, name in enumerate(SKILL_NAMES)}
UNKNOWN_LABEL = -1
LABEL_VERSION = "cot_final_directive_leading_verb_v1"

VERB_TO_SKILL = {
    **{verb: "pick_grasp" for verb in ("pick", "grasp", "grab", "lift")},
    **{verb: "place_release" for verb in ("place", "put", "release", "set", "stack")},
    **{
        verb: "move_transport"
        for verb in ("move", "carry", "bring", "position", "align", "approach")
    },
    **{verb: "open_close" for verb in ("open", "close")},
    **{verb: "turn_rotate" for verb in ("turn", "rotate")},
    **{verb: "push_pull" for verb in ("push", "pull")},
    **{verb: "null_finish" for verb in ("finish", "stop", "done", "check", "hold")},
}

_EPISODE_KEY = re.compile(r"^(?P<file>.*\.hdf5)_(?P<episode>\d+)_(?P<step>\d+)$")
_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_DISCOURSE = {"finally", "then", "now", "next", "carefully", "slowly", "gently"}


def final_directive(text: str) -> str:
    cleaned = _TAG.sub(" ", str(text)).strip()
    sentences = [part.strip() for part in re.split(r"[.!?]+", cleaned) if part.strip()]
    return sentences[-1] if sentences else cleaned


def leading_verb(text: str) -> str:
    words = _WORD.findall(final_directive(text).lower())
    while words and words[0] in _DISCOURSE:
        words.pop(0)
    return words[0] if words else ""


def label_directive(text: str) -> tuple[int, str, str]:
    verb = leading_verb(text)
    skill = VERB_TO_SKILL.get(verb)
    return (SKILL_TO_ID[skill] if skill is not None else UNKNOWN_LABEL, verb, final_directive(text))


def _dataset_from_key(key: str) -> str | None:
    aliases = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }
    for segment, dataset in aliases.items():
        if f"/{segment}/" in key or f"/{segment.upper()}/" in key:
            return dataset
    return None


class ExpertSkillSidecar:
    """Index sidecar labels by suite, global RLDS trajectory id, and timestep."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Skill annotation sidecar not found: {self.path}")
        raw_bytes = self.path.read_bytes()
        payload = json.loads(raw_bytes)
        if not isinstance(payload, dict):
            raise ValueError("cot_file.json must be a key-to-annotation mapping.")
        grouped: dict[str, OrderedDict[str, dict[int, tuple[int, str]]]] = {}
        unmatched = 0
        for key, annotation in payload.items():
            dataset = _dataset_from_key(str(key))
            match = _EPISODE_KEY.match(str(key))
            if dataset is None or match is None:
                unmatched += 1
                continue
            episode_key = f"{match.group('file')}_{match.group('episode')}"
            step = int(match.group("step"))
            label, _, directive = label_directive(str(annotation))
            episodes = grouped.setdefault(dataset, OrderedDict())
            episodes.setdefault(episode_key, {})[step] = (label, directive)

        self._episodes: dict[str, list[dict[str, object]]] = {}
        self._episodes_by_index: dict[str, dict[int, dict[str, object]]] = {}
        for dataset, episodes in grouped.items():
            values = []
            indexed = {}
            for episode_key, step_map in episodes.items():
                match = _EPISODE_KEY.match(f"{episode_key}_0")
                if match is None:
                    raise ValueError(f"Invalid indexed sidecar episode key: {episode_key}")
                episode_index = int(match.group("episode"))
                length = max(step_map, default=-1) + 1
                labels = [UNKNOWN_LABEL] * length
                directives = [""] * length
                present = [False] * length
                for step, (label, directive) in step_map.items():
                    labels[step] = label
                    directives[step] = directive
                    present[step] = True
                record = {
                    "episode_key": episode_key,
                    "episode_index": episode_index,
                    "labels": labels,
                    "directives": directives,
                    "present": present,
                }
                if episode_index in indexed:
                    raise ValueError(
                        f"Duplicate global trajectory id {episode_index} in sidecar suite {dataset}."
                    )
                indexed[episode_index] = record
                values.append(record)
            self._episodes[dataset] = values
            self._episodes_by_index[dataset] = indexed
        self.metadata = {
            "format": "expert_skill_sidecar_v1",
            "label_version": LABEL_VERSION,
            "fingerprint_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "assume_timestep_aligned": True,
            "alignment_verified": False,
            "join_key": "episode_metadata.file_path + global_trajectory_index + timestep",
            "unmatched_records": unmatched,
            "episode_counts": {name: len(values) for name, values in self._episodes.items()},
        }

    def episode(self, dataset_name: str, episode_index: int, expected_length: int | None = None):
        record = self._episodes_by_index.get(str(dataset_name), {}).get(int(episode_index))
        if record is None:
            length = int(expected_length or 0)
            return {
                "episode_key": None,
                "labels": [UNKNOWN_LABEL] * length,
                "directives": [""] * length,
                "present": [False] * length,
                "sources": ["unknown"] * length,
            }
        labels = list(record["labels"])
        directives = list(record["directives"])
        present = list(record["present"])
        if expected_length is not None:
            missing = max(0, int(expected_length) - len(labels))
            labels.extend([UNKNOWN_LABEL] * missing)
            directives.extend([""] * missing)
            present.extend([False] * missing)
            labels = labels[: int(expected_length)]
            directives = directives[: int(expected_length)]
            present = present[: int(expected_length)]
        return {
            "episode_key": record["episode_key"],
            "labels": labels,
            "directives": directives,
            "present": present,
            "sources": [
                "raw_annotation" if label >= 0 else ("raw_annotation_unmapped" if exists else "unknown")
                for label, exists in zip(labels, present)
            ],
        }

    def audit(self) -> dict[str, object]:
        counts = Counter()
        positions = Counter()
        transitions = Counter()
        suite_counts: dict[str, Counter] = {}
        for dataset, episodes in self._episodes.items():
            per_suite = suite_counts.setdefault(dataset, Counter())
            for record in episodes:
                previous = None
                for position, label in enumerate(record["labels"]):
                    counts[int(label)] += 1
                    per_suite[int(label)] += 1
                    positions[(position % 8, int(label))] += 1
                    if previous is not None:
                        transitions[(int(previous), int(label))] += 1
                    previous = int(label)
        label_ids = range(-1, len(SKILL_NAMES))

        def label_name(label):
            return SKILL_NAMES[label] if label >= 0 else "unknown"
        return {
            **self.metadata,
            "label_counts": {
                label_name(label): int(count)
                for label, count in sorted(counts.items())
            },
            "unknown_ratio": float(counts[UNKNOWN_LABEL] / max(sum(counts.values()), 1)),
            "all_motor_classes_present": all(counts[index] > 0 for index in range(6)),
            "suite_label_counts": {
                dataset: {label_name(label): int(counter[label]) for label in label_ids}
                for dataset, counter in suite_counts.items()
            },
            "position_mod_8_counts": {
                str(position): {
                    label_name(label): int(positions[(position, label)]) for label in label_ids
                }
                for position in range(8)
            },
            "transition_matrix": {
                label_name(source): {
                    label_name(target): int(transitions[(source, target)]) for target in label_ids
                }
                for source in label_ids
            },
        }
