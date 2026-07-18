#!/usr/bin/env python3
"""Audit the real LIBERO RLDS/skill-sidecar contract without loading OpenVLA."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.data import (
    ExpertSkillSidecar,
    LIBERO_SEQUENCE_DATASETS,
    SKILL_NAMES,
    rlds_manifest_fingerprint,
)


def _label_counts(labels: list[int]) -> dict[str, int]:
    counter = Counter(labels)
    return {
        **{name: int(counter[index]) for index, name in enumerate(SKILL_NAMES)},
        "unknown": int(counter[-1]),
    }


def _counter_label_counts(counter: Counter) -> dict[str, int]:
    return {
        **{name: int(counter[index]) for index, name in enumerate(SKILL_NAMES)},
        "unknown": int(counter[-1]),
    }


def audit_suite(dataset_root: Path, dataset_name: str, sidecar: ExpertSkillSidecar, max_horizon: int):
    try:
        import dlimp as dl
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Real RLDS audit requires the OpenVLA TensorFlow environment "
            "(tensorflow-datasets and dlimp)."
        ) from exc

    builder = tfds.builder(dataset_name, data_dir=str(dataset_root))
    dataset = dl.DLataset.from_rlds(
        builder,
        split="train",
        shuffle=False,
        num_parallel_reads=16,
    )
    episode_ids = []
    episode_lengths = []
    motions = []
    grippers = []
    joined_labels: list[int] = []
    window_position_counts = [Counter() for _ in range(max_horizon)]
    window_transition_counts = Counter()
    annotation_steps = 0
    exact_episode_key_matches = 0
    mismatched_episode_keys = []

    for trajectory in dataset.as_numpy_iterator():
        trajectory_id = int(trajectory["_traj_index"][0])
        file_path = trajectory["traj_metadata"]["episode_metadata"]["file_path"][0].decode("utf-8")
        action = np.asarray(trajectory["action"], dtype=np.float32)
        length = int(action.shape[0])
        record = sidecar.episode(dataset_name, trajectory_id, expected_length=length)
        expected_key = f"{file_path}_{trajectory_id}"
        if record["episode_key"] == expected_key:
            exact_episode_key_matches += 1
        else:
            mismatched_episode_keys.append(
                {
                    "trajectory_id": trajectory_id,
                    "expected": expected_key,
                    "sidecar": record["episode_key"],
                }
            )

        labels = [int(value) for value in record["labels"]]
        present = [bool(value) for value in record["present"]]
        episode_ids.append(trajectory_id)
        episode_lengths.append(length)
        motions.append(action[:, :6])
        # OpenVLA-OFT LIBERO canonicalization: raw -1=open/+1=close
        # -> clip to 0/1, then invert so 1=open and 0=close.
        grippers.append(1.0 - np.clip(action[:, 6], 0.0, 1.0))
        joined_labels.extend(labels)
        annotation_steps += sum(present)
        for window_start in range(max(0, length - max_horizon)):
            window_labels = labels[window_start : window_start + max_horizon]
            for position, label in enumerate(window_labels):
                window_position_counts[position][label] += 1
            for source, target in zip(window_labels[:-1], window_labels[1:]):
                window_transition_counts[(source, target)] += 1

    motion = np.concatenate(motions, axis=0)
    gripper = np.concatenate(grippers, axis=0)
    expected_ids = list(range(len(episode_ids)))
    valid_windows = sum(max(0, length - max_horizon) for length in episode_lengths)
    transitions = int(sum(episode_lengths))
    return {
        "dataset_name": dataset_name,
        "episodes": len(episode_ids),
        "trajectory_ids_contiguous": episode_ids == expected_ids,
        "episode_length": {
            "min": min(episode_lengths),
            "max": max(episode_lengths),
            "mean": float(np.mean(episode_lengths)),
        },
        "transitions": transitions,
        "max_horizon": int(max_horizon),
        "valid_windows": int(valid_windows),
        "exact_episode_key_matches": exact_episode_key_matches,
        "exact_episode_key_match_ratio": exact_episode_key_matches / max(len(episode_ids), 1),
        "annotation_steps": annotation_steps,
        "annotation_step_match_ratio": annotation_steps / max(transitions, 1),
        "parsed_skill_counts": _label_counts(joined_labels),
        "parsed_unknown_ratio": joined_labels.count(-1) / max(len(joined_labels), 1),
        "window_position_skill_counts": {
            str(position): _counter_label_counts(counter)
            for position, counter in enumerate(window_position_counts)
        },
        "window_transition_matrix": {
            source_name: {
                target_name: int(window_transition_counts[(source_id, target_id)])
                for target_id, target_name in [(-1, "unknown"), *list(enumerate(SKILL_NAMES))]
            }
            for source_id, source_name in [(-1, "unknown"), *list(enumerate(SKILL_NAMES))]
        },
        "motion_raw_q01": np.quantile(motion, 0.01, axis=0).tolist(),
        "motion_raw_q99": np.quantile(motion, 0.99, axis=0).tolist(),
        "canonical_gripper_counts": {
            "closed_0": int(np.count_nonzero(gripper == 0.0)),
            "open_1": int(np.count_nonzero(gripper == 1.0)),
            "non_binary": int(np.count_nonzero((gripper != 0.0) & (gripper != 1.0))),
        },
        "mismatched_episode_keys_sample": mismatched_episode_keys[:5],
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="datasets/modified_libero_rlds")
    parser.add_argument("--skill-sidecar", default="datasets/libero_cot_rlds/cot_file.json")
    parser.add_argument("--dataset-name", action="append", choices=LIBERO_SEQUENCE_DATASETS)
    parser.add_argument("--max-horizon", type=int, default=16)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).resolve()
    names = list(args.dataset_name or LIBERO_SEQUENCE_DATASETS)
    sidecar = ExpertSkillSidecar(args.skill_sidecar)
    suites = [audit_suite(root, name, sidecar, args.max_horizon) for name in names]
    total_episodes = sum(item["episodes"] for item in suites)
    total_transitions = sum(item["transitions"] for item in suites)
    total_annotations = sum(item["annotation_steps"] for item in suites)
    report = {
        "format": "flow_wam_rlds_audit_v1",
        "dataset_root": str(root),
        "dataset_manifest_fingerprint_sha256": rlds_manifest_fingerprint(root, names),
        "sidecar_fingerprint_sha256": sidecar.metadata["fingerprint_sha256"],
        "join_key": "episode_metadata.file_path + dlimp._traj_index + timestep",
        "shuffle_files_before_join": False,
        "num_parallel_reads": 16,
        "alignment_verified": False,
        "alignment_note": (
            "Exact key coverage is verified. Semantic frame-to-annotation alignment remains the configured v1 assumption."
        ),
        "totals": {
            "episodes": total_episodes,
            "transitions": total_transitions,
            "max_horizon": int(args.max_horizon),
            "valid_windows": sum(item["valid_windows"] for item in suites),
            "exact_episode_key_matches": sum(item["exact_episode_key_matches"] for item in suites),
            "annotation_steps": total_annotations,
            "annotation_step_match_ratio": total_annotations / max(total_transitions, 1),
        },
        "suites": suites,
    }
    if any(not item["trajectory_ids_contiguous"] for item in suites):
        raise RuntimeError("RLDS global trajectory ids are not contiguous under deterministic file order.")
    if any(item["exact_episode_key_match_ratio"] != 1.0 for item in suites):
        raise RuntimeError("At least one RLDS episode failed the exact sidecar episode-key join.")
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
