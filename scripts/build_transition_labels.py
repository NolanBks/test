#!/usr/bin/env python3
"""Build trajectory-level predicate, future-state, and event label caches.

Real mode expects JSONL where each row contains ``episode_id``, ``trajectory``
(a list of simulator/trajectory state dictionaries), and optional ``task_meta``.
This keeps simulator-specific extraction outside the training data loader.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.memory import build_memory_snapshots
from mowe_wam.predicates.labeler import build_mock_trajectory, label_trajectory
from mowe_wam.predicates.schema import PREDICATE_NAMES, predicate_dict_to_vector


def _phase_target(vector: list[float]) -> int:
    values = {name: vector[idx] for idx, name in enumerate(PREDICATE_NAMES)}
    if values["needs_recovery"] >= 0.5 or values["failure_risk"] >= 0.7:
        return 4
    if values["near_goal_region"] >= 0.7 or values["alignment_required"] >= 0.5:
        return 3
    if values["object_grasped"] >= 0.5 and values["object_moving_with_gripper"] >= 0.3:
        return 2
    if values["contact_likely"] >= 0.5:
        return 1
    return 0


def _episode_record(episode_id: str, trajectory: list[dict], task_meta: dict | None, horizon: int) -> dict:
    predicate_dicts = label_trajectory(trajectory, task_meta=task_meta)
    predicate_vectors = [predicate_dict_to_vector(item) for item in predicate_dicts]
    progress = [item["progress_score"] for item in predicate_dicts]
    risk = [item["failure_risk"] for item in predicate_dicts]
    phases = [_phase_target(vector) for vector in predicate_vectors]
    _, events = build_memory_snapshots(predicate_vectors, progress, risk, phases)
    steps = []
    for index, vector in enumerate(predicate_vectors):
        future_index = min(index + horizon, len(predicate_vectors) - 1)
        steps.append(
            {
                "step_id": index,
                "predicates": vector,
                "progress": progress[index],
                "risk": risk[index],
                "future_predicates": predicate_vectors[future_index],
                "progress_delta": progress[future_index] - progress[index],
                "future_risk": risk[future_index],
                "future_recovery": predicate_vectors[future_index][PREDICATE_NAMES.index("needs_recovery")],
                "event_target": events[index],
                "phase_target": phases[future_index],
            }
        )
    return {"episode_id": episode_id, "steps": steps}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--input", default=None, help="Trajectory JSONL exported from LIBERO/state replay.")
    parser.add_argument("--output", default="outputs/transition_labels/mock_transition_labels.jsonl")
    parser.add_argument("--prediction-horizon", type=int, default=8)
    args = parser.parse_args()
    if args.prediction_horizon < 1:
        raise SystemExit("--prediction-horizon must be positive.")
    if args.mock == bool(args.input):
        raise SystemExit("Pass exactly one of --mock or --input.")

    if args.mock:
        trajectory, task_meta = build_mock_trajectory()
        records = [_episode_record("mock_episode_0", trajectory, task_meta, args.prediction_horizon)]
    else:
        records = []
        for line in Path(args.input).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            records.append(
                _episode_record(
                    str(row["episode_id"]),
                    list(row["trajectory"]),
                    row.get("task_meta"),
                    args.prediction_horizon,
                )
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, sort_keys=True) + "\n")
    print(f"Wrote {len(records)} trajectory label records to {output}")


if __name__ == "__main__":
    main()
