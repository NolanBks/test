"""Heuristic pseudo-label generator for world predicates.

The first implementation supports mock trajectories and simple simulator-state
dictionaries. Real LIBERO bindings should be added after inspecting the exact
state keys exposed by the upstream benchmark.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

from mowe_wam.predicates.schema import PREDICATE_NAMES, validate_predicate_dict

DEFAULT_CFG = {
    "distance_norm": 0.5,
    "contact_distance": 0.06,
    "lift_height": 0.04,
    "goal_progress_epsilon": 0.01,
}


def _clip01(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def _pos(step: dict[str, Any], *names: str) -> tuple[float, ...] | None:
    for name in names:
        value = step.get(name)
        if value is not None:
            if len(value) < 3:
                warnings.warn(f"TBD: field {name!r} has fewer than 3 coordinates.", stacklevel=2)
                return None
            return tuple(float(v) for v in value[:3])
    return None


def _distance(a: tuple[float, ...] | None, b: tuple[float, ...] | None) -> float | None:
    if a is None or b is None:
        return None
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _inverse_distance_score(distance: float | None, norm: float) -> float:
    if distance is None:
        return 0.0
    return _clip01(1.0 - distance / max(norm, 1e-6))


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def _norm(a: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _direction_alignment(
    start_a: tuple[float, ...] | None,
    end_a: tuple[float, ...] | None,
    start_b: tuple[float, ...] | None,
    end_b: tuple[float, ...] | None,
) -> float:
    if None in (start_a, end_a, start_b, end_b):
        return 0.0
    assert start_a is not None and end_a is not None and start_b is not None and end_b is not None
    da = _sub(end_a, start_a)
    db = _sub(end_b, start_b)
    denom = _norm(da) * _norm(db)
    if denom < 1e-8:
        return 0.0
    cosine = _dot(da, db) / denom
    return _clip01((cosine + 1.0) / 2.0)


def _alignment_required(task_meta: dict[str, Any] | None, step: dict[str, Any]) -> float:
    text = " ".join(
        str(x).lower()
        for x in [
            step.get("task_description", ""),
            (task_meta or {}).get("task_description", ""),
            (task_meta or {}).get("task_name", ""),
        ]
    )
    keywords = ["insert", "place", "align", "drawer", "handle", "button", "peg", "stack"]
    return 1.0 if any(word in text for word in keywords) else 0.0


def _warn_missing(step: dict[str, Any], names: list[str]) -> None:
    if not any(name in step for name in names):
        warnings.warn(f"TBD: missing simulator fields {names}; emitting conservative label.", stacklevel=2)


def compute_predicates(
    step: dict[str, Any],
    next_steps: list[dict[str, Any]],
    task_meta: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Return normalized predicate labels for one timestep."""

    merged_cfg = {**DEFAULT_CFG, **(cfg or {})}
    distance_norm = float(merged_cfg["distance_norm"])
    contact_distance = float(merged_cfg["contact_distance"])
    lift_height = float(merged_cfg["lift_height"])
    progress_epsilon = float(merged_cfg["goal_progress_epsilon"])

    gripper = _pos(step, "gripper_pos", "eef_pos", "end_effector_pos")
    obj = _pos(step, "object_pos", "target_object_pos", "target_pos")
    target = _pos(step, "target_object_pos", "object_pos", "target_pos")
    goal = _pos(step, "goal_pos", "target_goal_pos")

    _warn_missing(step, ["gripper_pos", "eef_pos", "end_effector_pos"])
    _warn_missing(step, ["object_pos", "target_object_pos", "target_pos"])

    gripper_obj_dist = _distance(gripper, target)
    obj_goal_dist = _distance(obj, goal)
    near_target = _inverse_distance_score(gripper_obj_dist, distance_norm)
    near_goal = _inverse_distance_score(obj_goal_dist, distance_norm)

    contact_flag = bool(step.get("contact", False) or step.get("contact_likely", False))
    contact_likely = 1.0 if contact_flag else _clip01(1.0 - (gripper_obj_dist or distance_norm) / contact_distance)

    gripper_closed = bool(step.get("gripper_closed", False))
    next_step = next_steps[0] if next_steps else {}
    next_obj = _pos(next_step, "object_pos", "target_object_pos", "target_pos")
    next_gripper = _pos(next_step, "gripper_pos", "eef_pos", "end_effector_pos")
    moving_with_gripper = _direction_alignment(obj, next_obj, gripper, next_gripper)
    object_grasped = 1.0 if bool(step.get("object_grasped", False)) else _clip01(contact_likely * (1.0 if gripper_closed else 0.5) * moving_with_gripper)

    initial_height = float(step.get("initial_object_height", (task_meta or {}).get("initial_object_height", 0.0)))
    object_height = obj[2] if obj is not None else initial_height
    object_lifted = _clip01((object_height - initial_height) / max(lift_height, 1e-6))

    initial_goal_distance = step.get("initial_object_goal_distance", (task_meta or {}).get("initial_object_goal_distance"))
    if obj_goal_dist is not None and initial_goal_distance:
        progress = _clip01(1.0 - obj_goal_dist / max(float(initial_goal_distance), 1e-6))
    else:
        progress = _clip01(float(step.get("progress", 0.0)))
        if "progress" not in step:
            warnings.warn("TBD: missing goal distance/progress; progress_score defaults to 0.", stacklevel=2)

    next_goal_dist = _distance(next_obj, goal)
    goal_distance_increasing = (
        obj_goal_dist is not None
        and next_goal_dist is not None
        and next_goal_dist > obj_goal_dist + progress_epsilon
    )
    geometric_stall = (
        obj_goal_dist is not None
        and next_goal_dist is not None
        and abs(next_goal_dist - obj_goal_dist) < progress_epsilon
        and progress < 0.9
    )
    stalled = bool(step.get("stalled", False)) or (
        geometric_stall and (contact_likely > 0.5 or object_grasped > 0.5 or progress > 0.05)
    )
    dropped = bool(step.get("dropped", False))
    collision = bool(step.get("collision", False))
    risk = _clip01(0.35 * float(goal_distance_increasing) + 0.25 * float(stalled) + 0.25 * float(dropped) + 0.25 * float(collision))
    needs_recovery = _clip01(max(risk, float(stalled and progress < 0.6)))

    labels = {
        "near_target_object": near_target,
        "contact_likely": contact_likely,
        "object_grasped": object_grasped,
        "object_lifted": object_lifted,
        "object_moving_with_gripper": moving_with_gripper,
        "near_goal_region": near_goal,
        "alignment_required": _alignment_required(task_meta, step),
        "progress_score": progress,
        "failure_risk": risk,
        "needs_recovery": needs_recovery,
    }
    labels = {name: _clip01(labels[name]) for name in PREDICATE_NAMES}
    validate_predicate_dict(labels)
    return labels


def label_trajectory(
    trajectory: list[dict[str, Any]],
    task_meta: dict[str, Any] | None = None,
    cfg: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    """Return one predicate dictionary per trajectory step."""

    return [
        compute_predicates(step, trajectory[idx + 1 : idx + 3], task_meta=task_meta, cfg=cfg)
        for idx, step in enumerate(trajectory)
    ]


def build_mock_trajectory() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create a tiny deterministic trajectory for local smoke checks."""

    task_meta = {
        "task_name": "mock_place_object",
        "task_description": "place the block into the goal region",
        "initial_object_height": 0.0,
        "initial_object_goal_distance": 0.30,
    }
    trajectory = [
        {
            "gripper_pos": [0.0, 0.0, 0.12],
            "object_pos": [0.35, 0.0, 0.0],
            "goal_pos": [0.65, 0.0, 0.0],
            "gripper_closed": False,
        },
        {
            "gripper_pos": [0.30, 0.0, 0.05],
            "object_pos": [0.35, 0.0, 0.0],
            "goal_pos": [0.65, 0.0, 0.0],
            "gripper_closed": True,
            "contact": True,
        },
        {
            "gripper_pos": [0.39, 0.0, 0.10],
            "object_pos": [0.39, 0.0, 0.07],
            "goal_pos": [0.65, 0.0, 0.0],
            "gripper_closed": True,
        },
        {
            "gripper_pos": [0.55, 0.0, 0.10],
            "object_pos": [0.55, 0.0, 0.07],
            "goal_pos": [0.65, 0.0, 0.0],
            "gripper_closed": True,
        },
        {
            "gripper_pos": [0.65, 0.0, 0.04],
            "object_pos": [0.64, 0.0, 0.01],
            "goal_pos": [0.65, 0.0, 0.0],
            "gripper_closed": False,
        },
    ]
    return trajectory, task_meta
