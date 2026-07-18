"""Stable low-dimensional world-predicate schema."""

from __future__ import annotations

from collections.abc import Sequence

PREDICATE_NAMES: list[str] = [
    "near_target_object",
    "contact_likely",
    "object_grasped",
    "object_lifted",
    "object_moving_with_gripper",
    "near_goal_region",
    "alignment_required",
    "progress_score",
    "failure_risk",
    "needs_recovery",
]

_PREDICATE_TO_INDEX = {name: idx for idx, name in enumerate(PREDICATE_NAMES)}


def predicate_dim() -> int:
    """Return the fixed predicate vector dimension."""

    return len(PREDICATE_NAMES)


def predicate_index(name: str) -> int:
    """Return a stable integer index for a predicate name."""

    try:
        return _PREDICATE_TO_INDEX[name]
    except KeyError as exc:
        known = ", ".join(PREDICATE_NAMES)
        raise KeyError(f"Unknown predicate {name!r}. Known predicates: {known}") from exc


def validate_predicate_tensor(x: object) -> None:
    """Raise if the final dimension does not match ``predicate_dim()``."""

    shape = getattr(x, "shape", None)
    if shape is None and isinstance(x, Sequence):
        if not x:
            raise ValueError("Predicate sequence is empty; cannot infer final dimension.")
        shape = (len(x),)

    if shape is None:
        raise TypeError("Predicate tensor must expose a shape or be a sequence.")

    if len(shape) == 0:
        raise ValueError("Predicate tensor must have at least one dimension.")

    final_dim = int(shape[-1])
    expected = predicate_dim()
    if final_dim != expected:
        raise ValueError(f"Expected final predicate dimension {expected}, got {final_dim}.")


def predicate_dict_to_vector(values: dict[str, float]) -> list[float]:
    """Convert a predicate dictionary to a schema-ordered vector."""

    missing = [name for name in PREDICATE_NAMES if name not in values]
    if missing:
        raise ValueError(f"Missing predicate values: {', '.join(missing)}")
    return [float(values[name]) for name in PREDICATE_NAMES]


def validate_predicate_dict(values: dict[str, float]) -> None:
    """Validate predicate keys and zero-one value range."""

    expected = set(PREDICATE_NAMES)
    actual = set(values)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"Predicate keys mismatch. Missing={missing}; extra={extra}")
    bad = {name: value for name, value in values.items() if not 0.0 <= float(value) <= 1.0}
    if bad:
        raise ValueError(f"Predicate values must be in [0, 1], got {bad}")
