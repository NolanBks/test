"""Predicate timeline summaries for JSONL logs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(log_path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]


def build_predicate_timeline(log_path: str | Path) -> dict[str, Any]:
    rows = _read_jsonl(log_path)
    series: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for name, value in row.get("predicates", {}).items():
            series[name].append(float(value))
    return {"num_steps": len(rows), "predicate_timeline": dict(series)}


def compute_phase_expert_alignment(log_path: str | Path) -> dict[str, Any]:
    rows = _read_jsonl(log_path)
    phase_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        predicates = row.get("predicates", {})
        expert = str(row.get("router_top1", "unknown"))
        phase = _dominant_phase(predicates)
        phase_counts[phase][expert] += 1
    return {"phase_expert_alignment": {phase: dict(counts) for phase, counts in phase_counts.items()}}


def _dominant_phase(predicates: dict[str, float]) -> str:
    if predicates.get("needs_recovery", 0.0) > 0.5:
        return "recovery"
    if predicates.get("near_goal_region", 0.0) > 0.6:
        return "align_place"
    if predicates.get("object_grasped", 0.0) > 0.5:
        return "transport"
    if predicates.get("contact_likely", 0.0) > 0.5:
        return "contact"
    return "approach"
