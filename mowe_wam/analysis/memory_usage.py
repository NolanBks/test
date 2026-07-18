"""Summaries for event-memory and expert-switch diagnostics."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def summarize_memory_usage(log_path: str | Path) -> dict:
    events: Counter[str] = Counter()
    switches = 0
    repeat_failures = 0
    previous_expert = None
    with Path(log_path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            event = row.get("memory_event")
            if event is not None:
                events[str(event)] += 1
                if str(event) in {"contact_lost", "progress_stall"}:
                    repeat_failures += 1
            expert = row.get("router_top1")
            if expert is not None and previous_expert is not None and int(expert) != int(previous_expert):
                switches += 1
            if expert is not None:
                previous_expert = expert
    return {
        "memory_event_counts": dict(sorted(events.items())),
        "expert_switch_count": switches,
        "failure_or_stall_event_count": repeat_failures,
    }
