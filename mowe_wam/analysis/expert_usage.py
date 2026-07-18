"""Expert usage summaries for JSONL logs."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(log_path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(log_path).read_text().splitlines() if line.strip()]


def summarize_expert_usage(log_path: str | Path) -> dict[str, Any]:
    rows = _read_jsonl(log_path)
    counts: Counter[int] = Counter()
    for row in rows:
        if "router_top1" in row:
            counts[int(row["router_top1"])] += 1
        for expert in row.get("topk_experts", []):
            counts[int(expert)] += 0
    total = sum(counts.values())
    probs = [count / total for count in counts.values()] if total else []
    entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    return {
        "num_events": len(rows),
        "expert_usage_counts": {str(k): v for k, v in sorted(counts.items())},
        "expert_usage_entropy": entropy,
    }
