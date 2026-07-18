#!/usr/bin/env python3
"""Summarize latent-WAM training diagnostics without fabricating success rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in Path(args.log).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No JSONL rows found.")
    scalar_keys = (
        "total_loss",
        "action_loss",
        "nominal_action_loss",
        "world_loss",
        "delta_loss",
        "router_entropy",
        "nominal_target_l1",
        "final_target_l1",
        "residual_norm",
        "action_distance_gate_mean",
    )
    summary = {
        "kind": "training_mechanism_summary_not_benchmark",
        "steps": len(rows),
        "first_step": rows[0].get("step"),
        "last_step": rows[-1].get("step"),
        "last": {key: rows[-1].get(key) for key in scalar_keys if key in rows[-1]},
        "means": {
            key: sum(float(row[key]) for row in rows if key in row) / sum(key in row for row in rows)
            for key in scalar_keys
            if any(key in row for row in rows)
        },
        "last_expert_usage": rows[-1].get("expert_usage"),
        "last_future_horizon_mse": rows[-1].get("future_horizon_mse"),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

