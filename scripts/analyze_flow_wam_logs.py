#!/usr/bin/env python3
"""Summarize flow-WAM JSONL diagnostics without inventing benchmark metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SCALAR_FIELDS = (
    "total_loss",
    "nominal_flow_loss",
    "expert_flow_loss",
    "gripper_bce_loss",
    "gripper_accuracy",
    "route_ce_loss",
    "world_loss",
    "world_loss_gt_conditioned",
    "world_loss_nominal_conditioned",
    "delta_loss",
    "router_entropy",
    "current_skill_accuracy",
    "null_route_usage",
    "execution_steps_mean",
    "motion_residual_norm",
    "route_world_token_norm",
    "route_world_token_variance",
    "nominal_motion_target_l1",
    "motion_endpoint_l1",
    "future_position_accuracy",
    "boundary_precision",
    "boundary_recall",
    "boundary_f1",
    "schedule_edit_distance",
    "ground_truth_boundary_crossing_rate",
    "predicted_boundary_crossing_rate",
    "execution_boundary_entropy_mean",
    "execution_boundary_margin_mean",
    "execution_motion_jump_l2_mean",
    "execution_residual_l2_mean",
    "replanning_frequency",
    "forward_latency_ms",
    "future_shuffle_router_change_rate",
    "future_shuffle_router_logit_l1",
    "current_view_entropy_mean",
)


def summarize(rows):
    summary = {
        "records": len(rows),
        "first_step": rows[0].get("step"),
        "last_step": rows[-1].get("step"),
        "stages": sorted(
            {row.get("stage") for row in rows if row.get("stage") is not None}, key=str
        ),
        "ablations": sorted(
            {row.get("ablation", "main") for row in rows}, key=str
        ),
        "route_sources": {
            source: sum(row.get("route_source") == source for row in rows)
            for source in ("oracle", "st_gumbel", "predicted", "soft")
        },
        "null_motion_zero_violation_count": sum(
            int(row.get("null_motion_zero_violation_count", 0)) for row in rows
        ),
        "scalar_means": {},
        "scalar_means_by_route_source": {},
        "route_mode_diagnostic_means": {},
        "latest": rows[-1],
        "note": "Training diagnostics only; no success rate or benchmark claim is inferred.",
    }
    for field in SCALAR_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if values:
            summary["scalar_means"][field] = statistics.fmean(values)
    for source in ("oracle", "st_gumbel", "predicted", "soft"):
        source_rows = [row for row in rows if row.get("route_source") == source]
        if not source_rows:
            continue
        summary["scalar_means_by_route_source"][source] = {
            field: statistics.fmean(float(row[field]) for row in source_rows if row.get(field) is not None)
            for field in SCALAR_FIELDS
            if any(row.get(field) is not None for row in source_rows)
        }
    for mode in ("oracle", "st_gumbel", "hard_predicted"):
        records = [
            row["route_mode_diagnostics"][mode]
            for row in rows
            if mode in row.get("route_mode_diagnostics", {})
        ]
        if records:
            fields = set.intersection(*(set(record) for record in records))
            summary["route_mode_diagnostic_means"][mode] = {
                field: statistics.fmean(float(record[field]) for record in records)
                for field in sorted(fields)
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--compare", type=Path, help="Optional second run for mechanism/ablation deltas.")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.log.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No JSONL records found.")
    output = {"primary": summarize(rows)}
    if args.compare:
        compare_rows = [
            json.loads(line)
            for line in args.compare.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not compare_rows:
            raise SystemExit("Comparison JSONL contains no records.")
        comparison = summarize(compare_rows)
        shared = set(output["primary"]["scalar_means"]) & set(comparison["scalar_means"])
        output["comparison"] = comparison
        output["comparison_minus_primary"] = {
            field: comparison["scalar_means"][field] - output["primary"]["scalar_means"][field]
            for field in sorted(shared)
        }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
