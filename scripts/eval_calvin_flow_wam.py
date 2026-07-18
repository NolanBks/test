#!/usr/bin/env python3
"""Run MoWE through CALVIN's official sequences, task oracle, and result writer."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs/mowe_wam/calvin_abc_d.yaml"))
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--backbone-revision", required=True)
    parser.add_argument(
        "--local-config",
        default=str(PROJECT_ROOT / "configs/mowe_wam/train_flow_wam_skill_moe.yaml"),
    )
    parser.add_argument("--eval-log-dir", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _summarize_sequence_records(records):
    per_task = defaultdict(lambda: {"attempts": 0, "successes": 0})
    failure_positions = Counter()
    query_count = 0
    prefix_lengths = Counter()
    execution_reasons = Counter()
    predicted_boundary_crossings = 0
    for record in records:
        subtasks = list(record["subtasks"])
        completed = int(record["completed"])
        if not 0 <= completed <= len(subtasks):
            raise ValueError(
                f"Invalid CALVIN completed count {completed} for {len(subtasks)} subtasks."
            )
        for position, subtask in enumerate(subtasks):
            if position > completed:
                break
            per_task[str(subtask)]["attempts"] += 1
            if position < completed:
                per_task[str(subtask)]["successes"] += 1
        failure_positions["complete" if completed == len(subtasks) else str(completed + 1)] += 1
        query_count += int(record.get("policy_queries", 0))
        prefix_lengths.update(str(int(value)) for value in record.get("prefix_lengths", []))
        execution_reasons.update(
            str(int(value)) for value in record.get("execution_reason_codes", [])
        )
        predicted_boundary_crossings += sum(
            bool(value) for value in record.get("predicted_boundary_crossings", [])
        )
    rendered_tasks = {}
    for task, counts in sorted(per_task.items()):
        attempts = int(counts["attempts"])
        successes = int(counts["successes"])
        rendered_tasks[task] = {
            "attempts": attempts,
            "successes": successes,
            "success_rate": successes / attempts if attempts else 0.0,
        }
    return {
        "per_task": rendered_tasks,
        "failure_position_counts": dict(sorted(failure_positions.items())),
        "policy_queries": query_count,
        "prefix_length_histogram": dict(sorted(prefix_lengths.items(), key=lambda item: int(item[0]))),
        "execution_reason_histogram": dict(
            sorted(execution_reasons.items(), key=lambda item: int(item[0]))
        ),
        "predicted_boundary_crossing_count": predicted_boundary_crossings,
    }


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    calvin_root = Path(args.calvin_root).resolve()
    if not (calvin_root / "calvin_models/calvin_agent/evaluation/evaluate_policy.py").exists():
        raise SystemExit(f"Official CALVIN evaluation file not found under {calvin_root}.")
    sys.path.insert(0, str(calvin_root))
    sys.path.insert(0, str(calvin_root / "calvin_models"))
    os.environ.update(
        {
            "MOWE_CALVIN_CONFIG": str(Path(args.config).resolve()),
            "MOWE_FLOW_CHECKPOINT": str(Path(args.flow_checkpoint).resolve()),
            "MOWE_BACKBONE_CHECKPOINT": str(Path(args.backbone_checkpoint).resolve()),
            "MOWE_BACKBONE_REVISION": str(args.backbone_revision),
            "MOWE_LOCAL_CONFIG": str(Path(args.local_config).resolve()),
        }
    )
    try:
        from calvin_agent.evaluation import evaluate_policy as official
        from mowe_wam.benchmarks.calvin.custom_model import CustomModel
        from mowe_wam.utils.config import load_config
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "CALVIN dependencies are unavailable. Install the pinned official repo/environment first."
        ) from exc
    cfg = load_config(args.config)
    expected_sequences = int(cfg["benchmark"].get("evaluation_sequences", 1000))
    if int(official.NUM_SEQUENCES) != expected_sequences:
        raise RuntimeError(
            f"Official evaluator NUM_SEQUENCES={official.NUM_SEQUENCES} differs from config "
            f"{expected_sequences}."
        )
    random.seed(args.seed)
    try:
        import numpy as np

        np.random.seed(args.seed)
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ModuleNotFoundError:
        pass
    model = CustomModel()
    environment = official.make_env(args.dataset_path)
    original_evaluate_sequence = official.evaluate_sequence
    sequence_records = []

    def evaluate_sequence_with_explicit_model_reset(
        env, policy, task_checker, initial_state, eval_sequence, val_annotations, plans, debug
    ):
        policy.reset_sequence()
        completed = original_evaluate_sequence(
            env,
            policy,
            task_checker,
            initial_state,
            eval_sequence,
            val_annotations,
            plans,
            debug,
        )
        queries = list(policy.query_records)
        sequence_records.append(
            {
                "sequence_index": len(sequence_records),
                "subtasks": [str(value) for value in eval_sequence],
                "completed": int(completed),
                "policy_queries": len(queries),
                "prefix_lengths": [int(value["prefix_length"]) for value in queries],
                "execution_reason_codes": [
                    int(value["execution_reason_code"]) for value in queries
                ],
                "predicted_boundary_crossings": [
                    bool(value["execution_crosses_predicted_boundary"])
                    for value in queries
                ],
            }
        )
        return completed

    official.evaluate_sequence = evaluate_sequence_with_explicit_model_reset
    try:
        results = official.evaluate_policy(
            model,
            environment,
            epoch=str(model.checkpoint_metadata.get("step", "unknown")),
            eval_log_dir=args.eval_log_dir,
            debug=args.debug,
            create_plan_tsne=False,
        )
    finally:
        official.evaluate_sequence = original_evaluate_sequence
        close = getattr(environment, "close", None)
        if close is not None:
            close()
    success = official.count_success(results)
    if len(sequence_records) != len(results):
        raise RuntimeError(
            "CALVIN evaluator sequence trace count differs from the official result count."
        )
    diagnostics = _summarize_sequence_records(sequence_records)
    report = {
        "kind": "mowe_calvin_official_lh_mtlc",
        "sequence_count": len(results),
        "average_sequence_length": sum(int(value) for value in results) / max(len(results), 1),
        "success_rate_at_k": {str(index + 1): float(value) for index, value in enumerate(success)},
        "checkpoint_step": model.checkpoint_metadata.get("step"),
        "checkpoint_stage": model.checkpoint_metadata.get("stage"),
        "flow_checkpoint": str(Path(args.flow_checkpoint).resolve()),
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "backbone_revision": str(args.backbone_revision),
        "checkpoint_backbone_identity": model.checkpoint_metadata.get("backbone_identity"),
        "checkpoint_backbone_identifier": model.checkpoint_metadata.get(
            "backbone_identifier"
        ),
        "evaluation_seed": int(args.seed),
        "flow_seed": int(cfg.get("policy", {}).get("flow_seed", 1701)),
        "official_repo_commit": cfg["benchmark"].get("official_repo_commit"),
        "official_num_sequences": official.NUM_SEQUENCES,
        "official_episode_length": official.EP_LEN,
        "memory_lifecycle": {
            "sequence_reset": "explicit bridge at official environment reset",
            "subtask_reset": "discard goal/action suffix; preserve visual/action memory",
        },
        **diagnostics,
        "eval_log_dir": str(Path(args.eval_log_dir).resolve()),
    }
    output = Path(args.eval_log_dir) / "mowe_calvin_summary.json"
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
