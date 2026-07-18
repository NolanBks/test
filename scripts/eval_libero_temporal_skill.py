#!/usr/bin/env python3
"""Evaluate Flow-WAM offline, in one LIBERO task, or across a resumable full suite."""

from __future__ import annotations

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.evaluation import (
    TemporalSkillPolicyAdapter,
    VariablePrefixActionQueue,
    canonical_action_to_libero,
)
from mowe_wam.backbones import resolve_original_openvla_identity
from mowe_wam.backbones.openvla_oft_adapter import _ensure_openvla_path
from mowe_wam.training.flow_runtime import (
    build_flow_policy,
    deep_update,
    load_flow_checkpoint,
    read_flow_checkpoint_metadata,
    validate_backbone_identifier,
)
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def queue_smoke():
    lengths = iter([8, 4, 2])
    queries = []

    def policy(observation):
        query_id = len(queries) + 1
        queries.append(observation)
        length = next(lengths)
        return [f"query-{query_id}-action-{step}" for step in range(length)], {}

    queue = VariablePrefixActionQueue(policy)
    trace = []
    for step in range(14):
        action, metadata = queue.next_action(f"observation-{step}")
        trace.append({"action": action, **metadata})
    return {"kind": "variable_prefix_queue_smoke", "queries": queries, "trace": trace}


def load_policy(args):
    local_cfg = load_config(args.config)
    checkpoint_metadata = read_flow_checkpoint_metadata(args.policy_checkpoint)
    saved_cfg = checkpoint_metadata.get("config")
    if not isinstance(saved_cfg, dict) or not saved_cfg:
        raise ValueError("Policy checkpoint does not contain a resolved config.")
    requested_identity = resolve_original_openvla_identity(
        args.backbone_checkpoint,
        revision=args.backbone_revision,
        repo_id=local_cfg.get("backbone", {}).get("repo_id", "openvla/openvla-7b"),
    )
    validate_backbone_identifier(
        checkpoint_metadata,
        args.backbone_checkpoint,
        requested_identity=requested_identity,
    )
    cfg = copy.deepcopy(saved_cfg)
    deep_update(
        cfg,
        {
            "backbone": {
                "mode": "online_openvla",
                "feature_source": "pre_action_context",
                "checkpoint": args.backbone_checkpoint,
                "repo_id": requested_identity["repo_id"],
                "revision": requested_identity["revision"],
                "identity": requested_identity,
                "openvla_root": local_cfg.get("backbone", {}).get("openvla_root"),
                "freeze_backbone": True,
                "num_images_in_input": 2,
            },
            "teacher": {"cache_path": None, "inference_enabled": False},
            "data": {
                "backend": "rlds",
                "observation_views": ["primary", "wrist"],
                "image_aug": False,
            },
            "training": {"device": local_cfg.get("training", {}).get("device", "auto")},
        },
    )
    model = build_flow_policy(cfg, include_teacher=False)
    load_flow_checkpoint(
        args.policy_checkpoint,
        model,
        resume=False,
        metadata_out=checkpoint_metadata,
    )
    joint_statistics = checkpoint_metadata.get("data_contract", {}).get(
        "joint_action_statistics"
    ) or checkpoint_metadata.get("config", {}).get("data", {}).get("joint_action_statistics")
    model.eval()
    return cfg, model, checkpoint_metadata, joint_statistics


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_commit(path: str | Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _validate_libero_checkpoint(metadata, statistics, *, require_joint: bool) -> None:
    if statistics is None:
        raise RuntimeError(
            "Policy checkpoint has no data.joint_action_statistics. Run real training/conversion "
            "before evaluation so six-dimensional motion can be unnormalized safely."
        )
    if len(statistics.get("q01", [])) < 6 or len(statistics.get("q99", [])) < 6:
        raise ValueError("LIBERO checkpoint action statistics must contain six motion dimensions.")
    if require_joint and metadata.get("stage") != "joint":
        raise ValueError("Formal full-suite LIBERO evaluation requires a Stage 3 joint checkpoint.")
    data_contract = metadata.get("data_contract", {})
    source = (data_contract.get("feature_store_contract") or {}).get("source_contract", {})
    dataset_names = source.get("dataset_names") or metadata.get("config", {}).get("data", {}).get(
        "dataset_names", []
    )
    non_libero = [name for name in dataset_names if not str(name).startswith("libero_")]
    if non_libero:
        raise ValueError(
            "LIBERO evaluator refuses a checkpoint bound to non-LIBERO datasets: "
            f"{non_libero}."
        )


def _load_existing_records(
    path: Path | None,
    *,
    resume: bool,
    task_suite: str,
    checkpoint_path: str,
    seed: int,
    flow_seed: int,
    task_count: int,
    trials: int,
) -> list[dict]:
    if path is None or not path.exists():
        return []
    if not resume and path.stat().st_size:
        raise FileExistsError(
            f"Evaluation JSONL already exists: {path}. Pass --resume-results to continue."
        )
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("task_suite") != task_suite:
            raise ValueError("Existing evaluation JSONL belongs to a different task suite.")
        if record.get("policy_checkpoint") != checkpoint_path:
            raise ValueError("Existing evaluation JSONL belongs to a different checkpoint.")
        if int(record.get("seed", -1)) != seed or int(
            record.get("flow_seed", -1)
        ) != flow_seed:
            raise ValueError("Existing evaluation JSONL uses different seed settings.")
        if not 0 <= int(record["task_id"]) < task_count:
            raise ValueError("Existing evaluation JSONL contains an invalid task ID.")
        if not 0 <= int(record["trial"]) < trials:
            raise ValueError(
                "Existing evaluation JSONL is incompatible with the requested trial count."
            )
        records.append(record)
    keys = [(int(record["task_id"]), int(record["trial"])) for record in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Existing evaluation JSONL contains duplicate task/trial records.")
    return records


def simulator_evaluation(args):
    preview_cfg = load_config(args.config)
    _ensure_openvla_path(preview_cfg["backbone"].get("openvla_root", "external/openvla-oft"))
    try:
        import numpy as np
        from libero.libero import benchmark

        from experiments.robot.libero.libero_utils import (
            get_libero_dummy_action,
            get_libero_env,
            get_libero_image,
            get_libero_wrist_image,
            quat2axisangle,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "LIBERO simulator mode requires the upstream OpenVLA-OFT LIBERO environment; "
            "follow external/openvla-oft/LIBERO.md first."
        ) from exc

    cfg, model, checkpoint_metadata, statistics = load_policy(args)
    _validate_libero_checkpoint(
        checkpoint_metadata, statistics, require_joint=bool(args.all_tasks)
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch = require_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    suites = benchmark.get_benchmark_dict()
    if args.task_suite not in suites:
        raise ValueError(f"Unknown LIBERO task suite {args.task_suite!r}; available={sorted(suites)}")
    task_suite = suites[args.task_suite]()
    if not 0 <= args.task_id < task_suite.n_tasks:
        raise ValueError(f"task-id must be in [0, {task_suite.n_tasks - 1}].")
    task_ids = list(range(task_suite.n_tasks)) if args.all_tasks else [args.task_id]
    trials = args.trials if args.trials is not None else (50 if args.all_tasks else 1)
    if trials < 1:
        raise ValueError("trials must be positive.")
    output_path = Path(args.output_jsonl) if args.output_jsonl else None
    if args.all_tasks and output_path is None:
        raise ValueError("Formal --all-tasks evaluation requires --output-jsonl for resumability.")
    checkpoint_path = str(Path(args.policy_checkpoint).expanduser().resolve())
    records = _load_existing_records(
        output_path,
        resume=args.resume_results,
        task_suite=args.task_suite,
        checkpoint_path=checkpoint_path,
        seed=args.seed,
        flow_seed=args.flow_seed,
        task_count=task_suite.n_tasks,
        trials=trials,
    )
    completed = {(int(record["task_id"]), int(record["trial"])) for record in records}
    upstream_commit = _git_commit(cfg["backbone"].get("openvla_root", "external/openvla-oft"))
    max_steps_by_suite = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    max_steps = args.max_env_steps or max_steps_by_suite[args.task_suite]
    for task_id in task_ids:
        pending_trials = [trial for trial in range(trials) if (task_id, trial) not in completed]
        if not pending_trials:
            continue
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        if trials > len(initial_states):
            raise ValueError(
                f"Requested {trials} trials but task {task_id} has only "
                f"{len(initial_states)} initial states."
            )
        env, instruction = get_libero_env(
            task, "openvla", resolution=args.env_image_resolution
        )
        try:
            for trial in pending_trials:
                env.reset()
                obs = env.set_init_state(initial_states[trial])
                for _ in range(args.wait_steps):
                    obs, _, _, _ = env.step(get_libero_dummy_action("openvla"))
                adapter = TemporalSkillPolicyAdapter(
                    model,
                    model.backbone.processor.image_processor.apply_transform,
                    action_statistics=statistics,
                    history_length=int(cfg["data"].get("history_length", 8)),
                    long_memory_slots=int(cfg["data"].get("long_memory_slots", 4)),
                    flow_seed=args.flow_seed + task_id * 1_000_000 + trial * 10_000,
                )
                adapter.reset(instruction)
                success = False
                actions_executed = 0
                for _ in range(max_steps):
                    image = get_libero_image(obs)
                    wrist_image = get_libero_wrist_image(obs)
                    proprio = np.concatenate(
                        [
                            obs["robot0_eef_pos"],
                            quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]
                    )
                    action, _ = adapter.next_libero_action(image, wrist_image, proprio)
                    obs, _, done, _ = env.step(action.tolist())
                    actions_executed += 1
                    if done:
                        success = True
                        break
                prefix_lengths = [
                    query["prefix_length"] for query in adapter.query_records
                ]
                record = {
                    "kind": "libero_temporal_skill_episode_v1",
                    "task_suite": args.task_suite,
                    "task_id": task_id,
                    "instruction": instruction,
                    "trial": trial,
                    "success": success,
                    "actions_executed": actions_executed,
                    "policy_queries": len(prefix_lengths),
                    "replanning_frequency": len(prefix_lengths)
                    / max(actions_executed, 1),
                    "prefix_lengths": prefix_lengths,
                    "execution_reason_codes": [
                        query["execution_reason_code"] for query in adapter.query_records
                    ],
                    "execution_boundary_positions": [
                        query["execution_boundary_position"] for query in adapter.query_records
                    ],
                    "predicted_boundary_crossings": [
                        query["execution_crosses_predicted_boundary"]
                        for query in adapter.query_records
                    ],
                    "current_view_weights": [
                        query["current_view_weights"] for query in adapter.query_records
                    ],
                    "view_order": ["primary", "wrist"],
                    "checkpoint_step": checkpoint_metadata.get("step"),
                    "checkpoint_stage": checkpoint_metadata.get("stage"),
                    "policy_checkpoint": checkpoint_path,
                    "backbone_checkpoint": str(args.backbone_checkpoint),
                    "backbone_revision": str(args.backbone_revision),
                    "backbone_identity_sha256": cfg["backbone"]["identity"][
                        "identity_sha256"
                    ],
                    "upstream_openvla_oft_commit": upstream_commit,
                    "seed": args.seed,
                    "flow_seed": args.flow_seed,
                    "max_env_steps": max_steps,
                    "wait_steps": args.wait_steps,
                    "teacher_loaded": model.visual_teacher is not None,
                    "video_saved": False,
                }
                records.append(record)
                completed.add((task_id, trial))
                if output_path is not None:
                    _append_jsonl(output_path, record)
                print(
                    json.dumps(
                        {
                            "task_id": task_id,
                            "trial": trial,
                            "success": success,
                            "completed_episodes": len(records),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        finally:
            env.close()
    selected_records = [record for record in records if int(record["task_id"]) in task_ids]
    per_task = {}
    for task_id in task_ids:
        task_records = [
            record for record in selected_records if int(record["task_id"]) == task_id
        ]
        per_task[str(task_id)] = {
            "episodes": len(task_records),
            "successes": sum(bool(record["success"]) for record in task_records),
            "success_rate": sum(bool(record["success"]) for record in task_records)
            / max(len(task_records), 1),
        }
    summary = {
        "kind": "libero_temporal_skill_summary",
        "task_suite": args.task_suite,
        "task_ids": task_ids,
        "trials_per_task": trials,
        "episodes": len(selected_records),
        "expected_episodes": len(task_ids) * trials,
        "complete": len(selected_records) == len(task_ids) * trials,
        "successes": sum(bool(record["success"]) for record in selected_records),
        "success_rate": sum(bool(record["success"]) for record in selected_records)
        / max(len(selected_records), 1),
        "per_task": per_task,
        "policy_checkpoint": checkpoint_path,
        "checkpoint_step": checkpoint_metadata.get("step"),
        "checkpoint_stage": checkpoint_metadata.get("stage"),
        "backbone_checkpoint": str(Path(args.backbone_checkpoint).resolve()),
        "backbone_revision": str(args.backbone_revision),
        "checkpoint_backbone_identity": checkpoint_metadata.get("backbone_identity"),
        "checkpoint_backbone_identifier": checkpoint_metadata.get(
            "backbone_identifier"
        ),
        "upstream_openvla_oft_commit": upstream_commit,
        "seed": args.seed,
        "flow_seed": args.flow_seed,
        "note": "No rollout video or teacher model is used by this evaluator.",
    }
    if args.summary_output:
        _atomic_json(Path(args.summary_output), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_flow_wam_skill_moe.yaml")
    parser.add_argument("--policy-checkpoint")
    parser.add_argument("--backbone-checkpoint")
    parser.add_argument("--backbone-revision")
    parser.add_argument("--prepared-batch", type=Path)
    parser.add_argument("--flow-seed", type=int, default=7)
    parser.add_argument("--queue-smoke", action="store_true")
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--all-tasks", action="store_true")
    parser.add_argument("--trials", type=int)
    parser.add_argument("--wait-steps", type=int, default=10)
    parser.add_argument("--max-env-steps", type=int)
    parser.add_argument("--env-image-resolution", type=int, default=256)
    parser.add_argument("--output-jsonl")
    parser.add_argument("--summary-output")
    parser.add_argument("--resume-results", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.queue_smoke:
        print(json.dumps(queue_smoke(), indent=2))
        return
    if not args.policy_checkpoint or not args.backbone_checkpoint or not args.backbone_revision:
        raise SystemExit(
            "Policy evaluation requires --policy-checkpoint, --backbone-checkpoint, and "
            "--backbone-revision; "
            "use --queue-smoke for the dependency-light adapter check."
        )
    if args.simulator:
        print(json.dumps(simulator_evaluation(args), indent=2))
        return
    if not args.prepared_batch:
        raise SystemExit("Offline policy evaluation also requires --prepared-batch.")
    torch = require_torch()
    _, model, checkpoint_metadata, statistics = load_policy(args)
    batch = torch.load(args.prepared_batch, map_location="cpu")
    prefixes, output = model.predict_actions(batch, flow_seed=args.flow_seed)
    converted = [
        canonical_action_to_libero(prefix, statistics).detach().cpu().tolist() for prefix in prefixes
    ]
    print(
        json.dumps(
            {
                "kind": "offline_temporal_policy_not_benchmark",
                "prefix_lengths": [len(prefix) for prefix in prefixes],
                "route_indices": output["route_indices"].detach().cpu().tolist(),
                "execution_reason_codes": output["execution_reason_code"].detach().cpu().tolist(),
                "execution_boundary_positions": output["execution_boundary_position"].detach().cpu().tolist(),
                "predicted_boundary_crossings": output[
                    "execution_crosses_predicted_boundary"
                ].detach().cpu().tolist(),
                "libero_actions": converted,
                "motion_unnormalization_applied": statistics is not None,
                "checkpoint_step": checkpoint_metadata.get("step"),
                "checkpoint_backbone_identity": checkpoint_metadata.get(
                    "backbone_identity"
                ),
                "backbone_revision": str(args.backbone_revision),
                "teacher_loaded": model.visual_teacher is not None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
