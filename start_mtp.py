#!/usr/bin/env python3
"""MoWE 8-GPU formal training launcher with fail-closed audits and resume.

This file is intentionally self-contained so managed training platforms can
launch the complete MoWE pipeline through one Python entrypoint. Re-running the
same command resumes from the newest stage checkpoint. The launcher performs no
host, container, process-memory, OOM-event, or GPU-memory telemetry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCRIPT_ROOT = Path(__file__).resolve().parent
MOTOR_SKILLS = (
    "pick_grasp",
    "place_release",
    "move_transport",
    "open_close",
    "turn_rotate",
    "push_pull",
)
ALL_SKILLS = (*MOTOR_SKILLS, "null_finish")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def same_path(left: Any, right: Path) -> bool:
    if left in {None, ""}:
        return False
    return Path(str(left)).expanduser().resolve() == right.expanduser().resolve()


def report_matches_store(path: Path, store: Path, *, kind: str, samples: int = 0) -> bool:
    if not path.is_file():
        return False
    try:
        report = load_json(path)
        if kind == "feature":
            assignment = report.get("assignment", {})
            return (
                report.get("valid") is True
                and report.get("formal_training_ready") is True
                and report.get("checksums_verified") is True
                and same_path(report.get("root"), store)
                and int(assignment.get("world_size", 0)) == 8
                and assignment.get("episode_union_complete") is True
                and not assignment.get("episode_overlap")
                and assignment.get("target_skill_union_complete") is True
                and all(
                    assignment.get("imbalance_checks", {}).get(name) is True
                    for name in ("windows", "suites", "skills")
                )
            )
        if kind == "equivalence":
            return (
                report.get("passed") is True
                and same_path(report.get("store"), store)
                and int(report.get("compared_samples", 0)) >= samples
                and not report.get("missing_pairs")
                and report.get("masks_match") is True
                and report.get("comparison_contract", {}).get("name")
                == "mask_aware_training_metric_v1"
            )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return False


def checkpoint_metadata(path: Path, expected_stage: str | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    sidecar = path.with_suffix(path.suffix + ".metadata.json")
    if not sidecar.is_file():
        raise RuntimeError(f"Checkpoint metadata sidecar is missing: {sidecar}")
    metadata = load_json(sidecar)
    if metadata.get("format") != "flow_wam_skill_components_v2":
        raise RuntimeError(f"Unsupported checkpoint format: {path}")
    if expected_stage is not None and metadata.get("stage") != expected_stage:
        raise RuntimeError(
            f"Checkpoint stage mismatch: {metadata.get('stage')!r} != {expected_stage!r}: {path}"
        )
    return metadata


def checkpoint_predecessor_identity(metadata: dict[str, Any]) -> dict[str, Any]:
    from mowe_wam.training.flow_runtime import checkpoint_semantic_identity

    return checkpoint_semantic_identity(metadata)


def jsonl_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def stage1_quality_gate(
    records: list[dict[str, Any]],
    *,
    required_average_improvement: float = 0.10,
    required_min_horizon_improvement: float = 0.0,
    min_unique_episodes: int = 32,
    min_action_distance_gate: float = 0.10,
    validation_step: int | None = None,
) -> dict[str, Any]:
    """Evaluate the latest deployment-like Stage 1 mechanism evidence."""

    eligible = [
        record
        for record in records
        if record.get("stage") == "nominal_flow_pretrain"
        and record.get("validation_mode") == "deployment"
        and (
            validation_step is None
            or int(record.get("step", -1)) == int(validation_step)
        )
    ]
    if not eligible:
        return {
            "format": "mowe_stage1_quality_gate_v2",
            "passed": False,
            "errors": ["deployment_validation_missing"],
        }
    latest = max(eligible, key=lambda record: int(record.get("step", -1)))
    errors = []
    improvements = {}
    for horizon in (4, 8, 16):
        metrics = latest.get("future_horizon_metrics", {}).get(str(horizon), {})
        predicted = metrics.get("smooth_l1")
        baseline = metrics.get("current_copy_smooth_l1")
        if not isinstance(predicted, (int, float)) or not isinstance(
            baseline, (int, float)
        ) or not math.isfinite(float(predicted)) or not math.isfinite(float(baseline)):
            errors.append(f"horizon_{horizon}_metrics_missing")
            continue
        if float(baseline) <= 0:
            errors.append(f"horizon_{horizon}_copy_baseline_nonpositive")
            continue
        improvements[str(horizon)] = 1.0 - float(predicted) / float(baseline)
    average_improvement = (
        sum(improvements.values()) / len(improvements) if improvements else float("-inf")
    )
    if len(improvements) != 3:
        errors.append("required_horizons_incomplete")
    if average_improvement < required_average_improvement:
        errors.append("average_copy_current_improvement_below_threshold")
    if improvements and min(improvements.values()) < required_min_horizon_improvement:
        errors.append("one_or_more_horizons_worse_than_copy_current")
    unique_episodes = int(latest.get("unique_episodes", 0))
    if unique_episodes < min_unique_episodes:
        errors.append("insufficient_unique_validation_episodes")
    if latest.get("sampling_contract") != "episode_balanced_deterministic_v1":
        errors.append("validation_sampling_contract_mismatch")
    deployment = latest.get("deployment_metrics", {})
    action_gate = float(deployment.get("mean_nominal_action_distance_gate", 0.0))
    if not math.isfinite(action_gate) or action_gate < min_action_distance_gate:
        errors.append("nominal_action_distance_gate_collapsed")
    view_weights = deployment.get("current_view_weights_mean", [])
    view_weights_valid = (
        len(view_weights) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in view_weights)
        and math.isclose(sum(view_weights), 1.0, rel_tol=0.0, abs_tol=1e-4)
        and min(view_weights) >= 0.05
    )
    if not view_weights_valid:
        errors.append("view_weights_invalid_or_collapsed")
    return {
        "format": "mowe_stage1_quality_gate_v2",
        "passed": not errors,
        "stage": "nominal_flow_pretrain",
        "validation_step": int(latest.get("step", -1)),
        "validation_mode": "deployment",
        "unique_episodes": unique_episodes,
        "sampling_contract": latest.get("sampling_contract"),
        "fractional_improvement_over_copy_current": improvements,
        "average_improvement": average_improvement,
        "required_average_improvement": float(required_average_improvement),
        "required_min_horizon_improvement": float(
            required_min_horizon_improvement
        ),
        "mean_nominal_action_distance_gate": action_gate,
        "minimum_action_distance_gate": float(min_action_distance_gate),
        "current_view_weights_mean": view_weights,
        "view_weights_valid_and_not_collapsed": view_weights_valid,
        "errors": errors,
    }


class Launcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.repo = args.repo_root.resolve()
        self.run_root = (args.run_root_dir / args.run_id).resolve()
        self.report_root = (args.report_root or self.run_root / "reports").resolve()
        self.stage_root = (args.stage_root or self.run_root / "ddp8").resolve()
        self.state_path = self.run_root / "launcher_state.json"
        self.state = load_json(self.state_path) if self.state_path.is_file() else {
            "format": "mowe_mtp_launcher_state_v1",
            "created_at": utc_now(),
            "tasks": {},
        }
        self.python = str(args.python.resolve()) if args.python else sys.executable
        self.gpus = args.cuda_devices.split(",")
        self.virtual_checkpoints: dict[str, dict[str, Any]] = {}
        self.runtime_configs: dict[str, str] = {}
        self.env = os.environ.copy()
        self.env.update(
            {
                "CUDA_VISIBLE_DEVICES": args.cuda_devices,
                "TOKENIZERS_PARALLELISM": "false",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "PYTHONUNBUFFERED": "1",
                "PYTHONPATH": os.pathsep.join(
                    filter(
                        None,
                        (
                            str(self.repo),
                            str(self.repo / "external/openvla-oft"),
                            self.env.get("PYTHONPATH"),
                        ),
                    )
                ),
            }
        )
        # The one-click workflow intentionally performs no host/container/GPU
        # resource telemetry. Keep the environment guard enabled as a second
        # line of defense for imported training utilities.
        self.env["MOWE_DISABLE_SYSTEM_MONITORING"] = "1"
        os.environ["MOWE_DISABLE_SYSTEM_MONITORING"] = "1"

    def save_state(self) -> None:
        self.state["updated_at"] = utc_now()
        self.state["run_id"] = self.args.run_id
        self.state["paths"] = {
            "repo_root": str(self.repo),
            "data_root": str(self.args.data_root),
            "feature_store": str(self.args.feature_store),
            "openvla_checkpoint": str(self.args.openvla_checkpoint),
            "dino_checkpoint": str(self.args.dino_checkpoint),
            "skill_sidecar": str(self.args.skill_sidecar),
            "run_root": str(self.run_root),
        }
        atomic_json(self.state_path, self.state)

    def record(self, name: str, **values: Any) -> None:
        task = self.state.setdefault("tasks", {}).setdefault(name, {})
        task.update(values)
        self.save_state()

    def run(self, name: str, command: list[str], log_path: Path | None = None) -> None:
        rendered = shlex.join(command)
        print(f"\n[{utc_now()}] START {name}\n{rendered}", flush=True)
        if self.args.dry_run:
            self.record(name, status="dry_run", command=command, log=str(log_path) if log_path else None)
            return
        log_path = log_path or self.report_root / f"{name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.record(name, status="running", started_at=utc_now(), command=command, log=str(log_path))
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n[{utc_now()}] COMMAND {rendered}\n")
            handle.flush()
            process = subprocess.Popen(
                command,
                cwd=self.repo,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            previous_handlers = {}

            def forward(signum, _frame):
                self.record(name, status="terminating", signal=signum, updated_at=utc_now())
                try:
                    os.killpg(process.pid, signum)
                except ProcessLookupError:
                    pass

            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, forward)
            try:
                assert process.stdout is not None
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                    handle.write(line)
                    handle.flush()
            except KeyboardInterrupt:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                process.wait(timeout=30)
                raise
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
            returncode = process.wait()
        self.record(name, status="complete" if returncode == 0 else "failed", completed_at=utc_now(), returncode=returncode)
        if returncode != 0:
            raise RuntimeError(f"Task {name!r} failed with exit code {returncode}; see {log_path}")

    def python_command(self, script: str, *values: Any) -> list[str]:
        return [self.python, str(self.repo / script), *(str(value) for value in values)]

    def torchrun_command(self, script: str, *values: Any) -> list[str]:
        return [
            self.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            f"--nproc-per-node={self.args.world_size}",
            str(self.repo / script),
            *(str(value) for value in values),
        ]

    def preflight(self) -> None:
        required = {
            "repository": self.repo,
            "feature store": self.args.feature_store,
            "RLDS data root": self.args.data_root,
            "OpenVLA snapshot": self.args.openvla_checkpoint,
            "DINO snapshot": self.args.dino_checkpoint,
            "skill sidecar": self.args.skill_sidecar,
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        if self.args.python is not None and not self.args.python.is_file():
            raise FileNotFoundError(f"Python executable does not exist: {self.args.python}")
        for relative in (
            self.args.stage1_config,
            self.args.stage2_config,
            self.args.stage3_config,
            "scripts/audit_mowe_feature_store.py",
            "scripts/audit_feature_store_equivalence.py",
        ):
            if not (self.repo / relative).is_file():
                raise FileNotFoundError(f"Required repository file is missing: {relative}")
        if len(self.gpus) != self.args.world_size or len(set(self.gpus)) != self.args.world_size:
            raise ValueError("--cuda-devices must contain exactly --world-size unique device IDs.")
        if self.args.world_size != 8:
            raise ValueError("The current formal contract requires --world-size 8.")
        if len(self.args.openvla_revision) != 40 or any(
            character not in "0123456789abcdef" for character in self.args.openvla_revision.lower()
        ):
            raise ValueError("--openvla-revision must be a 40-character hexadecimal commit.")
        if not 100 <= self.args.equivalence_samples:
            raise ValueError("Formal training requires --equivalence-samples >= 100.")
        if min(self.args.stage1_max_steps, self.args.stage2_max_steps, self.args.stage3_max_steps) < 100:
            raise ValueError("Every formal stage must contain at least the 100-step smoke gate.")
        if self.args.validation_freq < 1:
            raise ValueError("--validation-freq must be positive.")
        if not math.isfinite(self.args.early_stop_min_delta) or self.args.early_stop_min_delta < 0:
            raise ValueError("--early-stop-min-delta must be non-negative.")
        if self.args.early_stop_patience < 1:
            raise ValueError("--early-stop-patience must be positive.")
        if self.args.early_stop_min_steps < 100:
            raise ValueError("--early-stop-min-steps must be at least 100.")
        if self.args.min_validation_episodes < 2:
            raise ValueError("--min-validation-episodes must be at least 2.")
        if (
            not math.isfinite(self.args.stage1_quality_average_improvement)
            or self.args.stage1_quality_average_improvement < 0
        ):
            raise ValueError("--stage1-quality-average-improvement must be non-negative.")
        if not math.isfinite(self.args.stage1_quality_min_horizon_improvement):
            raise ValueError("--stage1-quality-min-horizon-improvement must be finite.")
        if (
            not math.isfinite(self.args.stage1_quality_min_action_gate)
            or self.args.stage1_quality_min_action_gate <= 0
        ):
            raise ValueError("--stage1-quality-min-action-gate must be positive.")
        if not self.args.dry_run:
            probe = subprocess.run(
                [
                    self.python,
                    "-c",
                    "import torch; print(torch.cuda.device_count()); "
                    "assert torch.cuda.device_count() == 8, torch.cuda.device_count()",
                ],
                cwd=self.repo,
                env=self.env,
                check=False,
                text=True,
                capture_output=True,
            )
            if probe.returncode != 0:
                raise RuntimeError(f"Eight-GPU preflight failed: {probe.stdout}{probe.stderr}")
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.prepare_runtime_configs()
        self.record(
            "preflight",
            status="complete" if not self.args.dry_run else "dry_run",
            world_size=self.args.world_size,
            cuda_devices=list(self.gpus),
            resource_monitoring=False,
        )

    def prepare_runtime_configs(self) -> None:
        """Freeze inherited configs and launcher overrides into run-local JSON files."""

        sys.path.insert(0, str(self.repo))
        from mowe_wam.utils.config import load_config

        specifications = {
            "stage1": (self.args.stage1_config, self.args.stage1_max_steps),
            "stage2": (self.args.stage2_config, self.args.stage2_max_steps),
            "stage3": (self.args.stage3_config, self.args.stage3_max_steps),
        }
        config_root = self.run_root / "configs"
        for name, (source, max_steps) in specifications.items():
            config = load_config(str(self.repo / source))
            config["output_dir"] = str((self.stage_root / name).resolve())
            config["data"]["feature_store_path"] = str(self.args.feature_store.resolve())
            config["backbone"]["checkpoint"] = str(self.args.openvla_checkpoint.resolve())
            config["backbone"]["revision"] = self.args.openvla_revision
            config["teacher"]["checkpoint"] = str(self.args.dino_checkpoint.resolve())
            config["skill_expert_config"] = str(
                self.args.skill_config.resolve()
                if self.args.skill_config is not None
                else self.default_skill_config_path().resolve()
            )
            config["flow"]["num_inference_steps"] = self.args.flow_solver_steps
            config["training"].update(
                {
                    "max_steps": max_steps,
                    "grad_accumulation_steps": 1,
                    "precision": "bf16",
                }
            )
            validation = config.setdefault("validation", {})
            schedule_completion_ratio = float(
                config.get("action_condition", {}).get("nominal_end_ratio", 0.70)
            )
            if name == "stage3":
                schedule_completion_ratio = max(
                    schedule_completion_ratio,
                    float(
                        config.get("router", {}).get(
                            "predicted_route_end_ratio", 0.70
                        )
                    ),
                )
            effective_early_stop_min_steps = max(
                self.args.early_stop_min_steps,
                math.ceil(schedule_completion_ratio * max_steps),
            )
            validation.update(
                {
                    "enabled": True,
                    "eval_freq": self.args.validation_freq,
                    "num_batches": None,
                    "windows_per_episode": 1,
                    "min_unique_episodes": self.args.min_validation_episodes,
                    "modes": ["diagnostic", "deployment"],
                    "early_stopping": {
                        "enabled": True,
                        "metric": "total_loss",
                        "min_delta": self.args.early_stop_min_delta,
                        "patience": self.args.early_stop_patience,
                        "min_steps": effective_early_stop_min_steps,
                        "validation_mode": "deployment",
                        "require_schedule_completion": True,
                    },
                }
            )
            inherited_distributed = config["training"].get("distributed", {})
            config["training"]["distributed"] = {
                "enabled": "auto",
                "backend": "nccl",
                "timeout_seconds": int(
                    inherited_distributed.get("timeout_seconds", 1800)
                ),
                "broadcast_buffers": bool(
                    inherited_distributed.get("broadcast_buffers", False)
                ),
                "find_unused_parameters": bool(
                    inherited_distributed.get("find_unused_parameters", False)
                ),
                "resource_monitoring": False,
            }
            config["long_run_readiness"] = {
                "report_path": None,
                "mode": "disabled_no_resource_monitoring",
            }
            destination = config_root / f"{name}.json"
            atomic_json(destination, config)
            self.runtime_configs[name] = str(destination)

    def default_skill_config_path(self) -> Path:
        return self.report_root / "skill_experts_h16.json"

    def ensure_skill_config(self) -> Path:
        if self.args.skill_config is not None:
            skill_config = self.args.skill_config.resolve()
            if not skill_config.is_file():
                raise FileNotFoundError(f"Skill config does not exist: {skill_config}")
            return skill_config
        audit_path = self.report_root / "libero_rlds_h16_audit.json"
        if self.args.force_static_audits or not audit_path.is_file():
            self.run(
                "rlds_h16_audit",
                self.python_command(
                    "scripts/audit_flow_wam_rlds.py",
                    "--data-root", self.args.data_root,
                    "--skill-sidecar", self.args.skill_sidecar,
                    "--max-horizon", 16,
                    "--output", audit_path,
                ),
            )
        if self.args.dry_run and not audit_path.is_file():
            return self.report_root / "skill_experts_h16.json"
        report = load_json(audit_path)
        config = load_json(self.repo / "configs/mowe_wam/skill_experts.yaml")
        counts: Counter[str] = Counter()
        for suite in report["suites"]:
            counts.update(suite["parsed_skill_counts"])
        inverse = [1.0 / math.sqrt(max(int(counts[name]), 1)) for name in ALL_SKILLS]
        scale = len(inverse) / sum(inverse)
        config["source_path"] = str(self.args.skill_sidecar.resolve())
        config["audit"].update(
            {
                "report": str(audit_path.resolve()),
                "dataset_manifest_fingerprint_sha256": report["dataset_manifest_fingerprint_sha256"],
                "sidecar_fingerprint_sha256": report["sidecar_fingerprint_sha256"],
                "episodes": report["totals"]["episodes"],
                "transitions": report["totals"]["transitions"],
                "valid_windows_h16": report["totals"]["valid_windows"],
                "exact_episode_key_matches": report["totals"]["exact_episode_key_matches"],
                "annotation_step_match_ratio": report["totals"]["annotation_step_match_ratio"],
                "alignment_verified": False,
                "label_counts": {
                    name: int(counts[name]) for name in (*ALL_SKILLS, "unknown")
                },
            }
        )
        config["class_weights_inverse_sqrt"] = [round(value * scale, 6) for value in inverse]
        skill_config = self.report_root / "skill_experts_h16.json"
        atomic_json(skill_config, config)
        self.run(
            "skill_config_inspect",
            self.python_command(
                "scripts/inspect_skill_experts.py",
                "--data-root", self.args.data_root,
                "--skill-config", skill_config,
                "--sidecar", self.args.skill_sidecar,
            ),
        )
        return skill_config

    def ensure_static_evidence(self) -> Path:
        skill_config = self.ensure_skill_config()
        feature_report = self.report_root / "feature_store_audit.json"
        if self.args.force_static_audits or not report_matches_store(
            feature_report, self.args.feature_store, kind="feature"
        ):
            self.run(
                "feature_store_audit",
                self.python_command(
                    "scripts/audit_mowe_feature_store.py",
                    "--store", self.args.feature_store,
                    "--world-size", self.args.world_size,
                    "--seed", self.args.seed,
                    "--shuffle-block-size", self.args.shuffle_block_size,
                    "--verify-all-checksums",
                    "--sample-windows", 32,
                    "--max-window-imbalance-ratio", self.args.max_window_imbalance_ratio,
                    "--max-suite-imbalance-ratio", self.args.max_suite_imbalance_ratio,
                    "--max-skill-imbalance-ratio", self.args.max_skill_imbalance_ratio,
                    "--output", feature_report,
                ),
            )
        else:
            self.record(
                "feature_store_audit",
                status="complete",
                reused=True,
                report=str(feature_report),
            )
        equivalence_report = self.report_root / "feature_equivalence_100.json"
        evidence_inputs = self.state.get("static_evidence_inputs")
        current_inputs = {
            "store": str(self.args.feature_store.resolve()),
            "data_root": str(self.args.data_root.resolve()),
            "openvla_checkpoint": str(self.args.openvla_checkpoint.resolve()),
            "openvla_revision": self.args.openvla_revision,
            "dino_checkpoint": str(self.args.dino_checkpoint.resolve()),
            "skill_sidecar": str(self.args.skill_sidecar.resolve()),
            "samples": self.args.equivalence_samples,
        }
        if (
            self.args.force_static_audits
            or evidence_inputs != current_inputs
            or not report_matches_store(
                equivalence_report,
                self.args.feature_store,
                kind="equivalence",
                samples=self.args.equivalence_samples,
            )
        ):
            self.run(
                "feature_equivalence",
                self.python_command(
                    "scripts/audit_feature_store_equivalence.py",
                    "--config", "configs/mowe_wam/train_nominal_flow_wam.yaml",
                    "--store", self.args.feature_store,
                    "--data-root", self.args.data_root,
                    "--checkpoint", self.args.openvla_checkpoint,
                    "--backbone-revision", self.args.openvla_revision,
                    "--teacher-checkpoint", self.args.dino_checkpoint,
                    "--skill-sidecar", self.args.skill_sidecar,
                    "--samples", self.args.equivalence_samples,
                    "--seed", self.args.equivalence_seed,
                    "--stage", "nominal_flow_pretrain",
                    "--feature-atol", self.args.feature_atol,
                    "--output-atol", self.args.output_atol,
                    "--loss-atol", self.args.loss_atol,
                    "--output", equivalence_report,
                ),
            )
            self.state["static_evidence_inputs"] = current_inputs
            self.save_state()
        else:
            self.record(
                "feature_equivalence",
                status="complete",
                reused=True,
                report=str(equivalence_report),
            )
        return skill_config

    def common_train_args(self, config: str, output_dir: Path, max_steps: int, stop_step: int) -> list[Any]:
        return [
            "--config", config,
            "--feature-store", self.args.feature_store,
            "--checkpoint", self.args.openvla_checkpoint,
            "--backbone-revision", self.args.openvla_revision,
            "--teacher-checkpoint", self.args.dino_checkpoint,
            "--skill-expert-config", self.skill_config,
            "--output-dir", output_dir,
            "--max-steps", max_steps,
            "--stop-step", stop_step,
            "--grad-accumulation-steps", 1,
            "--flow-solver-steps", self.args.flow_solver_steps,
            "--precision", "bf16",
        ]

    def run_training_segment(
        self,
        *,
        task: str,
        script: str,
        config: str,
        output_dir: Path,
        max_steps: int,
        stop_step: int,
        save_freq: int,
        log_freq: int,
        expected_stage: str,
        init_checkpoint: Path | None = None,
        route_mode: str | None = None,
    ) -> None:
        checkpoint = output_dir / "checkpoint_latest.pt"
        early_stop_report = output_dir / "early_stopping.json"
        early_stop_contract = load_json(Path(config)).get("validation", {}).get(
            "early_stopping", {}
        )
        metadata = (
            checkpoint_metadata(checkpoint, expected_stage)
            if checkpoint.exists()
            else self.virtual_checkpoints.get(str(checkpoint))
        )
        if metadata is not None and int(metadata["step"]) >= stop_step:
            print(f"SKIP {task}: checkpoint already reached step {metadata['step']}", flush=True)
            return
        if metadata is not None and early_stop_report.is_file():
            report = load_json(early_stop_report)
            if (
                report.get("format") == "mowe_validation_loss_early_stop_v1"
                and report.get("stage") == expected_stage
                and bool(report.get("stopped_early", False))
                and int(report.get("step", -1)) == int(metadata["step"])
                and int(report.get("max_steps", -1)) == max_steps
                and report.get("metric") == early_stop_contract.get("metric")
                and float(report.get("min_delta", -1.0))
                == float(early_stop_contract.get("min_delta", -2.0))
                and int(report.get("patience", -1))
                == int(early_stop_contract.get("patience", -2))
                and int(report.get("min_steps", -1))
                == int(early_stop_contract.get("min_steps", -2))
                and report.get("validation_mode")
                == early_stop_contract.get("validation_mode", "deployment")
            ):
                print(
                    f"SKIP {task}: {expected_stage} already stopped early at step "
                    f"{metadata['step']}",
                    flush=True,
                )
                return
        args = self.common_train_args(config, output_dir, max_steps, stop_step)
        args += ["--save-freq", save_freq, "--log-freq", log_freq]
        if route_mode is not None:
            args += ["--route-mode", route_mode]
        if metadata is not None:
            args += ["--resume", checkpoint]
        elif init_checkpoint is not None:
            args += ["--init-wam", init_checkpoint]
        self.run(task, self.torchrun_command(script, *args))
        if self.args.dry_run:
            virtual_metadata = {
                "format": "flow_wam_skill_components_v2",
                "stage": expected_stage,
                "step": stop_step,
            }
            if metadata is not None:
                virtual_metadata["config"] = metadata.get("config", {})
            elif init_checkpoint is not None:
                predecessor_metadata = self.virtual_checkpoints.get(
                    str(init_checkpoint)
                )
                if predecessor_metadata is not None:
                    virtual_metadata["config"] = {
                        "initialization_contract": checkpoint_predecessor_identity(
                            predecessor_metadata
                        )
                    }
            self.virtual_checkpoints[str(checkpoint)] = virtual_metadata
        if not self.args.dry_run:
            completed = checkpoint_metadata(checkpoint, expected_stage)
            completed_step = int(completed["step"]) if completed is not None else -1
            stopped_early = False
            if completed is not None and early_stop_report.is_file():
                report = load_json(early_stop_report)
                stopped_early = (
                    report.get("format") == "mowe_validation_loss_early_stop_v1"
                    and report.get("stage") == expected_stage
                    and bool(report.get("stopped_early", False))
                    and int(report.get("step", -1)) == completed_step
                    and int(report.get("max_steps", -1)) == max_steps
                    and report.get("metric") == early_stop_contract.get("metric")
                    and float(report.get("min_delta", -1.0))
                    == float(early_stop_contract.get("min_delta", -2.0))
                    and int(report.get("patience", -1))
                    == int(early_stop_contract.get("patience", -2))
                    and int(report.get("min_steps", -1))
                    == int(early_stop_contract.get("min_steps", -2))
                    and report.get("validation_mode")
                    == early_stop_contract.get("validation_mode", "deployment")
                )
            if completed_step != stop_step and not stopped_early:
                raise RuntimeError(f"{task} did not produce expected step {stop_step}: {checkpoint}")

    def stage_stopped_early(
        self, stage_dir: Path, expected_stage: str, max_steps: int
    ) -> bool:
        checkpoint = stage_dir / "checkpoint_latest.pt"
        report_path = stage_dir / "early_stopping.json"
        metadata = checkpoint_metadata(checkpoint, expected_stage)
        if metadata is None or not report_path.is_file():
            return False
        report = load_json(report_path)
        config_name = {
            "nominal_flow_pretrain": "stage1",
            "expert_warmstart": "stage2",
            "joint": "stage3",
        }[expected_stage]
        early_stop_contract = load_json(
            Path(self.runtime_configs[config_name])
        ).get("validation", {}).get("early_stopping", {})
        return (
            report.get("format") == "mowe_validation_loss_early_stop_v1"
            and report.get("stage") == expected_stage
            and bool(report.get("stopped_early", False))
            and int(report.get("step", -1)) == int(metadata["step"])
            and int(report.get("max_steps", -1)) == max_steps
            and report.get("metric") == early_stop_contract.get("metric")
            and float(report.get("min_delta", -1.0))
            == float(early_stop_contract.get("min_delta", -2.0))
            and int(report.get("patience", -1))
            == int(early_stop_contract.get("patience", -2))
            and int(report.get("min_steps", -1))
            == int(early_stop_contract.get("min_steps", -2))
            and report.get("validation_mode")
            == early_stop_contract.get("validation_mode", "deployment")
        )

    def validate_stage1_quality(self, stage_dir: Path) -> dict[str, Any]:
        report_path = self.report_root / "stage1_quality_gate.json"
        if self.args.dry_run:
            report = {
                "format": "mowe_stage1_quality_gate_v2",
                "passed": None,
                "status": "dry_run",
            }
            atomic_json(report_path, report)
            self.record(
                "stage1_quality_gate", status="dry_run", report=str(report_path)
            )
            return report
        best_checkpoint = stage_dir / "checkpoint_best.pt"
        selected_checkpoint = (
            best_checkpoint
            if best_checkpoint.is_file()
            else stage_dir / "checkpoint_latest.pt"
        )
        selected_metadata = checkpoint_metadata(
            selected_checkpoint, "nominal_flow_pretrain"
        )
        if selected_metadata is None:
            raise RuntimeError(
                f"Stage 1 checkpoint is missing before quality validation: {selected_checkpoint}"
            )
        report = stage1_quality_gate(
            jsonl_rows(stage_dir / "validation_log.jsonl"),
            required_average_improvement=self.args.stage1_quality_average_improvement,
            required_min_horizon_improvement=self.args.stage1_quality_min_horizon_improvement,
            min_unique_episodes=self.args.min_validation_episodes,
            min_action_distance_gate=self.args.stage1_quality_min_action_gate,
            validation_step=int(selected_metadata["step"]),
        )
        report["checkpoint"] = str(selected_checkpoint)
        atomic_json(report_path, report)
        self.record(
            "stage1_quality_gate",
            status="complete" if report["passed"] else "failed",
            report=str(report_path),
            validation_step=report.get("validation_step"),
        )
        if not report["passed"]:
            raise RuntimeError(
                "Stage 1 deployment quality gate failed; Stage 2 is blocked. "
                f"See {report_path}: {report.get('errors', [])}"
            )
        return report

    def validate_stage_predecessor(
        self,
        metadata: dict[str, Any] | None,
        predecessor: Path,
        *,
        stage_name: str,
    ) -> None:
        if metadata is None:
            return
        predecessor_metadata = (
            checkpoint_metadata(predecessor)
            or self.virtual_checkpoints.get(str(predecessor))
        )
        if predecessor_metadata is None:
            raise RuntimeError(
                f"Cannot validate {stage_name} predecessor: {predecessor}"
            )
        expected = checkpoint_predecessor_identity(predecessor_metadata)
        observed = metadata.get("config", {}).get("initialization_contract")
        if observed != expected:
            raise RuntimeError(
                f"Existing {stage_name} checkpoint was initialized from a different "
                "predecessor. Preserve it and use a new --run-id/stage directory; "
                "do not resume it after the predecessor changed."
            )

    def validate_smoke(self, stage_dir: Path, stage: str) -> dict[str, Any]:
        train = jsonl_rows(stage_dir / "train_log.jsonl")
        validation = jsonl_rows(stage_dir / "validation_log.jsonl")
        if not train or not validation or not finite_tree(train) or not finite_tree(validation):
            raise RuntimeError(f"{stage} logs are missing or contain NaN/Inf.")
        result = {
            "stage": stage,
            "last_step": train[-1].get("step"),
            "logs_present": True,
            "all_logged_values_finite": True,
        }
        atomic_json(self.report_root / f"{stage}_smoke_gate.json", {"passed": True, **result})
        return result

    def train_stage1(self) -> Path:
        output = self.stage_root / "stage1"
        checkpoint = output / "checkpoint_latest.pt"
        metadata = (
            checkpoint_metadata(checkpoint, "nominal_flow_pretrain")
            if checkpoint.exists()
            else self.virtual_checkpoints.get(str(checkpoint))
        )
        step = int(metadata["step"]) if metadata else 0
        for target, save_freq, log_freq in ((2, 2, 1), (25, 25, 5), (100, 25, 5)):
            if step < target:
                self.run_training_segment(
                    task=f"ddp_stage1_{step}_{target}",
                    script="scripts/pretrain_nominal_flow_wam.py",
                    config=self.runtime_configs["stage1"],
                    output_dir=output,
                    max_steps=self.args.stage1_max_steps,
                    stop_step=target,
                    save_freq=save_freq,
                    log_freq=log_freq,
                    expected_stage="nominal_flow_pretrain",
                )
                step = target
        if not self.args.dry_run:
            self.validate_smoke(output, "stage1")
        if step < min(1000, self.args.stage1_max_steps):
            target = min(1000, self.args.stage1_max_steps)
            self.run_training_segment(
                task=f"ddp_stage1_{step}_{target}",
                script="scripts/pretrain_nominal_flow_wam.py",
                config=self.runtime_configs["stage1"],
                output_dir=output,
                max_steps=self.args.stage1_max_steps,
                stop_step=target,
                save_freq=100,
                log_freq=self.args.long_log_freq,
                expected_stage="nominal_flow_pretrain",
            )
            step = target
        if step < self.args.stage1_max_steps and not (
            not self.args.dry_run
            and self.stage_stopped_early(
                output, "nominal_flow_pretrain", self.args.stage1_max_steps
            )
        ):
            self.run_training_segment(
                task=f"ddp_stage1_{step}_{self.args.stage1_max_steps}",
                script="scripts/pretrain_nominal_flow_wam.py",
                config=self.runtime_configs["stage1"],
                output_dir=output,
                max_steps=self.args.stage1_max_steps,
                stop_step=self.args.stage1_max_steps,
                save_freq=self.args.long_save_freq,
                log_freq=self.args.long_log_freq,
                expected_stage="nominal_flow_pretrain",
            )
        best = output / "checkpoint_best.pt"
        return best if not self.args.dry_run and best.is_file() else checkpoint

    def train_stage2(self, predecessor: Path) -> Path:
        output = self.stage_root / "stage2"
        checkpoint = output / "checkpoint_latest.pt"
        metadata = (
            checkpoint_metadata(checkpoint, "expert_warmstart")
            if checkpoint.exists()
            else self.virtual_checkpoints.get(str(checkpoint))
        )
        self.validate_stage_predecessor(
            metadata, predecessor, stage_name="Stage 2"
        )
        step = int(metadata["step"]) if metadata else 0
        if step < 100:
            self.run_training_segment(
                task=f"ddp_stage2_{step}_100",
                script="scripts/warmstart_skill_flow_experts.py",
                config=self.runtime_configs["stage2"],
                output_dir=output,
                max_steps=self.args.stage2_max_steps,
                stop_step=100,
                save_freq=25,
                log_freq=5,
                expected_stage="expert_warmstart",
                init_checkpoint=predecessor,
                route_mode="oracle",
            )
            step = 100
        if not self.args.dry_run:
            self.validate_smoke(output, "stage2")
        if step < self.args.stage2_max_steps and not (
            not self.args.dry_run
            and self.stage_stopped_early(
                output, "expert_warmstart", self.args.stage2_max_steps
            )
        ):
            self.run_training_segment(
                task=f"ddp_stage2_{step}_{self.args.stage2_max_steps}",
                script="scripts/warmstart_skill_flow_experts.py",
                config=self.runtime_configs["stage2"],
                output_dir=output,
                max_steps=self.args.stage2_max_steps,
                stop_step=self.args.stage2_max_steps,
                save_freq=self.args.long_save_freq,
                log_freq=self.args.long_log_freq,
                expected_stage="expert_warmstart",
                route_mode="oracle",
            )
        best = output / "checkpoint_best.pt"
        return best if not self.args.dry_run and best.is_file() else checkpoint

    def train_stage3(self, predecessor: Path) -> Path:
        output = self.stage_root / "stage3"
        checkpoint = output / "checkpoint_latest.pt"
        metadata = (
            checkpoint_metadata(checkpoint, "joint")
            if checkpoint.exists()
            else self.virtual_checkpoints.get(str(checkpoint))
        )
        self.validate_stage_predecessor(
            metadata, predecessor, stage_name="Stage 3"
        )
        step = int(metadata["step"]) if metadata else 0
        if step < 100:
            self.run_training_segment(
                task=f"ddp_stage3_{step}_100",
                script="scripts/train_flow_wam_skill_moe.py",
                config=self.runtime_configs["stage3"],
                output_dir=output,
                max_steps=self.args.stage3_max_steps,
                stop_step=100,
                save_freq=25,
                log_freq=5,
                expected_stage="joint",
                init_checkpoint=predecessor,
            )
            step = 100
        if not self.args.dry_run:
            self.validate_smoke(output, "stage3")
        if step < self.args.stage3_max_steps and not (
            not self.args.dry_run
            and self.stage_stopped_early(output, "joint", self.args.stage3_max_steps)
        ):
            self.run_training_segment(
                task=f"ddp_stage3_{step}_{self.args.stage3_max_steps}",
                script="scripts/train_flow_wam_skill_moe.py",
                config=self.runtime_configs["stage3"],
                output_dir=output,
                max_steps=self.args.stage3_max_steps,
                stop_step=self.args.stage3_max_steps,
                save_freq=self.args.long_save_freq,
                log_freq=self.args.long_log_freq,
                expected_stage="joint",
            )
        best = output / "checkpoint_best.pt"
        return best if not self.args.dry_run and best.is_file() else checkpoint

    def execute(self) -> None:
        self.preflight()
        self.skill_config = self.ensure_static_evidence()
        stage1 = self.train_stage1()
        self.validate_stage1_quality(self.stage_root / "stage1")
        stage2 = self.train_stage2(stage1)
        stage3 = self.train_stage3(stage2)
        self.record(
            "pipeline",
            status="dry_run" if self.args.dry_run else "complete",
            completed_at=utc_now(),
            final_checkpoint=str(stage3),
        )
        print(f"\nMoWE training pipeline complete. Final checkpoint: {stage3}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="一键完成 MoWE 数据审计与正式 8 卡 Stage 1/2/3 训练。",
    )

    # 代码仓库根目录：换服务器后通常需要修改。
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_ROOT, help="MoWE 仓库根目录。")
    # 原始 LIBERO RLDS 根目录：用于重新生成跨服务器有效的 100-window 等价性证据。
    parser.add_argument("--data-root", type=Path, required=True, help="LIBERO RLDS 数据根目录。")
    # 已完成且 formal_training_ready=true 的 feature store；可挂载到不同绝对路径。
    parser.add_argument("--feature-store", type=Path, required=True, help="正式 mowe feature store 目录。")
    # 原始 openvla/openvla-7b 本地 snapshot；不得使用 LIBERO-finetuned 权重。
    parser.add_argument("--openvla-checkpoint", type=Path, required=True, help="原始 OpenVLA-7B snapshot。")
    # OpenVLA 的不可变 40 位 Hugging Face commit。
    parser.add_argument("--openvla-revision", required=True, help="OpenVLA 40 位 commit revision。")
    # DINOv2 teacher snapshot；等价性审计会用它重新编码真实窗口。
    parser.add_argument("--dino-checkpoint", type=Path, required=True, help="DINOv2-small snapshot。")
    # CoT skill sidecar JSON；必须与 feature store manifest 中的 fingerprint 对应。
    parser.add_argument("--skill-sidecar", type=Path, required=True, help="cot_file.json 路径。")
    # 可选的已审计 skill config；不传则自动运行 RLDS audit 并生成。
    parser.add_argument("--skill-config", type=Path, help="可选 skill_experts_h16.json；默认自动生成。")

    # MTP 输出根目录；兼容旧平台参数名 --run_root_dir。
    parser.add_argument("--run-root-dir", "--run_root_dir", type=Path, required=True, help="所有实验输出的父目录。")
    # 本次正式 lineage 的唯一 ID；兼容旧平台参数名 --run_id。
    parser.add_argument("--run-id", "--run_id", required=True, help="实验/lineage 名称，建议包含日期与模型版本。")
    # 报告目录；默认 <run-root-dir>/<run-id>/reports。
    parser.add_argument("--report-root", type=Path, help="数据审计和启动日志目录。")
    # Stage checkpoint 目录；默认 <run-root-dir>/<run-id>/ddp8。
    parser.add_argument("--stage-root", type=Path, help="Stage 1/2/3 checkpoint 根目录。")
    # 可选 Python 解释器；默认使用启动 start_mtp.py 的解释器。
    parser.add_argument("--python", type=Path, help="目标训练环境的 python 可执行文件。")

    # 正式合同固定 8 rank；保留参数便于平台显式展示，但不允许改成其他值。
    parser.add_argument("--world-size", type=int, default=8, help="正式 DDP world size，必须为 8。")
    # 逗号分隔的可见 GPU ID；数量必须等于 world size。
    parser.add_argument("--cuda-devices", default="0,1,2,3,4,5,6,7", help="CUDA_VISIBLE_DEVICES。")
    # 数据分配与训练的全局随机种子。
    parser.add_argument("--seed", type=int, default=7, help="feature assignment seed。")
    # shard-aware sampler 的 block size，必须与正式训练配置一致。
    parser.add_argument("--shuffle-block-size", type=int, default=256, help="sampler shuffle block size。")

    # 三个正式训练配置；路径相对于 repo-root。
    parser.add_argument("--stage1-config", default="configs/mowe_wam/ddp8_nominal_flow_wam_feature_store_formal.yaml", help="Stage 1 配置。")
    parser.add_argument("--stage2-config", default="configs/mowe_wam/ddp8_warmstart_skill_flow_feature_store.yaml", help="Stage 2 配置。")
    parser.add_argument("--stage3-config", default="configs/mowe_wam/ddp8_train_flow_wam_feature_store.yaml", help="Stage 3 配置。")
    # 三阶段完整 optimizer step 数；修改会改变实验合同。
    parser.add_argument("--stage1-max-steps", type=int, default=50000, help="Stage 1 总步数。")
    parser.add_argument("--stage2-max-steps", type=int, default=50000, help="Stage 2 总步数。")
    parser.add_argument("--stage3-max-steps", type=int, default=50000, help="Stage 3 总步数。")
    # Flow ODE solver steps；须与单卡验证和正式配置一致。
    parser.add_argument("--flow-solver-steps", type=int, default=4, help="Flow solver steps。")
    # 长训练 checkpoint/日志频率；中断恢复会使用最近 checkpoint。
    parser.add_argument("--long-save-freq", type=int, default=500, help="长训练 checkpoint 周期。")
    parser.add_argument("--long-log-freq", type=int, default=10, help="长训练日志周期。")
    # 三阶段统一按验证集 total_loss 判断平台期；未早停则各自跑满 max steps。
    parser.add_argument("--validation-freq", type=int, default=500, help="验证与早停判断周期（optimizer steps）。")
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4, help="刷新最佳 validation total_loss 所需的最小下降量。")
    parser.add_argument("--early-stop-patience", type=int, default=5, help="连续多少次验证改善不足后早停。")
    parser.add_argument("--early-stop-min-steps", type=int, default=5000, help="允许早停前每个阶段至少训练的步数。")
    parser.add_argument(
        "--min-validation-episodes",
        type=int,
        default=32,
        help="每次正式验证至少覆盖的不同 episode 数。",
    )
    parser.add_argument(
        "--stage1-quality-average-improvement",
        type=float,
        default=0.10,
        help="Stage 1 future smooth-L1 相对 copy-current 的三 horizon 最低平均改善。",
    )
    parser.add_argument(
        "--stage1-quality-min-horizon-improvement",
        type=float,
        default=0.0,
        help="Stage 1 任一 H=4/8/16 horizon 相对 copy-current 的最低改善。",
    )
    parser.add_argument(
        "--stage1-quality-min-action-gate",
        type=float,
        default=0.10,
        help="Stage 1 deployment validation 的最低平均 nominal-action distance gate。",
    )

    # 100-window raw/cache 等价性参数；不建议放宽默认容差。
    parser.add_argument("--equivalence-samples", type=int, default=100, help="等价性窗口数，正式至少 100。")
    parser.add_argument("--equivalence-seed", type=int, default=1701, help="等价性抽样 seed。")
    parser.add_argument("--feature-atol", type=float, default=0.03, help="feature gate 容差。")
    parser.add_argument("--output-atol", type=float, default=0.10, help="model output 容差。")
    parser.add_argument("--loss-atol", type=float, default=0.05, help="loss 容差。")
    # 8-rank 数据均衡上限；真实 audit 失败时不要仅为通过而放宽。
    parser.add_argument("--max-window-imbalance-ratio", type=float, default=1.25, help="window imbalance 上限。")
    parser.add_argument("--max-suite-imbalance-ratio", type=float, default=1.50, help="suite imbalance 上限。")
    parser.add_argument("--max-skill-imbalance-ratio", type=float, default=2.00, help="skill imbalance 上限。")

    # 强制重做与数据/store 相关的静态审计。
    parser.add_argument("--force-static-audits", action="store_true", help="强制重跑 RLDS/feature/equivalence 审计。")
    # 只打印完整命令和写 dry-run 状态，不启动审计或训练；可在单卡机器检查参数。
    parser.add_argument("--dry-run", action="store_true", help="仅验证路径并打印计划，不要求 8 张 GPU。")

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignored platform-injected arguments: {unknown}", file=sys.stderr)
    for name in (
        "data_root",
        "feature_store",
        "openvla_checkpoint",
        "dino_checkpoint",
        "skill_sidecar",
        "run_root_dir",
        "report_root",
        "stage_root",
        "skill_config",
        "python",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.expanduser())
    return args


def main() -> None:
    args = parse_args()
    print(json.dumps({"argv": sys.argv, "started_at": utc_now()}, ensure_ascii=False), flush=True)
    Launcher(args).execute()


if __name__ == "__main__":
    main()
