from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import start_mtp


class MtpLauncherTests(unittest.TestCase):
    def test_stage1_mechanism_checkpoint_prefers_gate_aligned_validation(self):
        from mowe_wam.training.flow_runtime import (
            stage1_mechanism_checkpoint_state,
        )

        def record(step, predicted, total_loss):
            return {
                "stage": "nominal_flow_pretrain",
                "step": step,
                "validation_mode": "deployment",
                "unique_episodes": 878,
                "sampling_contract": "episode_balanced_deterministic_v1",
                "metrics": {"total_loss": total_loss},
                "future_horizon_metrics": {
                    str(horizon): {
                        "smooth_l1": value,
                        "current_copy_smooth_l1": 1.0,
                    }
                    for horizon, value in zip((4, 8, 16), predicted)
                },
                "deployment_metrics": {
                    "mean_nominal_action_distance_gate": 0.5,
                    "current_view_weights_mean": [0.45, 0.55],
                },
            }

        state = stage1_mechanism_checkpoint_state(
            [
                record(70000, (1.40, 0.95, 0.70), 0.50),
                record(70500, (0.98, 0.82, 0.75), 0.60),
            ],
            min_steps=70000,
        )
        self.assertEqual(state["best_step"], 70500)
        self.assertTrue(state["current_is_best"])
        self.assertTrue(state["best"]["passes_mechanism_thresholds"])

    def test_launcher_requires_mechanism_checkpoint_when_configured(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage = root / "stage1"
            stage.mkdir()
            runtime = root / "stage1.json"
            runtime.write_text(
                json.dumps(
                    {
                        "validation": {
                            "mechanism_checkpoint": {"enabled": True}
                        }
                    }
                ),
                encoding="utf-8",
            )
            launcher = object.__new__(start_mtp.Launcher)
            launcher.runtime_configs = {"stage1": str(runtime)}
            with self.assertRaisesRegex(RuntimeError, "mechanism checkpoint"):
                launcher.selected_stage1_checkpoint(stage)

            checkpoint = stage / "checkpoint_best_mechanism.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint.with_suffix(".pt.metadata.json").write_text(
                json.dumps(
                    {
                        "format": "flow_wam_skill_components_v2",
                        "stage": "nominal_flow_pretrain",
                        "step": 70500,
                    }
                ),
                encoding="utf-8",
            )
            (stage / "mechanism_checkpoint.json").write_text(
                json.dumps(
                    {
                        "format": "mowe_stage1_mechanism_checkpoint_v1",
                        "best_step": 70500,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                launcher.selected_stage1_checkpoint(stage), checkpoint
            )

    def test_execute_invokes_each_training_stage_once(self):
        launcher = object.__new__(start_mtp.Launcher)
        launcher.args = Namespace(dry_run=True)
        launcher.preflight = MagicMock()
        launcher.ensure_static_evidence = MagicMock(return_value=Path("skill.json"))
        launcher.train_stage1 = MagicMock(return_value=Path("stage1.pt"))
        launcher.stage_root = Path("stages")
        launcher.validate_stage1_quality = MagicMock()
        launcher.train_stage2 = MagicMock(return_value=Path("stage2.pt"))
        launcher.train_stage3 = MagicMock(return_value=Path("stage3.pt"))
        launcher.record = MagicMock()

        with patch("builtins.print"):
            launcher.execute()

        launcher.train_stage1.assert_called_once_with()
        launcher.validate_stage1_quality.assert_called_once_with(Path("stages/stage1"))
        launcher.train_stage2.assert_called_once_with(Path("stage1.pt"))
        launcher.train_stage3.assert_called_once_with(Path("stage2.pt"))

    def test_dry_run_builds_complete_resume_chain(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            store = root / "store"
            openvla = root / "openvla"
            dino = root / "dino"
            for path in (data, store, openvla, dino):
                path.mkdir()
            sidecar = root / "cot_file.json"
            sidecar.write_text("{}", encoding="utf-8")
            skill = root / "skill.json"
            skill.write_text("{}", encoding="utf-8")
            output = root / "runs"
            command = [
                sys.executable,
                str(repo / "start_mtp.py"),
                "--repo-root",
                str(repo),
                "--data-root",
                str(data),
                "--feature-store",
                str(store),
                "--openvla-checkpoint",
                str(openvla),
                "--openvla-revision",
                "a" * 40,
                "--dino-checkpoint",
                str(dino),
                "--skill-sidecar",
                str(sidecar),
                "--skill-config",
                str(skill),
                "--run-root-dir",
                str(output),
                "--run-id",
                "dry-run",
                "--dry-run",
            ]

            completed = subprocess.run(
                command,
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            state = json.loads(
                (output / "dry-run/launcher_state.json").read_text(encoding="utf-8")
            )
            tasks = state["tasks"]
            for name in (
                "feature_store_audit",
                "feature_equivalence",
                "ddp_stage1_0_2",
                "ddp_stage1_2_25",
                "ddp_stage1_25_100",
                "ddp_stage1_100_1000",
                "ddp_stage1_1000_50000",
                "ddp_stage2_0_100",
                "ddp_stage2_100_50000",
                "ddp_stage3_0_100",
                "ddp_stage3_100_50000",
            ):
                self.assertEqual(tasks[name]["status"], "dry_run")
            self.assertIn(
                "--resume",
                tasks["ddp_stage1_2_25"]["command"],
            )
            self.assertIn(
                "--init-wam",
                tasks["ddp_stage2_0_100"]["command"],
            )
            self.assertIn(
                "--resume",
                tasks["ddp_stage2_100_50000"]["command"],
            )
            config = json.loads(
                (output / "dry-run/configs/stage1.json").read_text(encoding="utf-8")
            )
            self.assertEqual(config["training"]["max_steps"], 50000)
            self.assertEqual(config["training"]["grad_accumulation_steps"], 1)
            self.assertEqual(config["validation"]["eval_freq"], 500)
            self.assertEqual(
                config["validation"]["early_stopping"],
                {
                    "enabled": True,
                    "metric": "total_loss",
                    "min_delta": 0.0001,
                    "patience": 5,
                    "min_steps": 35000,
                    "validation_mode": "deployment",
                    "require_schedule_completion": True,
                },
            )
            self.assertIsNone(config["validation"]["num_batches"])
            self.assertEqual(config["validation"]["min_unique_episodes"], 32)
            self.assertEqual(
                config["validation"]["modes"], ["diagnostic", "deployment"]
            )
            self.assertEqual(tasks["stage1_quality_gate"]["status"], "dry_run")
            self.assertEqual(config["training"]["distributed"]["backend"], "nccl")
            self.assertFalse(
                config["training"]["distributed"]["resource_monitoring"]
            )
            self.assertNotIn(
                "require_cgroup_metrics", config["training"]["distributed"]
            )
            self.assertNotIn(
                "memory_guard_fraction", config["training"]["distributed"]
            )
            self.assertNotIn(
                "gpu_memory_guard_fraction", config["training"]["distributed"]
            )
            self.assertEqual(
                config["long_run_readiness"]["mode"],
                "disabled_no_resource_monitoring",
            )
            rendered_commands = "\n".join(
                " ".join(str(value) for value in task.get("command", []))
                for task in tasks.values()
            ).lower()
            for forbidden in (
                "soak_mowe_feature_store",
                "audit_ddp_runtime",
                "audit_long_training_readiness",
                "long-run-readiness",
                "system-monitoring",
                "cgroup",
                "memory-guard",
                "gpu-memory",
            ):
                self.assertNotIn(forbidden, rendered_commands)

    def test_stage_stopped_early_requires_matching_checkpoint_and_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            stage = Path(directory)
            checkpoint = stage / "checkpoint_latest.pt"
            checkpoint.write_bytes(b"checkpoint")
            checkpoint.with_suffix(".pt.metadata.json").write_text(
                json.dumps(
                    {
                        "format": "flow_wam_skill_components_v2",
                        "stage": "joint",
                        "step": 37500,
                    }
                ),
                encoding="utf-8",
            )
            (stage / "early_stopping.json").write_text(
                json.dumps(
                    {
                        "format": "mowe_validation_loss_early_stop_v1",
                        "stage": "joint",
                        "stopped_early": True,
                        "step": 37500,
                        "max_steps": 50000,
                        "metric": "total_loss",
                        "min_delta": 0.0001,
                        "patience": 5,
                        "min_steps": 35000,
                        "validation_mode": "deployment",
                    }
                ),
                encoding="utf-8",
            )
            launcher = object.__new__(start_mtp.Launcher)
            config = stage / "stage3.json"
            config.write_text(
                json.dumps(
                    {
                        "validation": {
                            "early_stopping": {
                                "metric": "total_loss",
                                "min_delta": 0.0001,
                                "patience": 5,
                                "min_steps": 35000,
                                "validation_mode": "deployment",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            launcher.runtime_configs = {"stage3": str(config)}

            self.assertTrue(launcher.stage_stopped_early(stage, "joint", 50000))
            self.assertFalse(launcher.stage_stopped_early(stage, "joint", 40000))

    def test_stage1_quality_gate_requires_diverse_deployment_improvement(self):
        record = {
            "stage": "nominal_flow_pretrain",
            "step": 35000,
            "validation_mode": "deployment",
            "unique_episodes": 78,
            "sampling_contract": "episode_balanced_deterministic_v1",
            "future_horizon_metrics": {
                str(horizon): {"smooth_l1": 0.8, "current_copy_smooth_l1": 1.0}
                for horizon in (4, 8, 16)
            },
            "deployment_metrics": {
                "mean_nominal_action_distance_gate": 0.5,
                "current_view_weights_mean": [0.45, 0.55],
            },
        }
        passed = start_mtp.stage1_quality_gate([record])
        self.assertTrue(passed["passed"], passed)
        failed = start_mtp.stage1_quality_gate(
            [{**record, "unique_episodes": 1}]
        )
        self.assertFalse(failed["passed"])
        self.assertIn("insufficient_unique_validation_episodes", failed["errors"])
        nonfinite = start_mtp.stage1_quality_gate(
            [
                {
                    **record,
                    "deployment_metrics": {
                        **record["deployment_metrics"],
                        "mean_nominal_action_distance_gate": float("nan"),
                    },
                }
            ]
        )
        self.assertFalse(nonfinite["passed"])
        self.assertIn("nominal_action_distance_gate_collapsed", nonfinite["errors"])

        short_horizon_tradeoff = {
            **record,
            "step": 45000,
            "future_horizon_metrics": {
                "4": {
                    "smooth_l1": 0.07406805,
                    "current_copy_smooth_l1": 0.07129669,
                },
                "8": {
                    "smooth_l1": 0.07277282,
                    "current_copy_smooth_l1": 0.11677783,
                },
                "16": {
                    "smooth_l1": 0.07853768,
                    "current_copy_smooth_l1": 0.18556504,
                },
            },
        }
        tolerated = start_mtp.stage1_quality_gate([short_horizon_tradeoff])
        self.assertTrue(tolerated["passed"], tolerated)
        self.assertGreater(tolerated["average_improvement"], 0.30)
        self.assertGreater(
            tolerated["fractional_improvement_over_copy_current"]["4"], -0.05
        )

        excessive_regression = {
            **short_horizon_tradeoff,
            "future_horizon_metrics": {
                **short_horizon_tradeoff["future_horizon_metrics"],
                "4": {"smooth_l1": 0.076, "current_copy_smooth_l1": 0.07129669},
            },
        }
        rejected = start_mtp.stage1_quality_gate([excessive_regression])
        self.assertFalse(rejected["passed"])
        self.assertGreater(rejected["average_improvement"], 0.10)
        self.assertIn(
            "one_or_more_horizons_below_allowed_improvement", rejected["errors"]
        )

    def test_existing_stage_must_match_selected_predecessor_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predecessor = root / "checkpoint_best.pt"
            predecessor.touch()
            predecessor_metadata = {
                "format": "flow_wam_skill_components_v2",
                "stage": "nominal_flow_pretrain",
                "step": 35000,
                "config": {"stage": "nominal_flow_pretrain", "training": {}},
                "data_contract": {"feature_store_contract": {"format": "test"}},
                "backbone_identifier": "backbone-test",
            }
            predecessor.with_suffix(".pt.metadata.json").write_text(
                json.dumps(predecessor_metadata), encoding="utf-8"
            )
            launcher = object.__new__(start_mtp.Launcher)
            matching = {
                "config": {
                    "initialization_contract": start_mtp.checkpoint_predecessor_identity(
                        predecessor_metadata
                    )
                }
            }
            launcher.validate_stage_predecessor(
                matching, predecessor, stage_name="Stage 2"
            )
            with self.assertRaisesRegex(RuntimeError, "different predecessor"):
                launcher.validate_stage_predecessor(
                    {"config": {"initialization_contract": None}},
                    predecessor,
                    stage_name="Stage 2",
                )


if __name__ == "__main__":
    unittest.main()
