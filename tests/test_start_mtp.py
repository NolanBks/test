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
    def test_execute_invokes_each_training_stage_once(self):
        launcher = object.__new__(start_mtp.Launcher)
        launcher.args = Namespace(dry_run=True)
        launcher.preflight = MagicMock()
        launcher.ensure_static_evidence = MagicMock(return_value=Path("skill.json"))
        launcher.ensure_node_evidence = MagicMock()
        launcher.train_stage1 = MagicMock(return_value=Path("stage1.pt"))
        launcher.train_stage2 = MagicMock(return_value=Path("stage2.pt"))
        launcher.train_stage3 = MagicMock(return_value=Path("stage3.pt"))
        launcher.record = MagicMock()

        with patch("builtins.print"):
            launcher.execute()

        launcher.train_stage1.assert_called_once_with()
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
                "--disable-system-monitoring",
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
                "feature_store_soak_8rank",
                "ddp_runtime_8gpu",
                "ddp_stage1_0_2",
                "ddp_stage1_2_25",
                "ddp_stage1_25_100",
                "ddp_stage1_100_1000",
                "ddp_stage1_1000_50000",
                "ddp_stage2_0_100",
                "ddp_stage2_100_50000",
                "ddp_stage3_0_100",
                "ddp_stage3_100_30000",
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
            self.assertEqual(config["training"]["distributed"]["backend"], "nccl")
            self.assertFalse(config["training"]["distributed"]["require_cgroup_metrics"])
            self.assertIn(
                "--disable-system-monitoring",
                tasks["feature_store_soak_8rank"]["command"],
            )
            self.assertIn(
                "--disable-system-monitoring",
                tasks["ddp_runtime_8gpu"]["command"],
            )
            self.assertIn(
                "--allow-missing-cgroup-metrics",
                tasks["readiness_stage1_step100"]["command"],
            )


if __name__ == "__main__":
    unittest.main()
