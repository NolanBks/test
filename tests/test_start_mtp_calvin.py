from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class CalvinMtpLauncherTests(unittest.TestCase):
    def test_dry_run_covers_conversion_audits_and_three_stage_resume_chain(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "calvin_rlds"
            data.mkdir()
            for index in range(512):
                (data / f"calvin_abc-train.tfrecord-{index:05d}-of-00512").write_bytes(
                    b"synthetic-shard!"
                )
            openvla = root / "openvla"
            dino = root / "dino"
            openvla.mkdir()
            dino.mkdir()
            runs = root / "runs"
            store = root / "stores/calvin"
            command = [
                sys.executable,
                str(repo / "start_mtp_calvin.py"),
                "--repo-root",
                str(repo),
                "--dataset-root",
                str(data),
                "--feature-store",
                str(store),
                "--openvla-checkpoint",
                str(openvla),
                "--openvla-revision",
                "a" * 40,
                "--dino-checkpoint",
                str(dino),
                "--run-root-dir",
                str(runs),
                "--run-id",
                "calvin-dry-run",
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
            run = runs / "calvin-dry-run"
            state = json.loads((run / "launcher_state.json").read_text())
            self.assertEqual(state["benchmark"], "calvin_abc_d")
            tasks = state["tasks"]
            for name in (
                "calvin_rlds_h16_audit",
                "calvin_feature_conversion",
                "feature_store_audit",
                "feature_equivalence",
                "ddp_stage1_0_2",
                "ddp_stage1_2_25",
                "ddp_stage1_25_100",
                "ddp_stage1_100_1000",
                "ddp_stage1_1000_100000",
                "ddp_stage2_0_100",
                "ddp_stage2_100_50000",
                "ddp_stage3_0_100",
                "ddp_stage3_100_50000",
            ):
                self.assertEqual(tasks[name]["status"], "dry_run", name)
            self.assertIn(
                "scripts/convert_calvin_to_mowe_store.py",
                " ".join(tasks["calvin_feature_conversion"]["command"]),
            )
            self.assertIn(
                "scripts/audit_calvin_feature_store_equivalence.py",
                " ".join(tasks["feature_equivalence"]["command"]),
            )
            stage1 = json.loads((run / "configs/stage1.json").read_text())
            self.assertEqual(stage1["data"]["dataset_names"], ["calvin_abc_rlds"])
            self.assertEqual(stage1["training"]["max_steps"], 100000)
            self.assertEqual(
                stage1["world_prediction_loss"]["horizon_weights"],
                [0.25, 1.0, 1.0, 1.0],
            )
            self.assertEqual(
                stage1["world_prediction_loss"]["delta_rms_normalization"]["mode"],
                "batch_horizon",
            )
            self.assertEqual(
                stage1["world_prediction_loss"]["delta_cosine"]["mode"],
                "magnitude_aware",
            )
            self.assertTrue(
                stage1["validation"]["mechanism_checkpoint"]["enabled"]
            )
            self.assertEqual(
                stage1["validation"]["early_stopping"]["min_steps"], 70000
            )
            self.assertEqual(
                stage1["validation"]["early_stopping"]["patience"], 10
            )
            stage2 = json.loads((run / "configs/stage2.json").read_text())
            self.assertEqual(
                stage2["validation"]["early_stopping"]["patience"], 5
            )
            self.assertFalse(stage1["training"]["distributed"]["resource_monitoring"])
            self.assertEqual(
                stage1["skill_expert_config"],
                str((run / "reports/calvin_skill_experts.json").resolve()),
            )


if __name__ == "__main__":
    unittest.main()
