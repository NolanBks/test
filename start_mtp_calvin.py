#!/usr/bin/env python3
"""One-click CALVIN ABC RLDS conversion, audit, and 8-GPU Stage 1/2/3 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import start_mtp


SCRIPT_ROOT = Path(__file__).resolve().parent


class CalvinLauncher(start_mtp.Launcher):
    """CALVIN-specific data gates on top of the shared LIBERO training chain."""

    def default_skill_config_path(self) -> Path:
        return self.report_root / "calvin_skill_experts.json"

    def raw_data_quick_signature(self) -> str:
        digest = hashlib.sha256()
        for shard in sorted(
            self.args.data_root.glob("calvin_abc-train.tfrecord-*-of-00512")
        ):
            stat = shard.stat()
            metadata = (
                self.args.data_root
                / ".cache/huggingface/download"
                / f"{shard.name}.metadata"
            )
            checksum = None
            if metadata.is_file():
                lines = metadata.read_text(encoding="utf-8").splitlines()
                if len(lines) >= 2 and len(lines[1].strip()) == 64:
                    checksum = lines[1].strip()
            digest.update(shard.name.encode("utf-8"))
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(
                (checksum or f"mtime_ns:{stat.st_mtime_ns}").encode("ascii")
            )
        return digest.hexdigest()

    def save_state(self) -> None:
        self.state["updated_at"] = start_mtp.utc_now()
        self.state["run_id"] = self.args.run_id
        self.state["benchmark"] = "calvin_abc_d"
        self.state["paths"] = {
            "repo_root": str(self.repo),
            "dataset_root": str(self.args.data_root),
            "feature_store": str(self.args.feature_store),
            "openvla_checkpoint": str(self.args.openvla_checkpoint),
            "openvla_revision": self.args.openvla_revision,
            "dino_checkpoint": str(self.args.dino_checkpoint),
            "run_root": str(self.run_root),
        }
        start_mtp.atomic_json(self.state_path, self.state)

    def preflight(self) -> None:
        required = {
            "repository": self.repo,
            "CALVIN ABC RLDS root": self.args.data_root,
            "OpenVLA snapshot": self.args.openvla_checkpoint,
            "DINO snapshot": self.args.dino_checkpoint,
        }
        for label, path in required.items():
            if not path.exists():
                raise FileNotFoundError(f"{label} does not exist: {path}")
        shards = sorted(self.args.data_root.glob("calvin_abc-train.tfrecord-*-of-*"))
        expected_names = {
            f"calvin_abc-train.tfrecord-{index:05d}-of-00512"
            for index in range(512)
        }
        if {path.name for path in shards} != expected_names or any(
            path.stat().st_size < 16 for path in shards
        ):
            raise ValueError(
                "CALVIN ABC RLDS preflight requires 512 non-empty "
                "calvin_abc-train TFRecord shards."
            )
        if self.args.python is not None and not self.args.python.is_file():
            raise FileNotFoundError(f"Python executable does not exist: {self.args.python}")
        for relative in (
            self.args.stage1_config,
            self.args.stage2_config,
            self.args.stage3_config,
            "scripts/audit_calvin_training_data.py",
            "scripts/convert_calvin_to_mowe_store.py",
            "scripts/audit_mowe_feature_store.py",
            "scripts/audit_calvin_feature_store_equivalence.py",
        ):
            if not (self.repo / relative).is_file():
                raise FileNotFoundError(f"Required repository file is missing: {relative}")
        if len(self.gpus) != self.args.world_size or len(set(self.gpus)) != self.args.world_size:
            raise ValueError("--cuda-devices must contain exactly --world-size unique IDs.")
        if self.args.world_size != 8:
            raise ValueError("The formal CALVIN contract requires --world-size 8.")
        revision = self.args.openvla_revision.lower()
        if len(revision) != 40 or any(value not in "0123456789abcdef" for value in revision):
            raise ValueError("--openvla-revision must be a 40-character hexadecimal commit.")
        if self.args.equivalence_samples < 100:
            raise ValueError("Formal CALVIN training requires at least 100 equivalence windows.")
        if min(
            self.args.stage1_max_steps,
            self.args.stage2_max_steps,
            self.args.stage3_max_steps,
        ) < 100:
            raise ValueError("Every formal stage must include the 100-step smoke gate.")
        if self.args.validation_freq < 1:
            raise ValueError("--validation-freq must be positive.")
        if not math.isfinite(self.args.early_stop_min_delta) or self.args.early_stop_min_delta < 0:
            raise ValueError("--early-stop-min-delta must be non-negative.")
        if self.args.early_stop_patience < 1 or self.args.early_stop_min_steps < 100:
            raise ValueError("Invalid early-stopping patience/minimum steps.")
        if self.args.min_validation_episodes < 2:
            raise ValueError("--min-validation-episodes must be at least 2.")
        if self.args.encode_batch_size < 1 or self.args.episodes_per_shard < 1:
            raise ValueError("Conversion batch/shard sizes must be positive.")
        if self.args.skill_config is not None and not self.args.skill_config.is_file():
            raise FileNotFoundError(f"Skill config does not exist: {self.args.skill_config}")
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
                raise RuntimeError(
                    f"Eight-GPU preflight failed: {probe.stdout}{probe.stderr}"
                )
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.report_root.mkdir(parents=True, exist_ok=True)
        self.stage_root.mkdir(parents=True, exist_ok=True)
        self.args.feature_store.parent.mkdir(parents=True, exist_ok=True)
        self.prepare_runtime_configs()
        self.record(
            "preflight",
            status="dry_run" if self.args.dry_run else "complete",
            benchmark="calvin_abc_d",
            dataset_format="rlds",
            shards=512,
            world_size=8,
            cuda_devices=list(self.gpus),
            resource_monitoring=False,
        )

    def ensure_skill_config(self) -> Path:
        audit_path = self.report_root / "calvin_abc_rlds_h16_audit.json"
        generated_skill = self.default_skill_config_path()
        raw_signature = self.raw_data_quick_signature()
        if (
            self.args.force_static_audits
            or self.state.get("raw_data_quick_signature") != raw_signature
            or not audit_path.is_file()
            or (
                self.args.skill_config is None and not generated_skill.is_file()
            )
        ):
            command: list[Any] = [
                "--dataset-root",
                self.args.data_root,
                "--dataset-format",
                "rlds",
                "--min-segment-length",
                17,
                "--output",
                audit_path,
            ]
            if self.args.skill_config is None:
                command += ["--skill-config-output", generated_skill]
            self.run(
                "calvin_rlds_h16_audit",
                self.python_command("scripts/audit_calvin_training_data.py", *command),
            )
            self.state["raw_data_quick_signature"] = raw_signature
            self.save_state()
        if self.args.skill_config is not None:
            return self.args.skill_config.resolve()
        if self.args.dry_run and not generated_skill.is_file():
            return generated_skill
        if not generated_skill.is_file():
            raise RuntimeError(f"CALVIN skill config was not generated: {generated_skill}")
        return generated_skill

    def ensure_formal_feature_store(self, skill_config: Path) -> None:
        manifest = self.args.feature_store / "manifest.json"
        ready = False
        if manifest.is_file():
            payload = start_mtp.load_json(manifest)
            source_names = set(
                payload.get("source_contract", {}).get("dataset_names", [])
            )
            ready = (
                payload.get("formal_training_ready") is True
                and source_names == {"calvin_abc_rlds"}
            )
        if ready and not self.args.force_conversion:
            self.record(
                "calvin_feature_conversion",
                status="complete",
                reused=True,
                manifest=str(manifest),
            )
            return
        command: list[Any] = [
            "--config",
            self.args.stage1_config,
            "--benchmark-config",
            self.args.benchmark_config,
            "--dataset-root",
            self.args.data_root,
            "--dataset-format",
            "rlds",
            "--checkpoint",
            self.args.openvla_checkpoint,
            "--backbone-revision",
            self.args.openvla_revision,
            "--teacher-checkpoint",
            self.args.dino_checkpoint,
            "--output",
            self.args.feature_store,
            "--audit-output",
            self.report_root / "calvin_abc_rlds_h16_audit.json",
            "--skill-config-output",
            skill_config,
            "--encode-batch-size",
            self.args.encode_batch_size,
            "--episodes-per-shard",
            self.args.episodes_per_shard,
            "--device",
            self.args.conversion_device,
            "--precision",
            "bf16",
        ]
        self.run(
            "calvin_feature_conversion",
            self.python_command("scripts/convert_calvin_to_mowe_store.py", *command),
        )
        if not self.args.dry_run:
            if not manifest.is_file():
                raise RuntimeError("CALVIN conversion did not publish manifest.json.")
            payload = start_mtp.load_json(manifest)
            if payload.get("formal_training_ready") is not True:
                raise RuntimeError(
                    "CALVIN feature store is not formal_training_ready; inspect completion_contract."
                )

    def ensure_static_evidence(self) -> Path:
        skill_config = self.ensure_skill_config()
        self.ensure_formal_feature_store(skill_config)
        feature_report = self.report_root / "calvin_feature_store_audit.json"
        if self.args.force_static_audits or not start_mtp.report_matches_store(
            feature_report, self.args.feature_store, kind="feature"
        ):
            self.run(
                "feature_store_audit",
                self.python_command(
                    "scripts/audit_mowe_feature_store.py",
                    "--store",
                    self.args.feature_store,
                    "--world-size",
                    8,
                    "--seed",
                    self.args.seed,
                    "--shuffle-block-size",
                    self.args.shuffle_block_size,
                    "--verify-all-checksums",
                    "--sample-windows",
                    32,
                    "--max-window-imbalance-ratio",
                    self.args.max_window_imbalance_ratio,
                    "--max-suite-imbalance-ratio",
                    self.args.max_suite_imbalance_ratio,
                    "--max-skill-imbalance-ratio",
                    self.args.max_skill_imbalance_ratio,
                    "--output",
                    feature_report,
                ),
            )
        else:
            self.record(
                "feature_store_audit",
                status="complete",
                reused=True,
                report=str(feature_report),
            )
        equivalence_report = self.report_root / "calvin_feature_equivalence_100.json"
        current_inputs = {
            "benchmark": "calvin_abc_d",
            "store": str(self.args.feature_store.resolve()),
            "dataset_root": str(self.args.data_root.resolve()),
            "openvla_checkpoint": str(self.args.openvla_checkpoint.resolve()),
            "openvla_revision": self.args.openvla_revision,
            "dino_checkpoint": str(self.args.dino_checkpoint.resolve()),
            "samples": self.args.equivalence_samples,
            "raw_data_quick_signature": self.raw_data_quick_signature(),
        }
        if (
            self.args.force_static_audits
            or self.state.get("static_evidence_inputs") != current_inputs
            or not start_mtp.report_matches_store(
                equivalence_report,
                self.args.feature_store,
                kind="equivalence",
                samples=self.args.equivalence_samples,
            )
        ):
            self.run(
                "feature_equivalence",
                self.python_command(
                    "scripts/audit_calvin_feature_store_equivalence.py",
                    "--config",
                    self.runtime_configs["stage1"],
                    "--benchmark-config",
                    self.args.benchmark_config,
                    "--store",
                    self.args.feature_store,
                    "--dataset-root",
                    self.args.data_root,
                    "--dataset-format",
                    "rlds",
                    "--checkpoint",
                    self.args.openvla_checkpoint,
                    "--backbone-revision",
                    self.args.openvla_revision,
                    "--teacher-checkpoint",
                    self.args.dino_checkpoint,
                    "--samples",
                    self.args.equivalence_samples,
                    "--seed",
                    self.args.equivalence_seed,
                    "--stage",
                    "nominal_flow_pretrain",
                    "--feature-atol",
                    self.args.feature_atol,
                    "--output-atol",
                    self.args.output_atol,
                    "--loss-atol",
                    self.args.loss_atol,
                    "--output",
                    equivalence_report,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "一键完成 CALVIN ABC RLDS 全量审计/feature-store 转换、100-window "
            "等价性和正式 8 卡 Stage 1→2→3 训练。"
        ),
    )
    parser.add_argument("--repo-root", type=Path, default=SCRIPT_ROOT)
    parser.add_argument(
        "--dataset-root",
        "--data-root",
        dest="data_root",
        type=Path,
        default=SCRIPT_ROOT / "dataset/Calvin_rlds",
    )
    parser.add_argument("--feature-store", type=Path, required=True)
    parser.add_argument("--openvla-checkpoint", type=Path, required=True)
    parser.add_argument("--openvla-revision", required=True)
    parser.add_argument("--dino-checkpoint", type=Path, required=True)
    parser.add_argument("--skill-config", type=Path)
    parser.add_argument(
        "--benchmark-config", default="configs/mowe_wam/calvin_abc_d.yaml"
    )
    parser.add_argument("--run-root-dir", "--run_root_dir", type=Path, required=True)
    parser.add_argument("--run-id", "--run_id", required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--cuda-devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--shuffle-block-size", type=int, default=256)
    parser.add_argument(
        "--stage1-config",
        default="configs/mowe_wam/ddp8_calvin_nominal_flow_feature_store.yaml",
    )
    parser.add_argument(
        "--stage2-config",
        default="configs/mowe_wam/ddp8_calvin_warmstart_skill_flow_feature_store.yaml",
    )
    parser.add_argument(
        "--stage3-config",
        default="configs/mowe_wam/ddp8_calvin_joint_flow_feature_store.yaml",
    )
    parser.add_argument("--stage1-max-steps", type=int, default=50000)
    parser.add_argument("--stage2-max-steps", type=int, default=50000)
    parser.add_argument("--stage3-max-steps", type=int, default=50000)
    parser.add_argument("--flow-solver-steps", type=int, default=4)
    parser.add_argument("--long-save-freq", type=int, default=500)
    parser.add_argument("--long-log-freq", type=int, default=10)
    parser.add_argument("--validation-freq", type=int, default=500)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--early-stop-min-steps", type=int, default=5000)
    parser.add_argument("--min-validation-episodes", type=int, default=32)
    parser.add_argument(
        "--stage1-quality-average-improvement", type=float, default=0.10
    )
    parser.add_argument(
        "--stage1-quality-min-horizon-improvement", type=float, default=0.0
    )
    parser.add_argument("--stage1-quality-min-action-gate", type=float, default=0.10)
    parser.add_argument("--equivalence-samples", type=int, default=100)
    parser.add_argument("--equivalence-seed", type=int, default=2701)
    parser.add_argument("--feature-atol", type=float, default=0.03)
    parser.add_argument("--output-atol", type=float, default=0.10)
    parser.add_argument("--loss-atol", type=float, default=0.05)
    parser.add_argument("--max-window-imbalance-ratio", type=float, default=1.25)
    parser.add_argument("--max-suite-imbalance-ratio", type=float, default=1.50)
    parser.add_argument("--max-skill-imbalance-ratio", type=float, default=2.00)
    parser.add_argument("--encode-batch-size", type=int, default=16)
    parser.add_argument("--episodes-per-shard", type=int, default=96)
    parser.add_argument("--conversion-device", default="cuda:0")
    parser.add_argument("--force-conversion", action="store_true")
    parser.add_argument("--force-static-audits", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Warning: ignored platform-injected arguments: {unknown}", file=sys.stderr)
    for name in (
        "repo_root",
        "data_root",
        "feature_store",
        "openvla_checkpoint",
        "dino_checkpoint",
        "skill_config",
        "run_root_dir",
        "report_root",
        "stage_root",
        "python",
    ):
        value = getattr(args, name, None)
        if value is not None:
            setattr(args, name, value.expanduser())
    return args


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            {"argv": sys.argv, "benchmark": "calvin_abc_d", "started_at": start_mtp.utc_now()},
            ensure_ascii=False,
        ),
        flush=True,
    )
    CalvinLauncher(args).execute()


if __name__ == "__main__":
    main()
