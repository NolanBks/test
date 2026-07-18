from __future__ import annotations

import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

try:
    import numpy as np
    import torch
except ModuleNotFoundError:
    np = None
    torch = None

from mowe_wam.data.feature_store import (
    EpisodeAwareDistributedSampler,
    MoWEFeatureStoreWriter,
    MoWEFeatureWindowDataset,
    audit_feature_store,
)
from mowe_wam.training.long_run_readiness import audit_long_run_readiness
from mowe_wam.training.flow_runtime import (
    enforce_long_run_readiness,
    resolve_feature_store_contract,
    validate_long_run_readiness_attestation,
)
from mowe_wam.utils.config import load_config
from openvla_test_utils import synthetic_openvla_identity


@unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are required")
class LongRunReadinessTests(unittest.TestCase):
    def test_feature_configs_remove_raw_backend_fields_and_stage2_constant_losses(self):
        feature_configs = (
            "configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml",
            "configs/mowe_wam/ddp8_warmstart_skill_flow_feature_store.yaml",
            "configs/mowe_wam/ddp8_train_flow_wam_feature_store.yaml",
            "configs/mowe_wam/ddp8_calvin_nominal_flow_feature_store.yaml",
            "configs/mowe_wam/ddp8_calvin_warmstart_skill_flow_feature_store.yaml",
            "configs/mowe_wam/ddp8_calvin_joint_flow_feature_store.yaml",
        )
        for path in feature_configs:
            with self.subTest(path=path):
                cfg = load_config(path)
                self.assertEqual(cfg["data"]["backend"], "mowe_feature_store_v1")
                self.assertIsNone(cfg["data"]["data_root"])
                self.assertIsNone(cfg["data"]["skill_sidecar_path"])
                self.assertIsNone(cfg["data"]["tf_frame_parallel_calls"])
                self.assertEqual(cfg["data"]["sampler_shuffle_block_size"], 256)
                self.assertEqual(
                    cfg["long_run_readiness"]["max_unattested_steps"], 100
                )
        for path in (
            "configs/mowe_wam/warmstart_skill_flow_experts.yaml",
            "configs/mowe_wam/ddp8_warmstart_skill_flow_feature_store.yaml",
            "configs/mowe_wam/ddp8_calvin_warmstart_skill_flow_feature_store.yaml",
        ):
            with self.subTest(stage2=path):
                weights = load_config(path)["loss_weights"]
                self.assertEqual(weights["flow_nominal"], 0.0)
                self.assertEqual(weights["gripper_bce"], 0.0)
                self.assertEqual(weights["load_balance"], 0.0)
                self.assertGreater(weights["flow_expert"], 0.0)
                self.assertGreater(weights["route"], 0.0)

    def _fixture(self, root: Path):
        store = root / "store"
        source_contract = {
            "formal_training_ready": True,
            "expected_counts": {
                "episode_count": 9,
                "frame_count": 162,
                "window_count": 18,
            },
            "rlds_manifest_fingerprint": "readiness-rlds",
            "dataset_names": ["libero_spatial_no_noops"],
            "skill_sidecar_fingerprint": "readiness-sidecar",
            "skill_sidecar_metadata": {
                "fingerprint_sha256": "readiness-sidecar"
            },
            "openvla_checkpoint": "readiness-openvla",
            "openvla_identity": synthetic_openvla_identity("readiness"),
            "teacher_checkpoint": "readiness-dino",
            "joint_action_statistics": {
                "q01": [-1.0] * 6 + [0.0],
                "q99": [1.0] * 6 + [1.0],
                "mask": [True] * 6 + [False],
            },
        }
        writer = MoWEFeatureStoreWriter(
            store,
            source_contract=source_contract,
            history_length=8,
            long_memory_slots=4,
            future_horizons=(1, 4, 8, 16),
            action_chunk_size=16,
            episodes_per_shard=3,
        )
        for episode_index in range(9):
            length = 18
            steps = np.arange(length, dtype=np.float32)
            actions = np.zeros((length, 7), dtype=np.float32)
            actions[:, 0] = steps / length
            actions[:, -1] = (steps.astype(np.int64) % 2).astype(np.float32)
            writer.add_episode(
                episode_id=f"readiness-{episode_index}",
                dataset_name=f"suite-{episode_index % 4}",
                partition="validation" if episode_index == 8 else "train",
                language=f"task-{episode_index}",
                language_feature=np.full(8, episode_index, dtype=np.float32),
                openvla_views=np.full(
                    (length, 2, 8), episode_index, dtype=np.float32
                ),
                dino_tokens=np.full(
                    (length, 2, 3), episode_index, dtype=np.float32
                ),
                actions=actions,
                skills=(steps.astype(np.int8) % 7),
            )
        writer.finalize()

        dataset = MoWEFeatureWindowDataset(store, partition="train")
        samplers = [
            EpisodeAwareDistributedSampler(
                dataset, rank=rank, world_size=8, seed=7
            )
            for rank in range(8)
        ]
        reports = [
            sampler.assignment_report(include_skill_counts=True)
            for sampler in samplers
        ]
        expected_skills = dataset.skill_counts_for_window_positions(
            range(len(dataset))
        )
        observed_skills: Counter[str] = Counter()
        for report in reports:
            observed_skills.update(report["target_skill_counts"])
        feature_audit = audit_feature_store(store, verify_all_checksums=True)
        feature_audit["assignment"] = {
            "world_size": 8,
            "reports": reports,
            "episode_union_complete": True,
            "episode_overlap": [],
            "fingerprints_agree": True,
            "target_skill_union_complete": dict(observed_skills) == expected_skills,
            "configured_imbalance_limits": {
                "windows": 1.25,
                "suites": 4.0,
                "skills": 1.25,
            },
            "imbalance_checks": {
                "windows": True,
                "suites": True,
                "skills": True,
            },
        }
        feature_audit["valid"] = True

        equivalence = {
            "format": "mowe_feature_store_equivalence_v1",
            "benchmark": "libero",
            "store": str(store.resolve()),
            "openvla_identity_sha256": source_contract["openvla_identity"][
                "identity_sha256"
            ],
            "passed": True,
            "compared_samples": 100,
            "missing_pairs": [],
            "comparison_contract": {
                "name": "mask_aware_training_metric_v1",
            },
            "masks_match": True,
            "max_feature_gate_error": 0.01,
            "max_output_gate_error": 0.01,
            "max_loss_gate_error": 0.01,
            "tolerances": {
                "feature_atol": 0.03,
                "output_atol": 0.10,
                "loss_atol": 0.05,
            },
        }
        node_identity = {
            "hostname": "target-a100-node",
            "boot_id": "boot-123",
            "cgroup_membership_sha256": "a" * 64,
            "cgroup_memory_max_mib": 512 * 1024.0,
        }
        gpu_identities = [
            {
                "local_rank": rank,
                "device": f"cuda:{rank}",
                "name": "NVIDIA A100-SXM4-40GB",
                "total_memory_mib": 40536.0,
                "compute_capability": [8, 0],
            }
            for rank in range(8)
        ]
        soak = {
            "format": "mowe_feature_store_soak_v1",
            "store": str(store.resolve()),
            "passed": True,
            "runtime_identity": {
                "node": node_identity,
                "rank_count": 8,
                "accelerators": [],
            },
            "limits": {
                "max_anon_growth_mib": 512.0,
                "max_working_set_growth_mib": 2048.0,
                "max_anon_slope_mib_per_1k_steps": 64.0,
                "max_working_set_slope_mib_per_1k_steps": 256.0,
                "min_post_warmup_samples": 3,
            },
            "reports": [
                {
                    "rank": rank,
                    "world_size": 8,
                    "steps": 10_000,
                    "passed": True,
                    "sampler": reports[rank],
                    "tensorflow_imported": False,
                    "missing_required_metrics": {},
                    "anon_growth_mib": 1.0,
                    "working_set_growth_mib": 2.0,
                    "anon_slope_mib_per_1k_steps": 0.1,
                    "working_set_slope_mib_per_1k_steps": 0.2,
                    "cgroup_event_deltas": {
                        "cgroup_event_oom": 0,
                        "cgroup_event_oom_kill": 0,
                    },
                }
                for rank in range(8)
            ],
        }
        ddp_runtime = {
            "format": "flow_wam_ddp_runtime_audit_v1",
            "distributed_contract": {
                "world_size": 8,
                "effective_global_batch": 8,
            },
            "ranks": [{"rank": rank, "local_rank": rank} for rank in range(8)],
            "runtime_identity": {
                "node": node_identity,
                "rank_count": 8,
                "accelerators": gpu_identities,
            },
            "checks": {
                "rank_union_complete": True,
                "local_ranks_unique": True,
                "all_devices_bound": True,
            },
            "resource_guard_thresholds": {
                "cgroup_working_set_fraction": 0.80,
                "gpu_peak_allocated_fraction": 0.85,
            },
        }
        skill_cfg = load_config("configs/mowe_wam/skill_experts.yaml")
        skill_cfg["audit"].update(
            {
                "dataset_manifest_fingerprint_sha256": "readiness-rlds",
                "sidecar_fingerprint_sha256": "readiness-sidecar",
                "alignment_verified": False,
                "transitions": 7,
                "label_counts": {
                    "pick_grasp": 1,
                    "place_release": 1,
                    "move_transport": 1,
                    "open_close": 1,
                    "turn_rotate": 1,
                    "push_pull": 1,
                    "null_finish": 1,
                },
            }
        )
        skill_path = root / "skills.json"
        skill_path.write_text(json.dumps(skill_cfg), encoding="utf-8")

        cfg = load_config(
            "configs/mowe_wam/ddp8_nominal_flow_wam_feature_store.yaml"
        )
        cfg["backbone"]["checkpoint"] = "readiness-openvla"
        cfg["teacher"].update(
            {
                "checkpoint": "readiness-dino",
                "spatial_tokens": 2,
                "target_dim": 3,
            }
        )
        cfg["skill_expert_config"] = str(skill_path)
        return cfg, store, feature_audit, equivalence, soak, ddp_runtime

    def test_complete_evidence_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = self._fixture(Path(directory))
            report = audit_long_run_readiness(
                fixture[0],
                store=fixture[1],
                feature_audit=fixture[2],
                equivalence_report=fixture[3],
                soak_report=fixture[4],
                ddp_runtime_audit=fixture[5],
            )
            self.assertTrue(report["passed"], report["errors"])
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(report["effective_global_batch"], 8)
            self.assertEqual(len(report["launch_contract_sha256"]), 64)

            runtime_cfg = copy.deepcopy(fixture[0])
            runtime_cfg["data"]["feature_store_path"] = str(fixture[1])
            resolve_feature_store_contract(runtime_cfg)
            runtime_cfg["skill_experts_resolved"] = load_config(
                runtime_cfg["skill_expert_config"]
            )
            validate_long_run_readiness_attestation(
                report,
                runtime_cfg,
                stage="nominal_flow_pretrain",
                world_size=8,
                checkpoint_metadata=None,
                checkpoint_mode="none",
                runtime_identity=report["runtime_identity"],
            )
            restarted_identity = copy.deepcopy(report["runtime_identity"])
            restarted_identity["node"]["boot_id"] = "boot-after-restart"
            with self.assertRaisesRegex(ValueError, "different node, boot, cgroup"):
                validate_long_run_readiness_attestation(
                    report,
                    runtime_cfg,
                    stage="nominal_flow_pretrain",
                    world_size=8,
                    checkpoint_metadata=None,
                    checkpoint_mode="none",
                    runtime_identity=restarted_identity,
                )

            bounded_cfg = copy.deepcopy(runtime_cfg)
            bounded_cfg["long_run_readiness"]["report_path"] = None
            bounded = enforce_long_run_readiness(
                bounded_cfg,
                stage="nominal_flow_pretrain",
                world_size=8,
                start_step=0,
                stop_step=100,
                checkpoint_metadata=None,
                checkpoint_mode="none",
                runtime_identity=report["runtime_identity"],
            )
            self.assertEqual(bounded["mode"], "bounded_smoke")
            self.assertEqual(bounded["unattested_lineage_steps"], 100)
            with self.assertRaisesRegex(ValueError, "readiness-report"):
                enforce_long_run_readiness(
                    bounded_cfg,
                    stage="nominal_flow_pretrain",
                    world_size=8,
                    start_step=0,
                    stop_step=101,
                    checkpoint_metadata=None,
                    checkpoint_mode="none",
                    runtime_identity=report["runtime_identity"],
                )
            segmented_checkpoint = {
                "stage": "nominal_flow_pretrain",
                "step": 100,
                "config": copy.deepcopy(bounded_cfg),
            }
            with self.assertRaisesRegex(ValueError, "unattested lineage steps"):
                enforce_long_run_readiness(
                    copy.deepcopy(bounded_cfg),
                    stage="nominal_flow_pretrain",
                    world_size=8,
                    start_step=100,
                    stop_step=101,
                    checkpoint_metadata=segmented_checkpoint,
                    checkpoint_mode="resume",
                    runtime_identity=report["runtime_identity"],
                )

            report_path = Path(directory) / "readiness.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            runtime_cfg["long_run_readiness"]["report_path"] = str(report_path)
            attested = enforce_long_run_readiness(
                runtime_cfg,
                stage="nominal_flow_pretrain",
                world_size=8,
                start_step=0,
                stop_step=101,
                checkpoint_metadata=None,
                checkpoint_mode="none",
                runtime_identity=report["runtime_identity"],
            )
            self.assertEqual(attested["mode"], "attested_long_run")

            changed_cfg = copy.deepcopy(runtime_cfg)
            changed_cfg["training"]["learning_rates"]["world_model"] *= 2
            with self.assertRaisesRegex(ValueError, "launch contract differs"):
                enforce_long_run_readiness(
                    changed_cfg,
                    stage="nominal_flow_pretrain",
                    world_size=8,
                    start_step=0,
                    stop_step=101,
                    checkpoint_metadata=None,
                    checkpoint_mode="none",
                    runtime_identity=report["runtime_identity"],
                )

    def test_explicit_degraded_system_monitoring_passes_without_cgroup_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, store, feature_audit, equivalence, soak, runtime = self._fixture(
                Path(directory)
            )
            cfg["training"]["distributed"]["require_cgroup_metrics"] = False
            disabled_node = {"system_monitoring_disabled": True}
            soak["cgroup_monitoring_enabled"] = False
            soak["resource_monitoring_mode"] = "degraded_gpu_and_process_unavailable"
            soak["runtime_identity"]["node"] = disabled_node
            for item in soak["reports"]:
                item.update(
                    {
                        "cgroup_monitoring_enabled": False,
                        "anon_growth_mib": None,
                        "working_set_growth_mib": None,
                        "anon_slope_mib_per_1k_steps": None,
                        "working_set_slope_mib_per_1k_steps": None,
                        "cgroup_event_deltas": {},
                    }
                )
            runtime["cgroup_monitoring_enabled"] = False
            runtime["resource_monitoring_mode"] = "gpu_only"
            runtime["runtime_identity"]["node"] = disabled_node

            report = audit_long_run_readiness(
                cfg,
                store=store,
                feature_audit=feature_audit,
                equivalence_report=equivalence,
                soak_report=soak,
                ddp_runtime_audit=runtime,
                allow_missing_cgroup_metrics=True,
            )

            self.assertTrue(report["passed"], report["errors"])
            self.assertFalse(report["system_monitoring_enabled"])

    def test_missing_formal_evidence_and_world_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, store, feature_audit, equivalence, soak, ddp_runtime = self._fixture(
                Path(directory)
            )
            feature_audit = copy.deepcopy(feature_audit)
            feature_audit["assignment"]["configured_imbalance_limits"][
                "skills"
            ] = None
            equivalence = copy.deepcopy(equivalence)
            equivalence["compared_samples"] = 99
            equivalence["openvla_identity_sha256"] = "f" * 64
            equivalence["masks_match"] = False
            soak = copy.deepcopy(soak)
            soak["reports"][7]["tensorflow_imported"] = True
            soak["reports"][6]["sampler"]["shuffle_block_size"] = 4
            checkpoint = {
                "stage": "nominal_flow_pretrain",
                "step": 100,
                "config": {
                    "training": {
                        "batch_size": 8,
                        "grad_accumulation_steps": 1,
                    }
                },
                "distributed_contract": {
                    "world_size": 1,
                    "effective_global_batch": 8,
                },
            }
            report = audit_long_run_readiness(
                cfg,
                store=store,
                feature_audit=feature_audit,
                equivalence_report=equivalence,
                soak_report=soak,
                ddp_runtime_audit=ddp_runtime,
                checkpoint_metadata=checkpoint,
                allow_world_size_change=False,
            )
            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["reviewed_imbalance_limits"])
            self.assertFalse(report["checks"]["equivalence_gate"])
            self.assertFalse(report["checks"]["equivalence_identity"])
            self.assertFalse(report["checks"]["soak_gate"])
            self.assertFalse(report["checks"]["checkpoint_contract"])
            self.assertTrue(
                any("allow-world-size-change" in value for value in report["errors"])
            )

    def test_legacy_equivalence_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, store, feature_audit, equivalence, soak, ddp_runtime = self._fixture(
                Path(directory)
            )
            legacy = copy.deepcopy(equivalence)
            legacy.pop("comparison_contract")
            legacy.pop("masks_match")
            legacy.pop("max_feature_gate_error")

            report = audit_long_run_readiness(
                cfg,
                store=store,
                feature_audit=feature_audit,
                equivalence_report=legacy,
                soak_report=soak,
                ddp_runtime_audit=ddp_runtime,
            )

            self.assertFalse(report["passed"])
            self.assertFalse(report["checks"]["equivalence_gate"])

    def test_launch_contract_ignores_teacher_remount_path(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg, store, feature_audit, equivalence, soak, ddp_runtime = self._fixture(
                Path(directory)
            )
            first = audit_long_run_readiness(
                cfg,
                store=store,
                feature_audit=feature_audit,
                equivalence_report=equivalence,
                soak_report=soak,
                ddp_runtime_audit=ddp_runtime,
            )
            remounted = copy.deepcopy(cfg)
            remounted["teacher"]["checkpoint"] = "/another/server/dinov2-small"
            second = audit_long_run_readiness(
                remounted,
                store=store,
                feature_audit=feature_audit,
                equivalence_report=equivalence,
                soak_report=soak,
                ddp_runtime_audit=ddp_runtime,
            )

            self.assertTrue(first["passed"], first["errors"])
            self.assertTrue(second["passed"], second["errors"])
            self.assertEqual(
                first["launch_contract_sha256"], second["launch_contract_sha256"]
            )


if __name__ == "__main__":
    unittest.main()
