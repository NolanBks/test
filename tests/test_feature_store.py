from __future__ import annotations

import sys
import json
import copy
import io
from contextlib import redirect_stdout
import tempfile
import unittest
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
    validate_episode_assignment_reports,
)
from openvla_test_utils import synthetic_openvla_identity


@unittest.skipIf(np is None or torch is None, "NumPy and PyTorch are required")
class FeatureStoreTests(unittest.TestCase):
    def test_soak_slope_detects_linear_growth_but_ignores_plateau_noise(self):
        from scripts.soak_mowe_feature_store import (
            _linear_slope_per_1k_steps,
            _missing_required_metrics,
        )

        growing = [
            {"step": step, "cgroup_memory_anon_mib": 100.0 + 0.1 * step}
            for step in (1000, 2000, 3000, 4000)
        ]
        plateau = [
            {"step": step, "cgroup_memory_anon_mib": value}
            for step, value in zip(
                (1000, 2000, 3000, 4000), (100.0, 101.0, 99.0, 100.0)
            )
        ]
        self.assertAlmostEqual(
            _linear_slope_per_1k_steps(growing, "cgroup_memory_anon_mib"),
            100.0,
        )
        self.assertAlmostEqual(
            _linear_slope_per_1k_steps(plateau, "cgroup_memory_anon_mib"),
            -0.2,
        )
        missing = _missing_required_metrics(
            [
                {"step": 1, "anon": 1.0, "working": 2.0},
                {"step": 2, "anon": 1.0},
            ],
            {"anon", "working"},
        )
        self.assertEqual(missing, {"2": ["working"]})

    def _build_store(self, root: Path):
        writer = MoWEFeatureStoreWriter(
            root,
            source_contract={
                "rlds_manifest_fingerprint": "rlds-test",
                "skill_sidecar_fingerprint": "sidecar-test",
                "openvla_checkpoint": "manifest-openvla",
                "openvla_identity": synthetic_openvla_identity("feature-store"),
                "teacher_checkpoint": "manifest-dino",
                "expected_counts": {
                    "episode_count": 4,
                    "frame_count": 34,
                    "window_count": 26,
                },
                "joint_action_statistics": {
                    "q01": [-1.0] * 6 + [0.0],
                    "q99": [1.0] * 6 + [1.0],
                    "mask": [True] * 6 + [False],
                },
            },
            history_length=4,
            long_memory_slots=2,
            future_horizons=(1, 2),
            action_chunk_size=3,
            episodes_per_shard=2,
        )
        for episode_index, length in enumerate((7, 8, 9, 10)):
            steps = np.arange(length, dtype=np.float32)
            views = np.stack(
                [np.full((2, 8), value, dtype=np.float32) for value in steps], axis=0
            )
            dino = np.stack(
                [np.full((2, 3), 100 + value, dtype=np.float32) for value in steps], axis=0
            )
            actions = np.zeros((length, 7), dtype=np.float32)
            actions[:, 0] = steps
            actions[:, -1] = (steps.astype(np.int64) % 2).astype(np.float32)
            skills = (steps.astype(np.int8) % 7).astype(np.int8)
            writer.add_episode(
                episode_id=f"ep-{episode_index}",
                dataset_name="suite-a" if episode_index < 2 else "suite-b",
                partition="validation" if episode_index == 3 else "train",
                language=f"task-{episode_index % 2}",
                language_feature=np.full(8, episode_index % 2, dtype=np.float32),
                openvla_views=views,
                dino_tokens=dino,
                actions=actions,
                skills=skills,
            )
        return writer.finalize()

    def test_expected_count_mismatch_marks_store_nonformal(self):
        from mowe_wam.training.flow_runtime import resolve_feature_store_contract
        from mowe_wam.utils.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            writer = MoWEFeatureStoreWriter(
                root,
                source_contract={
                    "formal_training_ready": True,
                    "expected_counts": {
                        "episode_count": 2,
                        "frame_count": 8,
                        "window_count": 4,
                    },
                    "joint_action_statistics": {
                        "q01": [-1.0] * 6 + [0.0],
                        "q99": [1.0] * 6 + [1.0],
                        "mask": [True] * 6 + [False],
                    },
                    "openvla_checkpoint": "synthetic-openvla",
                    "openvla_identity": synthetic_openvla_identity("count-mismatch"),
                },
                future_horizons=(1,),
                action_chunk_size=2,
            )
            writer.add_episode(
                episode_id="only-one",
                dataset_name="suite-a",
                partition="train",
                language="pick",
                language_feature=np.ones(4, dtype=np.float32),
                openvla_views=np.ones((4, 2, 4), dtype=np.float32),
                dino_tokens=np.ones((4, 2, 3), dtype=np.float32),
                actions=np.zeros((4, 7), dtype=np.float32),
                skills=np.zeros(4, dtype=np.int8),
            )
            manifest = writer.finalize()
            self.assertFalse(manifest["formal_training_ready"])
            self.assertFalse(manifest["completion_contract"]["counts_match"])
            self.assertIn("expected_counts", audit_feature_store(root)["issues"])

            cfg = load_config("configs/mowe_wam/flow_wam_base.yaml")
            cfg["data"].update(
                {
                    "backend": "mowe_feature_store_v1",
                    "feature_store_path": str(root),
                    "history_length": 8,
                    "long_memory_slots": 4,
                    "future_horizons": [1],
                    "action_chunk_size": 2,
                }
            )
            cfg["teacher"].update({"spatial_tokens": 2, "target_dim": 3})
            with self.assertRaisesRegex(ValueError, "incomplete"):
                resolve_feature_store_contract(cfg)

    def test_manifest_binds_frozen_model_identifiers_when_config_is_tbd(self):
        from mowe_wam.training.flow_runtime import resolve_feature_store_contract
        from mowe_wam.utils.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_store(root)
            cfg = load_config("configs/mowe_wam/flow_wam_base.yaml")
            cfg["backbone"]["checkpoint"] = "TBD"
            cfg["teacher"].update(
                {"checkpoint": "TBD", "spatial_tokens": 2, "target_dim": 3}
            )
            cfg["data"].update(
                {
                    "backend": "mowe_feature_store_v1",
                    "feature_store_path": str(root),
                    "history_length": 4,
                    "long_memory_slots": 2,
                    "future_horizons": [1, 2],
                    "action_chunk_size": 3,
                }
            )
            resolve_feature_store_contract(cfg)
            self.assertEqual(cfg["backbone"]["checkpoint"], "manifest-openvla")
            self.assertEqual(cfg["teacher"]["checkpoint"], "manifest-dino")
            self.assertTrue(cfg["feature_store_contract"]["formal_training_ready"])
            self.assertTrue(
                cfg["feature_store_contract"]["completion_contract"]["counts_match"]
            )

    def test_feature_store_allows_teacher_snapshot_path_remount(self):
        from mowe_wam.training.flow_runtime import resolve_feature_store_contract
        from mowe_wam.utils.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_store(root)
            cfg = load_config("configs/mowe_wam/flow_wam_base.yaml")
            cfg["teacher"].update(
                {
                    "checkpoint": "/new/server/dinov2-small",
                    "spatial_tokens": 2,
                    "target_dim": 3,
                }
            )
            cfg["data"].update(
                {
                    "backend": "mowe_feature_store_v1",
                    "feature_store_path": str(root),
                    "history_length": 4,
                    "long_memory_slots": 2,
                    "future_horizons": [1, 2],
                    "action_chunk_size": 3,
                }
            )

            resolve_feature_store_contract(cfg)

            self.assertEqual(cfg["teacher"]["checkpoint"], "/new/server/dinov2-small")
            self.assertEqual(cfg["teacher"]["source_checkpoint"], "manifest-dino")

    def test_pending_episode_is_checksum_verified_and_resumed_before_shard_flush(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_contract = {
                "rlds_manifest_fingerprint": "resume-test",
                "joint_action_statistics": {
                    "q01": [-1.0] * 6 + [0.0],
                    "q99": [1.0] * 6 + [1.0],
                    "mask": [True] * 6 + [False],
                },
            }
            writer = MoWEFeatureStoreWriter(
                root,
                source_contract=source_contract,
                future_horizons=(1,),
                action_chunk_size=2,
                episodes_per_shard=8,
            )
            writer.add_episode(
                episode_id="pending-0",
                dataset_name="suite-a",
                partition="train",
                language="pick object",
                language_feature=np.ones(4, dtype=np.float32),
                openvla_views=np.ones((4, 2, 4), dtype=np.float32),
                dino_tokens=np.ones((4, 2, 3), dtype=np.float32),
                actions=np.zeros((4, 7), dtype=np.float32),
                skills=np.zeros(4, dtype=np.int8),
                source_file_key="/data/episode.hdf5",
                source_traj_index=3,
            )
            self.assertTrue((root / ".staging/pending.json").exists())

            resumed = MoWEFeatureStoreWriter(
                root,
                source_contract=source_contract,
                future_horizons=(1,),
                action_chunk_size=2,
                episodes_per_shard=8,
            )
            self.assertTrue(resumed.has_episode("pending-0"))
            self.assertEqual(
                resumed.source_episode_identities(),
                {("/data/episode.hdf5", 3)},
            )
            staging = next((root / ".staging").glob("episode-*.npz"))
            original = staging.read_bytes()
            staging.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                MoWEFeatureStoreWriter(
                    root,
                    source_contract=source_contract,
                    future_horizons=(1,),
                    action_chunk_size=2,
                    episodes_per_shard=8,
                )
            staging.write_bytes(original)
            manifest = resumed.finalize()
            self.assertEqual(manifest["episode_count"], 1)
            self.assertTrue(audit_feature_store(root, verify_all_checksums=True)["valid"])
            self.assertFalse((root / ".staging/pending.json").exists())

    def _build_runtime_store(
        self, root: Path, *, formal_training_ready: bool | None = None
    ):
        source_contract = {
            "rlds_manifest_fingerprint": "rlds-test",
            "skill_sidecar_fingerprint": "sidecar-test",
            "openvla_checkpoint": "synthetic-openvla",
            "openvla_identity": synthetic_openvla_identity("runtime-store"),
            "joint_action_statistics": {
                "q01": [-1.0] * 6 + [0.0],
                "q99": [1.0] * 6 + [1.0],
                "mask": [True] * 6 + [False],
            },
        }
        if formal_training_ready is not None:
            source_contract["formal_training_ready"] = formal_training_ready
        writer = MoWEFeatureStoreWriter(
            root,
            source_contract=source_contract,
            history_length=8,
            long_memory_slots=4,
            future_horizons=(1, 4, 8, 16),
            action_chunk_size=16,
            episodes_per_shard=2,
        )
        for episode_index in range(2):
            length = 18
            steps = np.arange(length, dtype=np.float32)
            views = np.stack(
                [np.full((2, 8), value, dtype=np.float32) for value in steps]
            )
            dino = np.stack(
                [np.full((2, 3), value, dtype=np.float32) for value in steps]
            )
            actions = np.zeros((length, 7), dtype=np.float32)
            actions[:, 0] = steps / length
            actions[:, -1] = (steps.astype(np.int64) % 2).astype(np.float32)
            writer.add_episode(
                episode_id=f"runtime-{episode_index}",
                dataset_name="suite-a",
                partition="train",
                language=f"runtime-task-{episode_index}",
                language_feature=np.full(8, episode_index, dtype=np.float32),
                openvla_views=views,
                dino_tokens=dino,
                actions=actions,
                skills=(steps.astype(np.int8) % 7),
            )
        return writer.finalize()

    def test_window_contract_and_no_tensorflow_hot_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._build_store(root)
            self.assertEqual(manifest["episode_count"], 4)
            tensorflow_before = "tensorflow" in sys.modules
            dataset = MoWEFeatureWindowDataset(root, partition="train")
            sample = dataset[2]
            self.assertEqual(tuple(sample["current_visual_views"].shape), (2, 8))
            self.assertEqual(tuple(sample["history_visual_views"].shape), (3, 2, 8))
            self.assertEqual(sample["history_mask"].tolist(), [False, True, True, True])
            self.assertEqual(sample["history_actions"][:, 0].tolist(), [0.0, 0.0, 1.0])
            self.assertEqual(sample["long_history_mask"].tolist(), [False, False])
            self.assertEqual(sample["target_actions"][:, 0].tolist(), [2.0, 3.0, 4.0])
            self.assertEqual(sample["future_latent_targets"][:, 0, 0].tolist(), [103.0, 104.0])
            self.assertEqual("tensorflow" in sys.modules, tensorflow_before)
            audit = audit_feature_store(root, verify_all_checksums=True)
            self.assertTrue(audit["valid"], audit["issues"])

    def test_episode_assignment_is_disjoint_complete_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_store(root)
            dataset = MoWEFeatureWindowDataset(root, partition="train")
            samplers = [
                EpisodeAwareDistributedSampler(dataset, rank=rank, world_size=2, seed=13)
                for rank in range(2)
            ]
            rank_episodes = [set(sampler.local_episode_indices) for sampler in samplers]
            self.assertFalse(rank_episodes[0] & rank_episodes[1])
            self.assertEqual(
                rank_episodes[0] | rank_episodes[1], set(dataset.partition_episode_indices)
            )
            reports = [sampler.assignment_report(include_skill_counts=True) for sampler in samplers]
            observed_skill_counts = {}
            for report in reports:
                for skill, count in report["target_skill_counts"].items():
                    observed_skill_counts[skill] = observed_skill_counts.get(skill, 0) + count
            self.assertEqual(
                observed_skill_counts,
                dataset.skill_counts_for_window_positions(range(len(dataset))),
            )
            self.assertEqual(
                observed_skill_counts,
                dataset.partition_target_skill_counts(),
            )
            validation = validate_episode_assignment_reports(
                dataset, reports, world_size=2
            )
            self.assertTrue(validation["episode_union_complete"])
            self.assertTrue(validation["target_skill_union_complete"])
            corrupted = copy.deepcopy(reports)
            corrupted[1]["episode_indices"].append(
                corrupted[0]["episode_indices"][0]
            )
            corrupted[1]["episode_count"] += 1
            with self.assertRaisesRegex(RuntimeError, "episode_overlap"):
                validate_episode_assignment_reports(
                    dataset, corrupted, world_size=2
                )
            first = samplers[0]
            iterator = iter(first)
            consumed = [next(iterator), next(iterator)]
            state = first.state_dict()
            self.assertEqual(state["order_strategy"], "shard_aware_block_shuffle_v1")
            self.assertEqual(state["shuffle_block_size"], 256)
            restored = EpisodeAwareDistributedSampler(dataset, rank=0, world_size=2, seed=13)
            restored.load_state_dict(state)
            remaining = list(restored)
            reference = EpisodeAwareDistributedSampler(dataset, rank=0, world_size=2, seed=13)
            self.assertEqual(consumed + remaining, list(reference))
            incompatible = EpisodeAwareDistributedSampler(
                dataset,
                rank=0,
                world_size=2,
                seed=13,
                shuffle_block_size=4,
            )
            with self.assertRaisesRegex(ValueError, "Sampler resume contract"):
                incompatible.load_state_dict(state)

    def test_precomputed_backbone_preserves_context_shapes(self):
        from mowe_wam.backbones import PrecomputedFeatureBackbone

        backbone = PrecomputedFeatureBackbone(8, device="cpu", dtype="float32")
        batch = {
            "current_visual_views": torch.zeros(2, 2, 8),
            "history_visual_views": torch.zeros(2, 3, 2, 8),
            "long_history_visual_views": torch.zeros(2, 2, 2, 8),
            "precomputed_language": torch.ones(2, 8),
        }
        output = backbone.extract_context_features(batch)
        self.assertEqual(tuple(output["current_visual_views"].shape), (2, 2, 8))
        self.assertEqual(tuple(output["language_tokens"].shape), (2, 1, 8))
        with self.assertRaisesRegex(ValueError, "training targets"):
            backbone.extract_context_features({**batch, "target_actions": torch.zeros(2, 3, 7)})

    def test_flow_runtime_uses_feature_store_without_openvla_model(self):
        from mowe_wam.training.flow_runtime import (
            build_flow_dataloader,
            build_flow_policy,
            resolve_feature_store_contract,
        )
        from mowe_wam.utils.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_runtime_store(root)
            cfg = load_config("configs/mowe_wam/flow_wam_base.yaml")
            cfg["backbone"].update(
                {
                    "mode": "precomputed_features",
                    "feature_source": "pre_action_context_cache",
                    "checkpoint": "synthetic-openvla",
                    "dtype": "float32",
                }
            )
            cfg["teacher"].update(
                {"spatial_tokens": 2, "target_dim": 3, "cache_path": None}
            )
            cfg["data"].update(
                {
                    "backend": "mowe_feature_store_v1",
                    "feature_store_path": str(root),
                    "history_length": 8,
                    "long_memory_slots": 4,
                    "future_horizons": [1, 4, 8, 16],
                    "action_chunk_size": 16,
                    "num_workers": 0,
                    "pin_memory": False,
                }
            )
            cfg["memory"].update({"hidden_dim": 16, "heads": 4})
            cfg["flow"].update({"hidden_dim": 16, "depth": 2, "num_inference_steps": 2})
            cfg["world_model"].update(
                {"hidden_dim": 16, "layers": 1, "heads": 4, "route_world_dim": 4}
            )
            cfg["router"]["hidden_dim"] = 8
            cfg["view_fusion"]["score_hidden_dim"] = 8
            cfg["training"].update(
                {"device": "cpu", "precision": "float32", "batch_size": 2}
            )
            resolve_feature_store_contract(cfg)
            model = build_flow_policy(cfg, include_teacher=False)
            self.assertFalse(hasattr(model.backbone, "model"))
            batch = next(iter(build_flow_dataloader(cfg, model)))
            output = model(
                batch,
                action_condition_mode="ground_truth",
                route_mode="oracle",
                flow_seed=3,
                compute_teacher_targets=True,
            )
            self.assertEqual(tuple(output["actions"].shape), (2, 16, 7))
            self.assertEqual(tuple(output["future_latent_targets"].shape), (2, 4, 2, 3))

    def test_partial_conversion_store_is_rejected_for_training(self):
        from mowe_wam.training.flow_runtime import resolve_feature_store_contract
        from mowe_wam.utils.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_runtime_store(root, formal_training_ready=False)

            cfg = load_config("configs/mowe_wam/flow_wam_base.yaml")
            cfg["data"].update(
                {
                    "backend": "mowe_feature_store_v1",
                    "feature_store_path": str(root),
                    "history_length": 8,
                    "long_memory_slots": 4,
                    "future_horizons": [1, 4, 8, 16],
                    "action_chunk_size": 16,
                }
            )
            cfg["teacher"].update({"spatial_tokens": 2, "target_dim": 3})
            with self.assertRaisesRegex(ValueError, "smoke-only"):
                resolve_feature_store_contract(cfg)

    def test_one_step_checkpoint_and_sampler_resume(self):
        from mowe_wam.training.flow_runtime import read_flow_checkpoint_metadata, run_flow_training
        from mowe_wam.utils.config import load_config

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = root / "store"
            output = root / "output"
            self._build_runtime_store(store)
            skill_cfg = load_config("configs/mowe_wam/skill_experts.yaml")
            skill_cfg["audit"].update(
                {
                    "dataset_manifest_fingerprint_sha256": "rlds-test",
                    "sidecar_fingerprint_sha256": "sidecar-test",
                    "alignment_verified": False,
                    "transitions": 2,
                    "label_counts": {"pick_grasp": 2},
                }
            )
            skill_path = root / "skills.json"
            skill_path.write_text(json.dumps(skill_cfg), encoding="utf-8")

            cfg = load_config("configs/mowe_wam/train_nominal_flow_wam.yaml")
            cfg["output_dir"] = str(output)
            cfg["skill_expert_config"] = str(skill_path)
            cfg["backbone"].update(
                {
                    "mode": "precomputed_features",
                    "feature_source": "pre_action_context_cache",
                    "checkpoint": "synthetic-openvla",
                    "dtype": "float32",
                }
            )
            cfg["teacher"].update(
                {"spatial_tokens": 2, "target_dim": 3, "cache_path": None}
            )
            cfg["data"].update(
                {
                    "backend": "mowe_feature_store_v1",
                    "feature_store_path": str(store),
                    "history_length": 8,
                    "long_memory_slots": 4,
                    "future_horizons": [1, 4, 8, 16],
                    "action_chunk_size": 16,
                    "num_workers": 0,
                    "pin_memory": False,
                }
            )
            cfg["memory"].update({"hidden_dim": 16, "heads": 4})
            cfg["flow"].update({"hidden_dim": 16, "depth": 2, "num_inference_steps": 2})
            cfg["world_model"].update(
                {"hidden_dim": 16, "layers": 1, "heads": 4, "route_world_dim": 4}
            )
            cfg["router"]["hidden_dim"] = 8
            cfg["view_fusion"]["score_hidden_dim"] = 8
            cfg["training"].update(
                {
                    "device": "cpu",
                    "precision": "float32",
                    "batch_size": 1,
                    "grad_accumulation_steps": 1,
                    "max_steps": 2,
                    "stop_step": 1,
                    "save_freq": 1,
                    "log_freq": 1,
                    "distributed": {"enabled": False, "memory_guard_fraction": 0.99},
                }
            )
            cfg["validation"]["enabled"] = False
            with redirect_stdout(io.StringIO()):
                checkpoint = run_flow_training(copy.deepcopy(cfg), stage="nominal_flow_pretrain")
            metadata = read_flow_checkpoint_metadata(checkpoint)
            self.assertEqual(metadata["step"], 1)
            self.assertEqual(metadata["sampler_state_by_rank"][0]["cursor"], 1)

            resume_cfg = copy.deepcopy(cfg)
            resume_cfg["training"]["stop_step"] = 2
            with redirect_stdout(io.StringIO()):
                resumed = run_flow_training(
                    resume_cfg, stage="nominal_flow_pretrain", resume=str(checkpoint)
                )
            resumed_metadata = read_flow_checkpoint_metadata(resumed)
            self.assertEqual(resumed_metadata["step"], 2)
            self.assertEqual(resumed_metadata["sampler_state_by_rank"][0]["cursor"], 2)
            rows = [
                json.loads(line)
                for line in (output / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["step"] for row in rows], [1, 2])


if __name__ == "__main__":
    unittest.main()
