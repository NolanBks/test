from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from mowe_wam.data.expert_skill_labels import ExpertSkillSidecar, label_directive
from mowe_wam.data.cot_skill_sidecar import (
    COT_SKILL_MARKER,
    exclude_source_episodes,
    shard_episodic_dataset,
    source_episode_key,
    split_cot_skill_marker,
    split_mowe_transport_markers,
    tensorflow_source_episode_key,
)
from mowe_wam.evaluation.libero_temporal_policy import (
    VariablePrefixActionQueue,
    canonical_action_to_libero,
)
from mowe_wam.training.schedules import action_condition_probability, temporal_router_schedule
from mowe_wam.backbones.openvla_oft_adapter import OpenVLAOFTAdapter
from mowe_wam.data.libero_sequence_dataset import episode_partition
from mowe_wam.training.distributed import (
    DistributedContext,
    effective_global_batch,
    enforce_cgroup_memory_guard,
    enforce_gpu_memory_guard,
    enforce_no_new_oom_events,
    enforce_resource_metric_contract,
)
from mowe_wam.training.flow_runtime import (
    aggregate_distributed_records,
    distributed_episode_overlap,
    validate_backbone_identifier,
    validate_distributed_resume_contract,
)
from openvla_test_utils import synthetic_openvla_identity


class SkillLabelTests(unittest.TestCase):
    def test_post_overlay_rank_sharding_is_disjoint_and_complete(self):
        class FakeDataset:
            def __init__(self, records):
                self.records = list(records)

            def shard(self, world_size, rank):
                return FakeDataset(self.records[rank::world_size])

        records = [(index, f"label-{index}") for index in range(31)]
        shards = [
            shard_episodic_dataset(FakeDataset(records), rank=rank, world_size=8).records
            for rank in range(8)
        ]
        flattened = [record for shard in shards for record in shard]
        self.assertEqual(sorted(flattened), records)
        for left in range(8):
            for right in range(left + 1, 8):
                self.assertFalse(set(shards[left]) & set(shards[right]))

    def test_episode_partition_is_stable_and_disjoint(self):
        episode_ids = [f"suite:episode-{index}" for index in range(1000)]
        first = {
            value: episode_partition(value, validation_fraction=0.1, split_seed=17)
            for value in episode_ids
        }
        second = {
            value: episode_partition(value, validation_fraction=0.1, split_seed=17)
            for value in reversed(episode_ids)
        }
        self.assertEqual(first, second)
        train = {key for key, value in first.items() if value == "train"}
        validation = {key for key, value in first.items() if value == "validation"}
        self.assertFalse(train & validation)
        self.assertEqual(train | validation, set(episode_ids))
        self.assertGreater(len(validation), 50)
        self.assertLess(len(validation), 150)

    def test_leading_verb_taxonomy(self):
        cases = {
            "Pick up the cup.": 0,
            "Place the cup.": 1,
            "Move toward the plate.": 2,
            "Close the drawer.": 3,
            "Rotate the knob.": 4,
            "Pull the drawer.": 5,
            "Finally, finish the task.": 6,
            "Observe the scene.": -1,
        }
        for directive, expected in cases.items():
            self.assertEqual(label_directive(directive)[0], expected)

    def test_sidecar_preserves_per_timestep_boundaries(self):
        root = "/tmp/libero/libero_spatial/task_demo.hdf5"
        payload = {
            f"{root}_0_0": "<think>Pick up the cup.</think>",
            f"{root}_0_1": "<think>Move toward the plate.</think>",
            f"{root}_0_2": "<think>Place the cup.</think>",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cot_file.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            sidecar = ExpertSkillSidecar(path)
            record = sidecar.episode("libero_spatial_no_noops", 0, expected_length=4)
        self.assertEqual(record["labels"], [0, 2, 1, -1])
        self.assertEqual(record["sources"], ["raw_annotation"] * 3 + ["unknown"])
        self.assertFalse(sidecar.metadata["alignment_verified"])

    def test_transport_marker_is_removed_before_model_input(self):
        instruction, label = split_cot_skill_marker(f"put the cup away{COT_SKILL_MARKER}2")
        self.assertEqual(instruction, "put the cup away")
        self.assertEqual(label, 2)

    def test_tensorflow_overlay_uses_dlimp_global_trajectory_index(self):
        try:
            import tensorflow as tf
            from mowe_wam.data.cot_skill_sidecar import TensorFlowCotSkillOverlay
        except ModuleNotFoundError:
            self.skipTest("tensorflow is not installed")
        root = "/tmp/libero/task_demo.hdf5"
        payload = {
            f"{root}_6_0": "Approach the cup.",
            f"{root}_6_1": "Pick up the cup.",
            f"{root}_6_2": "Place the cup.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cot_file.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            overlay = TensorFlowCotSkillOverlay(path)
            wrapped = overlay.wrap_standardizer(lambda value: value)
            output = wrapped(
                {
                    "action": tf.zeros((3, 7), dtype=tf.float32),
                    "language_instruction": tf.constant(["task"] * 3),
                    "_traj_index": tf.constant([6, 6, 6], dtype=tf.int64),
                    "traj_metadata": {
                        "episode_metadata": {"file_path": tf.constant([root] * 3)}
                    },
                }
            )
        labels = [split_cot_skill_marker(value.numpy())[1] for value in output["language_instruction"]]
        self.assertEqual(labels, [2, 0, 1])
        self.assertEqual(output["_mowe_source_traj_index"].numpy().tolist(), [6, 6, 6])
        self.assertEqual(
            [value.decode("utf-8") for value in output["_mowe_source_file_key"].numpy()],
            [root, root, root],
        )
        instruction, label, source_file, source_index = split_mowe_transport_markers(
            output["language_instruction"][0].numpy()
        )
        self.assertEqual((instruction, label), ("task", 2))
        self.assertEqual((source_file, source_index), (root, 6))
        self.assertTrue(overlay.metadata["requires_deterministic_rlds_order"])
        self.assertEqual(overlay.metadata["required_num_parallel_reads"], 16)
        self.assertEqual(
            tensorflow_source_episode_key(output).numpy().decode("utf-8"),
            source_episode_key(root, 6),
        )
        episodic = tf.data.Dataset.from_tensor_slices(
            {
                "_mowe_source_file_key": tf.constant([[root], ["/tmp/other.hdf5"]]),
                "_mowe_source_traj_index": tf.constant([[6], [9]], dtype=tf.int64),
            }
        )
        filtered = exclude_source_episodes(
            episodic, {source_episode_key(root, 6)}
        )
        remaining = [
            tensorflow_source_episode_key(value).numpy().decode("utf-8")
            for value in filtered
        ]
        self.assertEqual(remaining, [source_episode_key("/tmp/other.hdf5", 9)])


class DistributedContractTests(unittest.TestCase):
    def test_evaluation_backbone_must_match_checkpoint_binding(self):
        identity = synthetic_openvla_identity("evaluation")
        metadata = {
            "backbone_identifier": identity["identity_sha256"],
            "backbone_identity": identity,
        }
        validate_backbone_identifier(
            metadata,
            "/models/openvla-7b",
            requested_identity=identity,
        )
        with self.assertRaisesRegex(ValueError, "differs"):
            validate_backbone_identifier(
                metadata,
                "/models/another-openvla",
                requested_identity=synthetic_openvla_identity("other"),
            )
        with self.assertRaisesRegex(ValueError, "does not bind"):
            validate_backbone_identifier(
                {}, "/models/openvla-7b", requested_identity=identity
            )

    def test_cgroup_key_value_parser_ignores_malformed_lines(self):
        from mowe_wam.training.distributed import _parse_cgroup_key_values

        self.assertEqual(
            _parse_cgroup_key_values("anon 1024\nfile 2048\ninvalid\nmax nope\n"),
            {"anon": 1024, "file": 2048},
        )

    def test_effective_global_batch_matches_single_gpu_accumulation(self):
        single = {"training": {"batch_size": 1, "grad_accumulation_steps": 8}}
        ddp = {"training": {"batch_size": 1, "grad_accumulation_steps": 1}}
        self.assertEqual(effective_global_batch(single, 1), 8)
        self.assertEqual(effective_global_batch(ddp, 8), 8)

    def test_resource_guards_enforce_cgroup_gpu_and_oom_limits(self):
        context = DistributedContext(False, 0, 0, 1, "none", "cpu")
        with self.assertRaisesRegex(RuntimeError, "metrics are missing"):
            enforce_resource_metric_contract(
                context,
                {},
                require_cgroup=True,
                require_gpu=False,
            )
        enforce_resource_metric_contract(
            context,
            {
                "cgroup_memory_current_mib": 40.0,
                "cgroup_memory_max_mib": 100.0,
                "cgroup_memory_working_set_mib": 30.0,
                "cgroup_event_oom": 0,
                "cgroup_event_oom_kill": 0,
            },
            require_cgroup=True,
            require_gpu=False,
        )
        enforce_cgroup_memory_guard(
            context,
            {"cgroup_memory_working_set_mib": 79.0, "cgroup_memory_max_mib": 100.0},
            0.80,
        )
        with self.assertRaisesRegex(RuntimeError, "Cgroup working-set"):
            enforce_cgroup_memory_guard(
                context,
                {"cgroup_memory_working_set_mib": 80.0, "cgroup_memory_max_mib": 100.0},
                0.80,
            )
        enforce_gpu_memory_guard(
            context,
            {"cuda_peak_allocated_mib": 84.0, "cuda_total_mib": 100.0},
            0.85,
        )
        with self.assertRaisesRegex(RuntimeError, "GPU peak"):
            enforce_gpu_memory_guard(
                context,
                {"cuda_peak_allocated_mib": 85.0, "cuda_total_mib": 100.0},
                0.85,
            )
        enforce_no_new_oom_events(
            context,
            {"cgroup_event_oom": 2, "cgroup_event_oom_kill": 1},
            {"cgroup_event_oom": 2, "cgroup_event_oom_kill": 1},
        )
        with self.assertRaisesRegex(RuntimeError, "new OOM"):
            enforce_no_new_oom_events(
                context,
                {"cgroup_event_oom": 3, "cgroup_event_oom_kill": 1},
                {"cgroup_event_oom": 2, "cgroup_event_oom_kill": 1},
            )

    def test_world_size_migration_requires_authorization_and_equal_batch(self):
        state = {
            "config": {"training": {"batch_size": 1, "grad_accumulation_steps": 8}},
        }
        cfg = {"training": {"batch_size": 1, "grad_accumulation_steps": 1}}
        context = DistributedContext(True, 0, 0, 8, "nccl", "cuda:0")
        with self.assertRaisesRegex(ValueError, "allow-world-size-change"):
            validate_distributed_resume_contract(
                state,
                cfg,
                context,
                allow_world_size_change=False,
            )
        validate_distributed_resume_contract(
            state,
            cfg,
            context,
            allow_world_size_change=True,
        )
        legacy_ddp_state = {
            "config": {
                "training": {"batch_size": 2, "grad_accumulation_steps": 1}
            },
            "distributed_contract": {"world_size": 4},
        }
        validate_distributed_resume_contract(
            legacy_ddp_state,
            cfg,
            context,
            allow_world_size_change=True,
        )
        cfg["training"]["grad_accumulation_steps"] = 2
        with self.assertRaisesRegex(ValueError, "effective global batch"):
            validate_distributed_resume_contract(
                state,
                cfg,
                context,
                allow_world_size_change=True,
            )

    def test_rank_records_are_reduced_for_rank_zero_logging(self):
        records = [
            {
                "rank": 0,
                "sample_count": 1,
                "step": 2,
                "total_loss": 2.0,
                "forward_latency_ms": 10.0,
                "step_latency_ms": 15.0,
                "route_label_coverage": [1, 2],
                "route_valid_by_position": [1, 0],
                "route_accuracy_by_position": [1.0, 0.0],
                "router_entropy_by_position": [1.0, 1.0],
                "boundary_true_positive_count": 1,
                "boundary_predicted_positive_count": 1,
                "boundary_target_positive_count": 1,
                "boundary_precision": 1.0,
                "boundary_recall": 1.0,
                "boundary_f1": 1.0,
                "schedule_edit_distance_sum": 0.0,
                "schedule_edit_distance_count": 1,
                "schedule_edit_distance": 0.0,
                "ground_truth_boundary_crossing_count": 0,
                "ground_truth_boundary_crossing_valid_count": 1,
                "ground_truth_boundary_crossing_rate": 0.0,
                "rank_episode_ids": ["episode-a"],
                "resource_metrics": {"rank": 0, "process_rss_mib": 100.0},
            },
            {
                "rank": 1,
                "sample_count": 3,
                "step": 2,
                "total_loss": 4.0,
                "forward_latency_ms": 12.0,
                "step_latency_ms": 18.0,
                "route_label_coverage": [3, 4],
                "route_valid_by_position": [3, 2],
                "route_accuracy_by_position": [1.0 / 3.0, 0.5],
                "router_entropy_by_position": [3.0, 5.0],
                "boundary_true_positive_count": 1,
                "boundary_predicted_positive_count": 9,
                "boundary_target_positive_count": 9,
                "boundary_precision": 1.0 / 9.0,
                "boundary_recall": 1.0 / 9.0,
                "boundary_f1": 1.0 / 9.0,
                "schedule_edit_distance_sum": 2.0,
                "schedule_edit_distance_count": 3,
                "schedule_edit_distance": 2.0 / 3.0,
                "ground_truth_boundary_crossing_count": 2,
                "ground_truth_boundary_crossing_valid_count": 3,
                "ground_truth_boundary_crossing_rate": 2.0 / 3.0,
                "rank_episode_ids": ["episode-b"],
                "resource_metrics": {"rank": 1, "process_rss_mib": 120.0},
            },
        ]
        output = aggregate_distributed_records(records, {"world_size": 2})
        self.assertEqual(output["step"], 2)
        self.assertEqual(output["sample_count"], 4)
        self.assertEqual(output["total_loss"], 3.5)
        self.assertEqual(output["forward_latency_ms"], 12.0)
        self.assertEqual(output["step_latency_ms"], 18.0)
        self.assertEqual(output["route_label_coverage"], [4, 6])
        self.assertEqual(output["route_accuracy_by_position"], [0.5, 0.5])
        self.assertEqual(output["router_entropy_by_position"], [2.5, 4.0])
        self.assertAlmostEqual(output["boundary_precision"], 0.2)
        self.assertAlmostEqual(output["boundary_recall"], 0.2)
        self.assertAlmostEqual(output["boundary_f1"], 0.2)
        self.assertAlmostEqual(output["schedule_edit_distance"], 0.5)
        self.assertAlmostEqual(output["ground_truth_boundary_crossing_rate"], 0.5)
        self.assertEqual(output["current_skill_accuracy"], 0.5)
        self.assertEqual(output["future_position_accuracy"], 0.5)
        self.assertEqual(len(output["rank_resource_metrics"]), 2)
        self.assertFalse(distributed_episode_overlap(records))
        records[1]["rank_episode_ids"] = ["episode-a"]
        self.assertEqual(distributed_episode_overlap(records), ["episode-a"])


class PrefixQueueTests(unittest.TestCase):
    def test_libero_eval_resume_contract_rejects_mixed_or_duplicate_records(self):
        from scripts.eval_libero_temporal_skill import _load_existing_records

        record = {
            "task_suite": "libero_spatial",
            "policy_checkpoint": "/checkpoint.pt",
            "seed": 7,
            "flow_seed": 1701,
            "task_id": 0,
            "trial": 0,
            "success": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            loaded = _load_existing_records(
                path,
                resume=True,
                task_suite="libero_spatial",
                checkpoint_path="/checkpoint.pt",
                seed=7,
                flow_seed=1701,
                task_count=10,
                trials=50,
            )
            self.assertEqual(loaded, [record])
            with self.assertRaisesRegex(FileExistsError, "resume-results"):
                _load_existing_records(
                    path,
                    resume=False,
                    task_suite="libero_spatial",
                    checkpoint_path="/checkpoint.pt",
                    seed=7,
                    flow_seed=1701,
                    task_count=10,
                    trials=50,
                )
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _load_existing_records(
                    path,
                    resume=True,
                    task_suite="libero_spatial",
                    checkpoint_path="/checkpoint.pt",
                    seed=7,
                    flow_seed=1701,
                    task_count=10,
                    trials=50,
                )

    def test_formal_libero_eval_rejects_wrong_stage_and_dataset(self):
        from scripts.eval_libero_temporal_skill import _validate_libero_checkpoint

        statistics = {"q01": [-1.0] * 6, "q99": [1.0] * 6}
        metadata = {
            "stage": "joint",
            "data_contract": {
                "feature_store_contract": {
                    "source_contract": {
                        "dataset_names": ["libero_spatial_no_noops"]
                    }
                }
            },
        }
        _validate_libero_checkpoint(metadata, statistics, require_joint=True)
        with self.assertRaisesRegex(ValueError, "Stage 3"):
            _validate_libero_checkpoint(
                {**metadata, "stage": "expert_warmstart"},
                statistics,
                require_joint=True,
            )
        calvin = copy.deepcopy(metadata)
        calvin["data_contract"]["feature_store_contract"]["source_contract"][
            "dataset_names"
        ] = ["calvin_abc_language_segments"]
        with self.assertRaisesRegex(ValueError, "non-LIBERO"):
            _validate_libero_checkpoint(calvin, statistics, require_joint=True)

    def test_canonical_gripper_conversion_is_eval_only(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")
        canonical = np.zeros((2, 7), dtype=np.float32)
        canonical[1, -1] = 1.0
        converted = canonical_action_to_libero(canonical)
        self.assertEqual(converted[:, -1].tolist(), [1.0, -1.0])
        self.assertEqual(canonical[:, -1].tolist(), [0.0, 1.0])

    def test_evaluation_unnormalizes_six_motion_dimensions(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")
        canonical = np.zeros((1, 7), dtype=np.float32)
        stats = {
            "q01": [-2.0] * 6 + [0.0],
            "q99": [2.0] * 6 + [1.0],
            "mask": [True] * 6 + [False],
        }
        converted = canonical_action_to_libero(canonical, stats)
        self.assertEqual(converted[0, :6].tolist(), [0.0] * 6)
        self.assertEqual(float(converted[0, -1]), 1.0)

    def test_queue_requeries_and_drops_old_suffix(self):
        calls = []

        def policy(observation):
            calls.append(observation)
            query = len(calls)
            return [f"q{query}-a0", f"q{query}-a1"], {"source": query}

        queue = VariablePrefixActionQueue(policy)
        actions = [queue.next_action("obs0")[0], queue.next_action("unused")[0]]
        third, metadata = queue.next_action("obs1")
        self.assertEqual(actions, ["q1-a0", "q1-a1"])
        self.assertEqual(third, "q2-a0")
        self.assertEqual(metadata["query_id"], 2)
        self.assertEqual(calls, ["obs0", "obs1"])

    def test_eight_four_short_prefixes_requery_exactly_at_exhaustion(self):
        lengths = [8, 4, 2]
        queries = []

        def policy(observation):
            query = len(queries)
            queries.append(observation)
            length = lengths[min(query, len(lengths) - 1)]
            return [f"q{query + 1}-{step}" for step in range(length)], {}

        queue = VariablePrefixActionQueue(policy)
        observed = []
        for step in range(sum(lengths)):
            action, metadata = queue.next_action(f"obs{step}")
            observed.append((action, metadata["query_id"]))
        self.assertEqual(
            [item[1] for item in observed],
            [1] * 8 + [2] * 4 + [3] * 2,
        )
        self.assertEqual(queries, ["obs0", "obs8", "obs12"])


class CalvinAdapterTests(unittest.TestCase):
    def test_evaluation_action_adapter_comes_from_checkpoint_contract(self):
        from mowe_wam.benchmarks.calvin.custom_model import (
            action_adapter_from_checkpoint,
        )

        raw = {
            "motion_q01": [-0.5] * 6,
            "motion_q99": [0.5] * 6,
            "motion_mask": [True] * 6,
            "action_mode": "relative_cartesian",
            "rotation_representation": "euler_xyz",
            "gripper_open_value": 1.0,
            "gripper_closed_value": -1.0,
            "clip_normalized_motion": True,
        }
        metadata = {
            "data_contract": {
                "joint_action_statistics": {
                    "q01": [-0.5] * 6 + [0.0],
                    "q99": [0.5] * 6 + [1.0],
                    "raw_calvin_contract": raw,
                }
            }
        }
        benchmark = {"action": {"motion_q01": "TBD", "motion_q99": "TBD"}}
        adapter = action_adapter_from_checkpoint(metadata, benchmark)
        self.assertEqual(adapter.motion_q01, (-0.5,) * 6)
        self.assertEqual(benchmark["action"]["gripper_open_value"], 1.0)
        incompatible = copy.deepcopy(benchmark)
        incompatible["action"]["motion_q01"] = [-0.4] * 6
        with self.assertRaisesRegex(ValueError, "differs"):
            action_adapter_from_checkpoint(metadata, incompatible)

    def test_calvin_sequence_diagnostics_track_attempted_tasks_and_failure_position(self):
        from scripts.eval_calvin_flow_wam import _summarize_sequence_records

        report = _summarize_sequence_records(
            [
                {
                    "subtasks": ["open_drawer", "pick_block", "place_block"],
                    "completed": 2,
                    "policy_queries": 4,
                    "prefix_lengths": [1, 2, 2, 3],
                },
                {
                    "subtasks": ["open_drawer", "pick_block", "place_block"],
                    "completed": 3,
                    "policy_queries": 5,
                    "prefix_lengths": [1, 1, 2, 3, 3],
                },
            ]
        )
        self.assertEqual(report["per_task"]["open_drawer"]["success_rate"], 1.0)
        self.assertEqual(report["per_task"]["pick_block"]["attempts"], 2)
        self.assertEqual(report["per_task"]["place_block"]["successes"], 1)
        self.assertEqual(report["failure_position_counts"], {"3": 1, "complete": 1})
        self.assertEqual(report["policy_queries"], 9)
        self.assertEqual(report["prefix_length_histogram"], {"1": 3, "2": 3, "3": 3})

    def test_action_roundtrip_and_gripper_contract(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is not installed")
        from mowe_wam.benchmarks.calvin import CalvinActionAdapter

        adapter = CalvinActionAdapter(
            motion_q01=[-2.0] * 6,
            motion_q99=[2.0] * 6,
            gripper_open_value=-1.0,
            gripper_closed_value=1.0,
        )
        raw = np.asarray(
            [[-2.0] * 6 + [-1.0], [2.0] * 6 + [1.0]], dtype=np.float32
        )
        shared = adapter.to_shared_action(raw)
        self.assertTrue(np.allclose(shared[:, :6], [[-1.0] * 6, [1.0] * 6]))
        self.assertEqual(shared[:, -1].tolist(), [0.0, 1.0])
        self.assertTrue(np.allclose(adapter.from_shared_action(shared), raw))

    def test_goal_change_drops_suffix_but_preserves_episode_memory(self):
        try:
            import numpy as np
            import torch
        except ModuleNotFoundError:
            self.skipTest("numpy and torch are required")
        from mowe_wam.benchmarks.calvin import (
            CalvinActionAdapter,
            CalvinTemporalPolicyAdapter,
        )

        class FakeModel:
            def __init__(self):
                self.calls = []

            def predict_actions(self, batch, *, flow_seed):
                self.calls.append(
                    {
                        "language": batch["language"][0],
                        "history_mask": batch["history_mask"][0].tolist(),
                        "flow_seed": flow_seed,
                    }
                )
                action = torch.zeros(3, 7)
                action[:, 0] = len(self.calls) / 10.0
                action[:, -1] = 1.0
                return [action], {
                    "route_indices": torch.zeros(1, 8, dtype=torch.long),
                    "current_view_weights": torch.full((1, 2), 0.5),
                    "view_order": ["primary", "wrist"],
                }

        model = FakeModel()

        def transform(image):
            return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()

        action_adapter = CalvinActionAdapter(
            motion_q01=[-1.0] * 6,
            motion_q99=[1.0] * 6,
            gripper_open_value=-1.0,
            gripper_closed_value=1.0,
        )
        policy = CalvinTemporalPolicyAdapter(
            model,
            transform,
            action_adapter,
            history_length=4,
            long_memory_slots=2,
            flow_seed=11,
        )
        observation = {
            "rgb_obs": {
                "rgb_static": np.zeros((2, 2, 3), dtype=np.uint8),
                "rgb_gripper": np.ones((2, 2, 3), dtype=np.uint8),
            },
            "robot_obs": np.zeros(15, dtype=np.float32),
        }
        with self.assertRaisesRegex(RuntimeError, "reset"):
            policy.step(observation, "first task")
        policy.reset_sequence()
        first = policy.step(observation, "first task")
        second = policy.step(observation, "first task")
        self.assertEqual(len(model.calls), 1)
        policy.reset()  # Official CALVIN currently calls this before each subtask.
        third = policy.step(observation, {"language": "second task"})
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(model.calls[0]["history_mask"], [False, False, False, True])
        self.assertEqual(model.calls[1]["history_mask"], [False, True, True, True])
        self.assertEqual(model.calls[1]["language"], "second task")
        self.assertTrue(policy.last_step_metadata["goal_changed"])
        self.assertAlmostEqual(float(first[0]), 0.1)
        self.assertAlmostEqual(float(second[0]), 0.1)
        self.assertAlmostEqual(float(third[0]), 0.2)
        self.assertEqual(float(first[-1]), 1.0)

        policy.reset_sequence()
        policy.step(observation, "third task")
        self.assertEqual(model.calls[-1]["history_mask"], [False, False, False, True])
        self.assertEqual(model.calls[-1]["flow_seed"], 10011)


class ScheduleTests(unittest.TestCase):
    def test_instruction_prompt_matches_upstream_pre_action_form(self):
        prompt = OpenVLAOFTAdapter.format_instruction_prompt("  Pick UP   the Cup ")
        self.assertEqual(prompt, "In: What action should the robot take to pick up the cup?\nOut:")
        self.assertNotIn("<ACTION", prompt)

    def test_action_condition_schedule(self):
        self.assertEqual(action_condition_probability(0, 100), 0.0)
        self.assertAlmostEqual(action_condition_probability(50, 100), 0.4)
        self.assertEqual(action_condition_probability(100, 100), 0.8)

    def test_router_schedule_reaches_st_gumbel_endpoint(self):
        start = temporal_router_schedule(0, 100)
        end = temporal_router_schedule(100, 100)
        self.assertEqual(start["oracle_route_probability"], 1.0)
        self.assertEqual(start["gumbel_temperature"], 1.0)
        self.assertEqual(end["oracle_route_probability"], 0.0)
        self.assertEqual(end["gumbel_temperature"], 0.1)


if __name__ == "__main__":
    unittest.main()
