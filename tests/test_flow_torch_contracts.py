from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from mowe_wam.training import flow_wam_skill_losses
from mowe_wam.data.libero_sequence_dataset import build_episode_windows
from mowe_wam.data import (
    ShardedVisualTargetCache,
    ShardedVisualTargetCacheWriter,
    feature_cache_key,
    validate_visual_cache_metadata,
)
from mowe_wam.training.flow_runtime import (
    build_flow_optimizer,
    build_warmup_cosine_scheduler,
    configure_flow_stage,
    evaluate_flow_model,
    load_flow_checkpoint,
    make_grad_scaler,
    read_flow_checkpoint_metadata,
    save_flow_checkpoint,
    validation_loss_early_stopping_state,
    validate_checkpoint_contract,
    validate_resume_schedule_contract,
    _mechanism_metrics,
    _route_metrics,
)
from mowe_wam.utils.config import load_config


@unittest.skipIf(torch is None, "torch is not installed")
class FlowTorchTests(unittest.TestCase):
    def setUp(self):
        from scripts.check_flow_wam_forward import build_model, make_batch

        torch.manual_seed(7)
        self.model = build_model(torch, 2)
        self.batch = make_batch(torch, 2)
        self.weights = load_config("configs/mowe_wam/train_flow_wam_skill_moe.yaml")[
            "loss_weights"
        ]

    def test_validation_loss_early_stopping_is_resume_stable(self):
        def record(step, loss):
            return {
                "stage": "joint",
                "step": step,
                "metrics": {"total_loss": loss},
            }

        records = [
            record(0, 3.0),
            record(500, 2.0),
            record(1000, 1.0),
            record(1500, 0.99995),
            record(2000, 0.99994),
            record(2500, 0.99993),
            record(3000, 0.99992),
            record(3500, 0.99991),
            record(3500, 0.99991),
        ]
        before_minimum = validation_loss_early_stopping_state(
            records,
            stage="joint",
            min_delta=1e-4,
            patience=5,
            min_steps=5000,
        )
        self.assertFalse(before_minimum["should_stop"])
        self.assertEqual(before_minimum["bad_validation_count"], 5)
        self.assertEqual(before_minimum["validation_count"], 8)

        records.append(record(5000, 0.99990))
        stopped = validation_loss_early_stopping_state(
            records,
            stage="joint",
            min_delta=1e-4,
            patience=5,
            min_steps=5000,
        )
        self.assertTrue(stopped["should_stop"])
        self.assertEqual(stopped["current_step"], 5000)

    def test_view_fusion_starts_uniform_and_is_language_conditioned(self):
        from mowe_wam.models import LanguageConditionedViewFusion

        fusion = LanguageConditionedViewFusion(8, hidden_dim=4)
        views = torch.randn(3, 2, 8)
        language = torch.randn(3, 8)
        neutral = fusion(views, language)
        self.assertTrue(torch.allclose(neutral["weights"], torch.full((3, 2), 0.5)))
        self.assertTrue(torch.allclose(neutral["fused"], views.mean(dim=1)))
        with torch.no_grad():
            fusion.score_head.weight.fill_(0.25)
        changed_a = fusion(views, language)["weights"]
        changed_b = fusion(views, -language)["weights"]
        self.assertFalse(torch.equal(changed_a, changed_b))
        self.assertTrue(torch.allclose(changed_a.sum(dim=-1), torch.ones(3)))

    def test_risk_gated_execution_selects_eight_four_and_boundary_stop(self):
        from mowe_wam.models import risk_gated_execution

        routes = torch.tensor(
            [
                [0, 0, 0, 2, 2, 1, 1, 1],
                [0, 0, 0, 2, 2, 2, 2, 2],
                [0, 0, 2, 2, 2, 2, 2, 2],
            ]
        )
        probabilities = torch.nn.functional.one_hot(routes, 7).float()
        motion = torch.zeros(3, 8, 6)
        motion[1, 3:, 0] = 0.7
        motion[2, 2:, 0] = 1.0
        execution = risk_gated_execution(
            routes,
            probabilities,
            motion,
            torch.zeros_like(motion),
        )
        self.assertEqual(execution["execution_steps"].tolist(), [8, 4, 2])
        self.assertEqual(execution["execution_reason_code"].tolist(), [0, 1, 2])
        self.assertEqual(
            execution["execution_crosses_predicted_boundary"].tolist(),
            [True, True, False],
        )

    def test_all_experts_and_st_router_receive_gradients(self):
        output = self.model(
            self.batch,
            action_condition_mode="ground_truth",
            route_mode="oracle",
            flow_seed=7,
        )
        self.assertTrue(torch.equal(output["action_distance_gate"], torch.ones_like(output["action_distance_gate"])))
        losses = flow_wam_skill_losses(output, self.batch, self.weights, stage="joint")
        losses["total_loss"].backward()
        self.assertTrue(
            any(
                parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
                for parameter in self.model.view_fusion.parameters()
            )
        )
        coverage = [
            any(parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()) for parameter in head.parameters())
            for head in self.model.residual_experts.velocity_heads
        ]
        self.assertTrue(all(coverage))
        null = output["route_indices"].eq(6).unsqueeze(-1)
        self.assertFalse(bool((output["residual_motion"].ne(0) & null).any()))
        residual_norm = output["residual_motion"].float().norm(dim=-1)
        self.assertLessEqual(
            float(residual_norm.max().detach()), self.model.max_residual_l2 + 1e-6
        )

        oversized = torch.full((2, 8, 6), 10.0)
        projected, clipped = self.model._project_residual(oversized)
        self.assertTrue(bool(clipped.all()))
        self.assertLessEqual(
            float(projected.float().norm(dim=-1).max().detach()),
            self.model.max_residual_l2 + 1e-6,
        )

        self.model.zero_grad(set_to_none=True)
        st_output = self.model(
            self.batch,
            action_condition_mode="nominal",
            route_mode="st_gumbel",
            flow_seed=8,
            compute_teacher_targets=False,
        )
        st_output["motion_actions"].square().mean().backward()
        self.assertTrue(
            any(
                parameter.grad is not None and float(parameter.grad.abs().sum()) > 0
                for parameter in self.model.router.parameters()
            )
        )

    def test_mechanism_diagnostics_cover_documented_temporal_metrics(self):
        output = self.model(
            self.batch,
            action_condition_mode="ground_truth",
            route_mode="predicted",
            flow_seed=7,
            compute_route_diagnostics=True,
        )
        route = _route_metrics(output, self.batch)
        mechanism = _mechanism_metrics(output, self.batch)
        self.assertEqual(len(route["route_confusion_matrix"]), 7)
        self.assertIn("schedule_edit_distance", route)
        self.assertIn("future_position_accuracy", route)
        self.assertIn("execution_steps_histogram", route)
        self.assertEqual(len(route["router_entropy_by_position"]), 16)
        self.assertEqual(
            set(mechanism["future_horizon_metrics"]), {"1", "4", "8", "16"}
        )
        self.assertEqual(set(mechanism["per_skill_diagnostics"]), {
            "pick_grasp",
            "place_release",
            "move_transport",
            "open_close",
            "turn_rotate",
            "push_pull",
            "null_finish",
        })
        self.assertEqual(
            set(output["route_mode_diagnostics"]),
            {"oracle", "st_gumbel", "hard_predicted"},
        )
        self.assertEqual(len(mechanism["per_skill_position_diagnostics"]["pick_grasp"]), 16)
        self.assertTrue(
            {"action_query", "world_token_query", "world_belief", "future", "delta"}
            <= set(mechanism["router_branch_norms"])
        )
        self.assertEqual(mechanism["view_order"], ["primary", "wrist"])
        self.assertEqual(len(mechanism["current_view_weights_mean"]), 2)

    def test_frozen_visual_lru_deduplicates_frames_and_evicts(self):
        from mowe_wam.backbones import OpenVLAOFTAdapter

        class FakeAdapter(OpenVLAOFTAdapter):
            def __init__(self):
                torch.nn.Module.__init__(self)
                self.freeze_backbone = True
                self.visual_cache_size = 1
                self.num_images_in_input = 2
                self._visual_cache = __import__("collections").OrderedDict()
                self.device = torch.device("cpu")
                self.dtype = torch.float32
                self.encoded_frames = 0

            def encode_image_tokens(self, pixel_values):
                self.encoded_frames += pixel_values.shape[0]
                pooled = pixel_values.float().mean(dim=(1, 2, 3), keepdim=False)
                return pooled[:, None, None].repeat(1, 2, 4)

        adapter = FakeAdapter()
        first = torch.zeros(3, 2, 2)
        second = torch.ones(3, 2, 2)
        duplicated = adapter.encode_pooled_views(
            torch.stack([first, first]), torch.stack([second, second])
        )
        self.assertEqual(adapter.encoded_frames, 1)
        self.assertTrue(torch.equal(duplicated[0], duplicated[1]))
        simultaneous = adapter.encode_pooled_views(
            torch.stack([first, second]), torch.stack([second, first])
        )
        self.assertEqual(simultaneous.shape, (2, 2, 4))
        self.assertEqual(adapter.encoded_frames, 2)
        adapter.encode_pooled_views(first.unsqueeze(0), second.unsqueeze(0))
        self.assertEqual(adapter.encoded_frames, 3)
        adapter.encode_pooled_views(second.unsqueeze(0), first.unsqueeze(0))
        self.assertEqual(adapter.encoded_frames, 4)
        adapter.encode_pooled_views(first.unsqueeze(0), second.unsqueeze(0))
        self.assertEqual(adapter.encoded_frames, 5)

    def test_flow_log_analyzer_groups_route_modes(self):
        from scripts.analyze_flow_wam_logs import summarize

        rows = [
            {
                "step": 1,
                "stage": "joint",
                "ablation": "main",
                "route_source": "oracle",
                "total_loss": 2.0,
                "route_mode_diagnostics": {
                    "oracle": {"motion_endpoint_l1": 0.2, "route_accuracy": 1.0}
                },
            },
            {
                "step": 2,
                "stage": "joint",
                "ablation": "main",
                "route_source": "predicted",
                "total_loss": 4.0,
                "route_mode_diagnostics": {
                    "oracle": {"motion_endpoint_l1": 0.4, "route_accuracy": 0.5}
                },
            },
        ]
        summary = summarize(rows)
        self.assertEqual(summary["scalar_means"]["total_loss"], 3.0)
        self.assertEqual(summary["route_sources"]["oracle"], 1)
        self.assertAlmostEqual(
            summary["route_mode_diagnostic_means"]["oracle"]["motion_endpoint_l1"],
            0.3,
        )

    def test_checkpoint_restores_step_optimizer_and_schedule(self):
        cfg = load_config("configs/mowe_wam/train_flow_wam_skill_moe.yaml")
        cfg["training"].update({"device": "cpu", "precision": "float32", "max_steps": 2})
        configure_flow_stage(self.model, "joint")
        optimizer = build_flow_optimizer(cfg, self.model)
        scheduler = build_warmup_cosine_scheduler(cfg, optimizer)
        scaler = make_grad_scaler("cpu", "float32")
        output = self.model(
            self.batch,
            action_condition_mode="ground_truth",
            route_mode="oracle",
            flow_seed=7,
        )
        losses = flow_wam_skill_losses(output, self.batch, self.weights, stage="joint")
        losses["total_loss"].backward()
        optimizer.step()
        scheduler.step()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_flow_checkpoint(
                path,
                self.model,
                optimizer,
                scheduler,
                scaler,
                1,
                cfg,
                "joint",
                {"route_mode": "oracle", "step": 1},
                distributed_metadata={
                    "enabled": False,
                    "world_size": 1,
                    "per_device_batch_size": 1,
                    "grad_accumulation_steps": 1,
                    "effective_global_batch": 1,
                },
                rng_state_by_rank=[{"rank": 0}],
            )
            step, schedule = load_flow_checkpoint(
                path,
                self.model,
                optimizer,
                scheduler,
                scaler,
                resume=True,
            )
            metadata = read_flow_checkpoint_metadata(path)
            self.assertTrue(path.with_suffix(path.suffix + ".metadata.json").exists())
            self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())
            self.assertFalse(
                path.with_suffix(path.suffix + ".metadata.json.tmp").exists()
            )
            validate_checkpoint_contract(metadata, cfg)
            validate_resume_schedule_contract(metadata, cfg)
            changed_schedule = load_config(
                "configs/mowe_wam/train_flow_wam_skill_moe.yaml"
            )
            changed_schedule["training"].update(
                {
                    "device": "cpu",
                    "precision": "float32",
                    "max_steps": 3,
                    "stop_step": 2,
                }
            )
            with self.assertRaisesRegex(ValueError, "Same-stage resume"):
                validate_resume_schedule_contract(metadata, changed_schedule)
            unchanged_schedule = load_config(
                "configs/mowe_wam/train_flow_wam_skill_moe.yaml"
            )
            unchanged_schedule["training"].update(
                {
                    "device": "cpu",
                    "precision": "float32",
                    "max_steps": 2,
                    "stop_step": 2,
                    "save_freq": 1,
                    "log_freq": 1,
                }
            )
            validate_resume_schedule_contract(metadata, unchanged_schedule)
            mismatched_cfg = load_config("configs/mowe_wam/train_flow_wam_skill_moe.yaml")
            mismatched_cfg["view_fusion"]["view_order"] = ["wrist", "primary"]
            with self.assertRaisesRegex(ValueError, "dual-view contract"):
                validate_checkpoint_contract(metadata, mismatched_cfg)
            path.with_suffix(path.suffix + ".metadata.json").unlink()
            fallback_metadata = read_flow_checkpoint_metadata(path)
            self.assertEqual(fallback_metadata["step"], 1)
        self.assertEqual(step, 1)
        self.assertEqual(schedule, {"route_mode": "oracle", "step": 1})
        self.assertEqual(metadata["stage"], "joint")
        self.assertEqual(metadata["distributed_contract"]["world_size"], 1)
        self.assertEqual(metadata["distributed_contract"]["effective_global_batch"], 1)
        self.assertEqual(metadata["flow_contract"]["implementation_id"], "rectified_flow_euler_v1")
        self.assertEqual(
            [group["name"] for group in optimizer.param_groups],
            [
                "view_fusion",
                "memory_encoder",
                "nominal_flow",
                "gripper_head",
                "world_model",
                "router",
                "residual_experts",
                "expert_context",
            ],
        )

    def test_checkpoint_stage_guard_rejects_wrong_predecessor(self):
        cfg = load_config("configs/mowe_wam/train_flow_wam_skill_moe.yaml")
        cfg["training"].update({"device": "cpu", "precision": "float32", "max_steps": 2})
        configure_flow_stage(self.model, "joint")
        optimizer = build_flow_optimizer(cfg, self.model)
        scheduler = build_warmup_cosine_scheduler(cfg, optimizer)
        scaler = make_grad_scaler("cpu", "float32")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_flow_checkpoint(
                path,
                self.model,
                optimizer,
                scheduler,
                scaler,
                1,
                cfg,
                "joint",
                {"step": 1},
            )
            with self.assertRaisesRegex(ValueError, "Checkpoint stage"):
                load_flow_checkpoint(path, self.model, allowed_stages={"expert_warmstart"})

    def test_context_boundary_removes_all_training_targets(self):
        original = self.model.backbone

        class RecordingBackbone:
            hidden_dim = original.hidden_dim

            def __init__(self):
                self.keys = None

            def extract_context_features(self, batch):
                self.keys = set(batch)
                return original.extract_context_features(batch)

            def keep_frozen_backbone_eval(self):
                return None

        recorder = RecordingBackbone()
        self.model.backbone = recorder
        self.model(
            self.batch,
            action_condition_mode="ground_truth",
            route_mode="oracle",
            flow_seed=7,
        )
        self.assertFalse(
            recorder.keys
            & {
                "target_actions",
                "target_motion",
                "target_gripper",
                "expert_skill_labels",
                "expert_skill_mask",
                "future_latent_targets",
            }
        )

    def test_future_shuffle_is_not_identity_for_batch_one(self):
        from scripts.check_flow_wam_forward import build_model, make_batch

        torch.manual_seed(11)
        model = build_model(torch, 1)
        batch = make_batch(torch, 1)
        model.ablation = {}
        baseline = model(
            batch,
            action_condition_mode="nominal",
            route_mode="predicted",
            flow_seed=5,
            compute_teacher_targets=False,
        )
        model.ablation = {"shuffle_future_before_router": True}
        shuffled = model(
            batch,
            action_condition_mode="nominal",
            route_mode="predicted",
            flow_seed=5,
            compute_teacher_targets=False,
        )
        self.assertGreater(float(shuffled["future_shuffle_router_logit_l1"].detach()), 0.0)
        self.assertFalse(torch.equal(baseline["router_logits"], shuffled["router_logits"]))

    def test_sharded_visual_cache_roundtrip_and_metadata_gate(self):
        metadata = {"teacher_checkpoint": "teacher", "dataset_fingerprint": "abc"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            writer = ShardedVisualTargetCacheWriter(root, metadata, shard_size=2)
            episode = "episode"
            for step in (0, 1, 4, 8):
                writer.add(feature_cache_key(episode, step), torch.full((2, 3), float(step)))
            manifest = writer.close()
            self.assertTrue(manifest.exists())
            cache = ShardedVisualTargetCache(root, max_open_shards=1)
            current, future = cache.window(episode, 0, [1, 4, 8])
            self.assertEqual(current.tolist(), [[0.0] * 3, [0.0] * 3])
            self.assertEqual(future[:, 0, 0].tolist(), [1.0, 4.0, 8.0])
            validate_visual_cache_metadata(cache.metadata, metadata)
            with self.assertRaisesRegex(ValueError, "metadata mismatch"):
                validate_visual_cache_metadata(cache.metadata, {"dataset_fingerprint": "wrong"})

    def test_sequence_window_splits_motion_gripper_and_per_step_skills(self):
        episode = []
        for index in range(16):
            actions = torch.zeros(8, 7)
            actions[:, -1] = index % 2
            episode.append(
                {
                    "dataset_name": "libero_spatial_no_noops",
                    "language": "pick and place",
                    "policy_pixel_values_primary": torch.zeros(3, 4, 4),
                    "policy_pixel_values_wrist": torch.ones(3, 4, 4),
                    "raw_pixel_values": torch.zeros(3, 4, 4, dtype=torch.uint8),
                    "raw_wrist_pixel_values": torch.ones(3, 4, 4, dtype=torch.uint8),
                    "actions": actions,
                    "expert_skill_label": index % 7,
                    "expert_label_source": "raw_annotation",
                }
            )
        sample = next(build_episode_windows(episode))
        self.assertEqual(tuple(sample["target_motion"].shape), (8, 6))
        self.assertEqual(tuple(sample["target_gripper"].shape), (8, 1))
        self.assertEqual(tuple(sample["pixel_values_primary"].shape), (3, 4, 4))
        self.assertEqual(tuple(sample["pixel_values_wrist"].shape), (3, 4, 4))
        self.assertEqual(sample["expert_skill_labels"].tolist(), [0, 1, 2, 3, 4, 5, 6, 0])
        self.assertEqual(sample["target_gripper"].flatten().tolist(), [0.0, 1.0] * 4)

    def test_online_memory_matches_training_time_indices_and_resets(self):
        from mowe_wam.memory import OnlineMemoryState

        state = OnlineMemoryState(history_length=8, long_memory_slots=4)
        for index in range(13):
            image = torch.full((3, 2, 2), float(index))
            wrist = torch.full((3, 2, 2), float(index + 100))
            previous_action = None if index == 0 else torch.full((7,), float(index - 1))
            state.append(image, wrist, previous_action)
        tensors = state.tensors()
        self.assertEqual(tensors["history_pixel_values_primary"][:, 0, 0, 0].tolist(), list(map(float, range(5, 12))))
        self.assertEqual(tensors["history_pixel_values_wrist"][:, 0, 0, 0].tolist(), list(map(float, range(105, 112))))
        self.assertEqual(tensors["history_actions"][:, 0].tolist(), list(map(float, range(5, 12))))
        self.assertEqual(tensors["long_history_pixel_values_primary"][:, 0, 0, 0].tolist(), [0.0, 1.0, 2.0, 4.0])
        self.assertEqual(tensors["long_history_pixel_values_wrist"][:, 0, 0, 0].tolist(), [100.0, 101.0, 102.0, 104.0])
        self.assertEqual(tensors["long_history_actions"][:, 0].tolist(), [0.0, 1.0, 2.0, 4.0])
        state.reset()
        self.assertEqual(state.images, [])
        self.assertEqual(state.actions, [])

    def test_temporal_policy_adapter_observes_queued_intermediate_steps(self):
        import numpy as np

        from mowe_wam.evaluation import TemporalSkillPolicyAdapter

        class FakeModel:
            def __init__(self):
                self.calls = []

            def predict_actions(self, batch, flow_seed=None):
                self.calls.append((batch, flow_seed))
                length = len(self.calls)
                prefix = torch.zeros((length, 7))
                prefix[:, 0] = float(length)
                return [prefix], {
                    "route_indices": torch.zeros((1, 8), dtype=torch.long),
                    "current_view_weights": torch.tensor([[0.5, 0.5]]),
                    "view_order": ("primary", "wrist"),
                }

        model = FakeModel()
        adapter = TemporalSkillPolicyAdapter(
            model,
            lambda image: torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float(),
            action_statistics={
                "q01": [-1.0] * 6 + [0.0],
                "q99": [1.0] * 6 + [1.0],
                "mask": [True] * 6 + [False],
            },
        )
        adapter.reset("pick up the cup")
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        adapter.next_canonical_action(image, image)
        adapter.next_canonical_action(image, image)
        adapter.next_canonical_action(image, image)
        self.assertEqual(len(model.calls), 2)
        self.assertEqual(len(adapter.memory.images), 3)
        second_batch = model.calls[1][0]
        self.assertEqual(second_batch["history_mask"][0].tolist()[-2:], [True, True])
        self.assertEqual(float(second_batch["history_actions"][0, -1, 0]), 1.0)

    def test_validation_is_deterministic_and_reports_partition(self):
        cfg = load_config("configs/mowe_wam/train_nominal_flow_wam.yaml")
        cfg["training"].update({"device": "cpu", "precision": "float32"})
        cfg["validation"].update({"num_batches": 1, "seed": 1701})
        configure_flow_stage(self.model, "nominal_flow_pretrain")
        first = evaluate_flow_model(
            cfg,
            self.model,
            [self.batch],
            stage="nominal_flow_pretrain",
            step=0,
        )
        second = evaluate_flow_model(
            cfg,
            self.model,
            [self.batch],
            stage="nominal_flow_pretrain",
            step=0,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["episode_partition"], "validation")
        self.assertEqual(first["batches"], 1)
        self.assertIn("total_loss", first["metrics"])

    def test_stage1_has_no_trainable_parameter_without_gradient(self):
        cfg = load_config("configs/mowe_wam/train_nominal_flow_wam.yaml")
        configure_flow_stage(self.model, "nominal_flow_pretrain")
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in self.model.nominal_action_head.flow_trunk.token_condition_projection.parameters()
            )
        )
        self.assertFalse(
            any(
                parameter.requires_grad
                for parameter in self.model.world_model.route_world_head.parameters()
            )
        )

        output = self.model(
            self.batch,
            action_condition_mode="ground_truth",
            route_mode="predicted",
            flow_seed=7,
            compute_residual=False,
        )
        losses = flow_wam_skill_losses(
            output,
            self.batch,
            cfg["loss_weights"],
            stage="nominal_flow_pretrain",
        )
        losses["total_loss"].backward()
        missing = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        self.assertEqual(missing, [])

    def test_route_world_head_reenabled_after_stage1(self):
        for stage in ("expert_warmstart", "joint"):
            with self.subTest(stage=stage):
                configure_flow_stage(self.model, stage)
                self.assertTrue(
                    all(
                        parameter.requires_grad
                        for parameter in self.model.world_model.route_world_head.parameters()
                    )
                )
                self.assertFalse(
                    any(
                        parameter.requires_grad
                        for parameter in self.model.nominal_action_head.flow_trunk.token_condition_projection.parameters()
                    )
                )


if __name__ == "__main__":
    unittest.main()
