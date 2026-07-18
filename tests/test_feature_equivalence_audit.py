import unittest

import torch

from scripts.audit_feature_store_equivalence import (
    _feature_gate_error,
    _output_gate_error,
    _tensor_error,
)


class FeatureEquivalenceAuditTests(unittest.TestCase):
    def test_masked_error_excludes_padding_vectors(self):
        cached = torch.zeros(1, 3, 2, 4)
        online = cached.clone()
        online[:, 0] = 8.0
        online[:, 1:] = 0.01
        mask = torch.tensor([[False, True, True]])

        error = _tensor_error(online, cached, mask=mask)

        self.assertAlmostEqual(error["max_abs"], 0.01, places=6)
        self.assertEqual(error["compared_vectors"], 4)
        self.assertEqual(error["ignored_vectors"], 2)

    def test_dino_gate_uses_training_metrics_not_absolute_outlier(self):
        base = {
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "smooth_l1": 0.0,
            "mean_cosine_distance": 0.0,
        }
        errors = {
            name: dict(base)
            for name in (
                "current_visual_views",
                "history_visual_views",
                "long_history_visual_views",
                "language",
                "current_dino",
                "future_dino",
            )
        }
        errors["current_dino"].update(
            max_abs=0.7,
            mean_abs=0.07,
            smooth_l1=0.004,
            mean_cosine_distance=0.002,
        )

        self.assertAlmostEqual(_feature_gate_error(errors), 0.004)

    def test_output_gate_excludes_assembled_binary_action(self):
        errors = {
            "nominal_actions": {"max_abs": 0.001},
            "future_latents": {"max_abs": 0.002},
            "route_world_tokens": {"max_abs": 0.003},
            "router_logits": {"max_abs": 0.001},
            "motion_actions": {"max_abs": 0.001},
            "gripper_logits": {"max_abs": 0.004},
            "actions": {"max_abs": 1.0},
        }

        self.assertAlmostEqual(_output_gate_error(errors), 0.004)


if __name__ == "__main__":
    unittest.main()
