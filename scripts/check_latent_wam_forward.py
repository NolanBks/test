#!/usr/bin/env python3
"""Run a checkpoint-free synthetic forward/backward check of the new main path."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.memory import MultiScaleMemoryEncoder
from mowe_wam.models import (
    LegacyFutureGroundedRouter,
    LatentWAMPolicy,
    LegacyLatentWorldModel,
    RegressionNominalActionHead,
    ResidualActionExperts,
)
from mowe_wam.training import latent_wam_training_losses
from mowe_wam.utils.optional import require_torch


class SyntheticContextBackbone:
    hidden_dim = 128

    def extract_context_features(self, batch):
        return {
            "current_visual": batch["synthetic_current_visual"],
            "history_visual": batch["synthetic_history_visual"],
            "long_history_visual": batch["synthetic_long_visual"],
            "language": batch["synthetic_language"],
        }

    def keep_frozen_backbone_eval(self):
        return None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", action="store_true", help="Accepted for explicit smoke-command clarity.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lightweight", action="store_true", help="Use a smaller WAM for CPU-only debugging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    batch_size = args.batch_size
    context_dim = 64 if args.lightweight else 128
    hidden_dim = 64 if args.lightweight else 512
    target_dim = 32 if args.lightweight else 384
    world_layers = 2 if args.lightweight else 6
    history_length, long_slots, horizons = 8, 4, 3
    memory = MultiScaleMemoryEncoder(context_dim, context_dim, hidden_dim=hidden_dim, heads=8)
    nominal = RegressionNominalActionHead(context_dim, hidden_dim, hidden_dim, action_dim=7, chunk_size=8)
    world = LegacyLatentWorldModel(
        context_dim,
        hidden_dim=hidden_dim,
        layers=world_layers,
        heads=8,
        target_tokens=16,
        target_dim=target_dim,
    )
    router = LegacyFutureGroundedRouter(hidden_dim, hidden_dim, latent_dim=target_dim, hidden_dim=hidden_dim)
    experts = ResidualActionExperts(hidden_dim=hidden_dim)
    model = LatentWAMPolicy(
        SyntheticContextBackbone(),
        memory,
        nominal,
        world,
        router,
        experts,
        visual_teacher=None,
        context_dim=context_dim,
        memory_dim=hidden_dim,
        world_dim=hidden_dim,
        expert_hidden_dim=hidden_dim,
    )
    batch = {
        "synthetic_current_visual": torch.randn(batch_size, context_dim),
        "synthetic_history_visual": torch.randn(batch_size, history_length - 1, context_dim),
        "synthetic_long_visual": torch.randn(batch_size, long_slots, context_dim),
        "synthetic_language": torch.randn(batch_size, context_dim),
        "history_actions": torch.randn(batch_size, history_length - 1, 7),
        "history_mask": torch.ones(batch_size, history_length, dtype=torch.bool),
        "long_history_actions": torch.randn(batch_size, long_slots, 7),
        "long_history_mask": torch.ones(batch_size, long_slots, dtype=torch.bool),
        "target_actions": torch.randn(batch_size, 8, 7),
        "future_mask": torch.ones(batch_size, horizons, dtype=torch.bool),
        "current_latent_target": torch.randn(batch_size, 16, target_dim),
        "future_latent_targets": torch.randn(batch_size, horizons, 16, target_dim),
    }
    outputs = model(
        batch,
        action_condition_mode="scheduled",
        teacher_forcing_probability=0.5,
        router_hard_topk=False,
        compute_teacher_targets=True,
    )
    losses = latent_wam_training_losses(
        outputs,
        batch,
        {"action": 1.0, "nominal_action": 0.25, "world": 1.0, "delta": 0.5, "load_balance": 0.01, "residual": 0.001},
    )
    losses["total_loss"].backward()
    print(
        json.dumps(
            {
                "final_actions": list(outputs["final_actions"].shape),
                "future_latents": list(outputs["predicted_future_latents"].shape),
                "router_probs": list(outputs["router_probs"].shape),
                "selected_experts": list(outputs["topk_experts"].shape),
                "total_loss": float(losses["total_loss"].detach()),
                "world_parameters": world.parameter_count(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
