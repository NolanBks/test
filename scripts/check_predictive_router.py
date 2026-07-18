#!/usr/bin/env python3
"""Synthetic forward/backward check for predictive-memory MoWE routing."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.memory import EventMemoryEncoder, EventMemoryState
from mowe_wam.models import MoEActionExperts, MoWEPolicyWrapper, PredictiveExpertRouter, WorldTransitionHead
from mowe_wam.predicates.schema import predicate_dim
from mowe_wam.training.losses import weighted_training_losses
from mowe_wam.training.train_utils import make_synthetic_predictive_batch, set_seed
from mowe_wam.utils.optional import require_torch


class IdentityBackbone:
    def extract_features(self, batch):
        return batch["features"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--history-steps", type=int, default=4)
    parser.add_argument("--num-experts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Only --synthetic is supported by this smoke check.")

    try:
        torch = require_torch()
        set_seed(7)
        memory_state = EventMemoryState(num_experts=args.num_experts)
        batch = make_synthetic_predictive_batch(
            args.batch_size,
            args.hidden_dim,
            args.action_dim,
            args.chunk_size,
            history_steps=args.history_steps,
            num_experts=args.num_experts,
            memory_dim=memory_state.vector_dim,
        )
        memory_encoder = EventMemoryEncoder(memory_state.vector_dim, context_dim=128)
        model = MoWEPolicyWrapper(
            IdentityBackbone(),
            WorldTransitionHead(
                args.hidden_dim,
                args.action_dim,
                predicate_dim(),
                memory_context_dim=128,
                temporal_dim=128,
                temporal_layers=2,
                temporal_heads=8,
                temporal_ff_dim=256,
                max_history_steps=args.history_steps,
            ),
            PredictiveExpertRouter(
                args.hidden_dim,
                predicate_dim(),
                memory_context_dim=128,
                num_experts=args.num_experts,
                top_k=args.top_k,
                state_dim=128,
                transition_dim=64,
            ),
            MoEActionExperts(args.hidden_dim, args.action_dim, args.chunk_size, args.num_experts),
            memory_encoder=memory_encoder,
            predictive=True,
        )
        outputs = model(batch)
        required = (
            "actions",
            "future_predicate_logits",
            "future_predicates",
            "progress_delta",
            "future_risk_logits",
            "future_recovery_logits",
            "memory_context",
            "router_logits",
            "router_probs",
            "topk_experts",
        )
        for key in required:
            if key not in outputs:
                raise AssertionError(f"Missing predictive output: {key}")
            if not torch.isfinite(outputs[key].float()).all():
                raise AssertionError(f"Non-finite predictive output: {key}")
        losses = weighted_training_losses(
            outputs,
            batch,
            {
                "action": 1.0,
                "future_predicate": 0.2,
                "progress_delta": 0.1,
                "future_risk": 0.1,
                "future_recovery": 0.1,
                "phase_router": 0.1,
                "load_balance": 0.01,
                "switch": 0.02,
            },
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.zero_grad(set_to_none=True)
        losses["total_loss"].backward()
        selected_experts = {int(index) for index in outputs["topk_experts"].detach().flatten().tolist()}
        for expert_index, expert in enumerate(model.experts.experts):
            has_gradient = any(
                parameter.grad is not None and bool(parameter.grad.detach().abs().sum() > 0)
                for parameter in expert.parameters()
            )
            if expert_index not in selected_experts and has_gradient:
                raise AssertionError(f"Unselected expert {expert_index} received a gradient in sparse Top-k execution.")
        optimizer.step()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print("Predictive synthetic forward/backward OK")
    print(f"future_predicates shape: {tuple(outputs['future_predicates'].shape)}")
    print(f"memory_context shape: {tuple(outputs['memory_context'].shape)}")
    print(f"router_probs shape: {tuple(outputs['router_probs'].shape)}")
    print(f"selected experts: {sorted(selected_experts)}")
    print(f"total_loss: {float(losses['total_loss'].detach())}")


if __name__ == "__main__":
    main()
