#!/usr/bin/env python3
"""Synthetic forward check for MoWE-WAM model components."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.models import ExpertRouter, MoEActionExperts, MoWEPolicyWrapper, WorldPredicateHead
from mowe_wam.predicates.schema import predicate_dim
from mowe_wam.training.train_utils import make_synthetic_batch, set_seed


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
    parser.add_argument("--num-experts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("Only --synthetic is implemented before upstream feature binding.")

    try:
        set_seed(7)
        batch = make_synthetic_batch(args.batch_size, args.hidden_dim, args.action_dim, args.chunk_size)
        wrapper = MoWEPolicyWrapper(
            IdentityBackbone(),
            WorldPredicateHead(args.hidden_dim, predicate_dim(), [512]),
            ExpertRouter(args.hidden_dim, predicate_dim(), args.num_experts, args.top_k),
            MoEActionExperts(args.hidden_dim, args.action_dim, args.chunk_size, args.num_experts),
        )
        outputs = wrapper(batch)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    for key in ["actions", "predicate_logits", "predicates", "router_logits", "router_probs", "topk_experts", "expert_actions"]:
        if key not in outputs:
            raise AssertionError(f"Missing model output key: {key}")
    print("Synthetic forward OK")
    print(f"actions shape: {tuple(outputs['actions'].shape)}")
    print(f"router_probs shape: {tuple(outputs['router_probs'].shape)}")
    print(f"topk_experts shape: {tuple(outputs['topk_experts'].shape)}")


if __name__ == "__main__":
    main()
