#!/usr/bin/env python3
"""Two-step router/expert dry run on synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.models import ExpertRouter, MoEActionExperts, MoWEPolicyWrapper, WorldPredicateHead
from mowe_wam.predicates.schema import predicate_dim
from mowe_wam.training.losses import action_loss, load_balance_loss, predicate_loss
from mowe_wam.training.train_utils import make_synthetic_batch, set_seed
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


class IdentityBackbone:
    def extract_features(self, batch):
        return batch["features"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mowe_wam/train_mowe_router.yaml")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=2)
    args = parser.parse_args()
    if not (args.mock and args.dry_run):
        raise SystemExit("Only --mock --dry-run is implemented locally.")

    try:
        torch = require_torch()
        cfg = load_config(args.config)
        model_cfg = cfg.get("model", {})
        hidden_dim = int(model_cfg.get("hidden_dim", 1024))
        action_dim = int(model_cfg.get("action_dim", 7))
        chunk_size = int(model_cfg.get("chunk_size", 8))
        num_experts = int(model_cfg.get("num_experts", 5))
        top_k = int(model_cfg.get("top_k", 2))
        set_seed(int(cfg.get("seed", 7)))
        model = MoWEPolicyWrapper(
            IdentityBackbone(),
            WorldPredicateHead(hidden_dim, predicate_dim(), model_cfg.get("hidden_layers", [512])),
            ExpertRouter(hidden_dim, predicate_dim(), num_experts, top_k),
            MoEActionExperts(hidden_dim, action_dim, chunk_size, num_experts),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))
        for step in range(args.max_steps):
            batch = make_synthetic_batch(2, hidden_dim, action_dim, chunk_size)
            outputs = model(batch, use_oracle_predicates=True)
            losses = {
                "action_loss": action_loss(outputs["actions"], batch["actions"]),
                "predicate_loss": predicate_loss(outputs["predicates"], batch["predicates"]),
                "load_balance_loss": load_balance_loss(outputs["router_probs"]),
            }
            total = sum(losses.values())
            opt.zero_grad()
            total.backward()
            opt.step()
            print({"step": step, **{name: float(value.detach()) for name, value in losses.items()}})
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
