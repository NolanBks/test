#!/usr/bin/env python3
"""Two-step predicate-head dry run on synthetic data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.models.world_head import WorldPredicateHead
from mowe_wam.predicates.schema import predicate_dim
from mowe_wam.training.losses import predicate_loss
from mowe_wam.training.train_utils import make_synthetic_batch, set_seed
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mowe_wam/train_predicate_head.yaml")
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
        set_seed(int(cfg.get("seed", 7)))
        head = WorldPredicateHead(hidden_dim, predicate_dim(), model_cfg.get("hidden_layers", [512]))
        opt = torch.optim.AdamW(head.parameters(), lr=float(cfg.get("learning_rate", 1e-4)))
        for step in range(args.max_steps):
            batch = make_synthetic_batch(2, hidden_dim, 7, 8)
            outputs = head(batch["features"])
            loss = predicate_loss(outputs["predicates"], batch["predicates"])
            opt.zero_grad()
            loss.backward()
            opt.step()
            print({"step": step, "predicate_loss": float(loss.detach())})
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
