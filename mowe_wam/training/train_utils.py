"""Training smoke-test helpers."""

from __future__ import annotations

import random

from mowe_wam.predicates.schema import predicate_dim
from mowe_wam.utils.optional import require_torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_batch(batch_size: int, hidden_dim: int, action_dim: int, chunk_size: int):
    torch = require_torch()
    return {
        "features": torch.randn(batch_size, hidden_dim),
        "predicates": torch.rand(batch_size, predicate_dim()),
        "actions": torch.randn(batch_size, chunk_size, action_dim),
    }


def make_synthetic_predictive_batch(
    batch_size: int,
    hidden_dim: int,
    action_dim: int,
    chunk_size: int,
    history_steps: int = 4,
    num_experts: int = 5,
    memory_dim: int | None = None,
):
    """Create a complete future-predictive routing batch for smoke checks."""

    torch = require_torch()
    pred_dim = predicate_dim()
    resolved_memory_dim = memory_dim or (num_experts + 6 + 6)
    future_predicates = torch.rand(batch_size, pred_dim)
    return {
        "features": torch.randn(batch_size, hidden_dim),
        "actions": torch.randn(batch_size, chunk_size, action_dim),
        "history_actions": torch.randn(batch_size, history_steps, action_dim),
        "history_predicates": torch.rand(batch_size, history_steps, pred_dim),
        "current_predicates": torch.rand(batch_size, pred_dim),
        "future_predicates": future_predicates,
        "progress_delta": torch.randn(batch_size, 1).clamp(-1.0, 1.0),
        "future_risk": torch.rand(batch_size, 1),
        "future_recovery": torch.rand(batch_size, 1),
        "memory_state": torch.rand(batch_size, resolved_memory_dim),
        "event_target": torch.randint(0, 6, (batch_size,)),
        "phase_target": torch.randint(0, num_experts, (batch_size,)),
        "previous_expert": torch.randint(-1, num_experts, (batch_size,)),
    }
