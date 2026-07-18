"""Fixed-size event-predicate memory for long-horizon expert routing.

The first version deliberately stores only low-dimensional routing evidence. It
does not retain images, perform retrieval, or generate language summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.predicates.schema import predicate_index
from mowe_wam.utils.optional import require_torch

EVENT_TYPES: tuple[str, ...] = (
    "none",
    "grasp_acquired",
    "contact_lost",
    "progress_stall",
    "subgoal_complete",
    "recovery_started",
)
EVENT_TO_INDEX = {name: idx for idx, name in enumerate(EVENT_TYPES)}


def _scalar(values: Any, name: str, default: float = 0.0) -> float:
    if isinstance(values, dict):
        return float(values.get(name, default))
    index = predicate_index(name)
    try:
        return float(values[index])
    except (IndexError, TypeError):
        return default


@dataclass
class EventMemoryState:
    """Deterministic routing state reconstructed without future information."""

    num_experts: int = 5
    stall_threshold: int = 3
    progress_epsilon: float = 0.01
    previous_expert: int = -1
    last_event: str = "none"
    stall_steps: int = 0
    retry_count: int = 0
    last_progress: float = 0.0
    last_risk: float = 0.0
    last_grasped: float = 0.0
    last_near_goal: float = 0.0

    @property
    def vector_dim(self) -> int:
        return self.num_experts + len(EVENT_TYPES) + 6

    def reset(self) -> None:
        self.previous_expert = -1
        self.last_event = "none"
        self.stall_steps = 0
        self.retry_count = 0
        self.last_progress = 0.0
        self.last_risk = 0.0
        self.last_grasped = 0.0
        self.last_near_goal = 0.0

    def as_list(self) -> list[float]:
        previous = [0.0] * self.num_experts
        if 0 <= self.previous_expert < self.num_experts:
            previous[self.previous_expert] = 1.0
        event = [0.0] * len(EVENT_TYPES)
        event[EVENT_TO_INDEX.get(self.last_event, 0)] = 1.0
        # Counts are clipped to keep the contract normalized and stable.
        scalars = [
            min(self.stall_steps / 10.0, 1.0),
            min(self.retry_count / 5.0, 1.0),
            max(0.0, min(1.0, self.last_progress)),
            max(0.0, min(1.0, self.last_risk)),
            max(0.0, min(1.0, self.last_grasped)),
            max(0.0, min(1.0, self.last_near_goal)),
        ]
        return previous + event + scalars

    def as_tensor(self, device=None):
        torch_mod = require_torch()
        return torch_mod.tensor(self.as_list(), dtype=torch_mod.float32, device=device)

    def update(self, predicates: Any, progress: float | None = None, risk: float | None = None, selected_expert: int | None = None) -> str:
        """Update state from the *current* outcome and return the written event.

        Callers constructing a training snapshot must record ``as_list()``
        before calling this method so the snapshot at t never leaks event t.
        """

        current_progress = _scalar(predicates, "progress_score") if progress is None else float(progress)
        current_risk = _scalar(predicates, "failure_risk") if risk is None else float(risk)
        current_grasped = _scalar(predicates, "object_grasped")
        current_goal = _scalar(predicates, "near_goal_region")
        recovery = _scalar(predicates, "needs_recovery")

        progress_delta = current_progress - self.last_progress
        if progress_delta <= self.progress_epsilon:
            self.stall_steps += 1
        else:
            self.stall_steps = 0

        event = "none"
        if recovery >= 0.5:
            event = "recovery_started"
            self.retry_count += 1
        elif self.last_grasped < 0.5 and current_grasped >= 0.5:
            event = "grasp_acquired"
        elif self.last_grasped >= 0.5 and current_grasped < 0.5 and current_risk >= self.last_risk:
            event = "contact_lost"
            self.retry_count += 1
        elif self.last_near_goal < 0.8 and current_goal >= 0.8 and progress_delta > self.progress_epsilon:
            event = "subgoal_complete"
        elif self.stall_steps >= self.stall_threshold:
            event = "progress_stall"

        if selected_expert is not None:
            self.previous_expert = int(selected_expert)
        self.last_event = event
        self.last_progress = current_progress
        self.last_risk = current_risk
        self.last_grasped = current_grasped
        self.last_near_goal = current_goal
        return event


def build_memory_snapshots(
    predicate_sequence: Iterable[Any],
    progress_sequence: Iterable[float] | None = None,
    risk_sequence: Iterable[float] | None = None,
    expert_sequence: Iterable[int] | None = None,
    num_experts: int = 5,
) -> tuple[list[list[float]], list[int]]:
    """Return leak-free snapshots and current-step event labels for a trajectory."""

    predicates = list(predicate_sequence)
    progress = list(progress_sequence) if progress_sequence is not None else [None] * len(predicates)
    risk = list(risk_sequence) if risk_sequence is not None else [None] * len(predicates)
    experts = list(expert_sequence) if expert_sequence is not None else [None] * len(predicates)
    if not (len(predicates) == len(progress) == len(risk) == len(experts)):
        raise ValueError("Memory sequences must have equal lengths.")

    state = EventMemoryState(num_experts=num_experts)
    snapshots: list[list[float]] = []
    events: list[int] = []
    for pred, prog, step_risk, expert in zip(predicates, progress, risk, experts):
        snapshots.append(state.as_list())
        event_name = state.update(pred, progress=prog, risk=step_risk, selected_expert=expert)
        events.append(EVENT_TO_INDEX[event_name])
    return snapshots, events


class EventMemoryEncoder(nn.Module if nn is not None else object):
    """Project structured memory state into a compact routing context."""

    def __init__(self, memory_dim: int, context_dim: int = 128) -> None:
        require_torch()
        super().__init__()
        self.memory_dim = int(memory_dim)
        self.context_dim = int(context_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.memory_dim),
            nn.Linear(self.memory_dim, self.context_dim),
            nn.GELU(),
            nn.Linear(self.context_dim, self.context_dim),
        )

    def forward(self, memory_state):
        if memory_state.shape[-1] != self.memory_dim:
            raise ValueError(f"Expected memory dim {self.memory_dim}, got {memory_state.shape[-1]}.")
        return self.net(memory_state)
