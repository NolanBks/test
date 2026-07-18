"""Official CALVIN ``reset``/``step(obs, goal)`` bridge for Flow-WAM."""

from __future__ import annotations

from typing import Any

from mowe_wam.evaluation.libero_temporal_policy import TemporalSkillPolicyAdapter


def _goal_text(goal: Any) -> str:
    if isinstance(goal, str):
        text = goal
    elif isinstance(goal, bytes):
        text = goal.decode("utf-8")
    elif isinstance(goal, dict):
        text = None
        for key in ("language_instruction", "language", "lang", "annotation", "text"):
            value = goal.get(key)
            if isinstance(value, (str, bytes)):
                text = value.decode("utf-8") if isinstance(value, bytes) else value
                break
        if text is None:
            raise ValueError(
                "CALVIN goal must expose raw language text; precomputed benchmark language embeddings "
                "cannot preserve the OpenVLA prompt contract."
            )
    else:
        raise ValueError("CALVIN goal must be raw text or a mapping containing raw text.")
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        raise ValueError("CALVIN language goal cannot be empty.")
    return normalized


def _observation_views(obs: dict[str, Any]):
    rgb = obs.get("rgb_obs", obs)
    primary = rgb.get("rgb_static", rgb.get("image_primary"))
    wrist = rgb.get("rgb_gripper", rgb.get("image_wrist"))
    if primary is None or wrist is None:
        raise KeyError(
            "CALVIN observation requires rgb_obs.rgb_static and rgb_obs.rgb_gripper."
        )
    proprio = obs.get("robot_obs", obs.get("proprio"))
    return primary, wrist, proprio


class CalvinTemporalPolicyAdapter:
    """CALVIN CustomModel-compatible adapter with goal-boundary replanning."""

    def __init__(
        self,
        model,
        image_transform,
        action_adapter,
        *,
        history_length: int = 8,
        long_memory_slots: int = 4,
        flow_seed: int = 7,
        use_proprio: bool = False,
        preserve_memory_across_subtasks: bool = True,
    ) -> None:
        self.action_adapter = action_adapter
        self.base_flow_seed = int(flow_seed)
        self.use_proprio = bool(use_proprio)
        self.preserve_memory_across_subtasks = bool(preserve_memory_across_subtasks)
        self.sequence_index = -1
        self.current_goal = None
        self.last_step_metadata = None
        self.temporal = TemporalSkillPolicyAdapter(
            model,
            image_transform,
            action_statistics=None,
            history_length=history_length,
            long_memory_slots=long_memory_slots,
            flow_seed=self.base_flow_seed,
        )

    def reset_sequence(self) -> None:
        """Clear all online state at the official environment sequence reset."""

        self.sequence_index += 1
        self.current_goal = None
        self.last_step_metadata = None
        self.temporal.flow_seed = self.base_flow_seed + self.sequence_index * 10_000
        self.temporal.reset_episode()

    def reset(self) -> None:
        """Official CALVIN callback, currently invoked before every subtask.

        The official evaluator does not expose a distinct sequence-reset model
        callback.  Formal MoWE evaluation therefore uses the local bridge to
        call :meth:`reset_sequence` at environment reset, while this method
        only invalidates the previous goal/action suffix by default.
        """

        if self.sequence_index < 0 or not self.preserve_memory_across_subtasks:
            self.reset_sequence()
            return
        self.current_goal = None
        self.last_step_metadata = None
        self.temporal.instruction = None
        self.temporal.queue.reset()

    def step(self, obs, goal):
        """Official CALVIN callback: return one raw 7D environment action."""

        if self.sequence_index < 0:
            raise RuntimeError("CALVIN evaluator must call reset() before step().")
        instruction = _goal_text(goal)
        goal_changed = self.temporal.set_instruction(instruction)
        self.current_goal = instruction
        primary, wrist, proprio = _observation_views(obs)
        canonical, metadata = self.temporal.next_canonical_action(
            primary,
            wrist,
            proprio if self.use_proprio else None,
        )
        action = self.action_adapter.from_shared_action(canonical)
        if hasattr(action, "detach"):
            action = action.detach().float().cpu().numpy()
        self.last_step_metadata = {
            **metadata,
            "goal_changed": bool(goal_changed),
            "sequence_index": self.sequence_index,
            "goal": instruction,
            "action_contract": self.action_adapter.contract(),
            "preserve_memory_across_subtasks": self.preserve_memory_across_subtasks,
        }
        return action

    @property
    def query_records(self):
        return self.temporal.query_records
