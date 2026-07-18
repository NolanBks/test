"""Variable-prefix LIBERO action queue for temporal skill replanning."""

from __future__ import annotations

from collections import deque

from mowe_wam.memory import OnlineMemoryState


def canonical_action_to_libero(action, action_statistics=None):
    """Unnormalize motion and convert canonical gripper to LIBERO semantics."""

    converted = action.clone() if hasattr(action, "clone") else action.copy()
    if action_statistics is not None:
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        low = action_statistics["q01"][:6]
        high = action_statistics["q99"][:6]
        mask = action_statistics.get("mask", [True] * 6)[:6]
        if torch is not None and torch.is_tensor(converted):
            low = torch.as_tensor(low, device=converted.device, dtype=converted.dtype)
            high = torch.as_tensor(high, device=converted.device, dtype=converted.dtype)
            mask = torch.as_tensor(mask, device=converted.device, dtype=torch.bool)
            raw = (converted[..., :6] + 1.0) * 0.5 * (high - low) + low
            converted[..., :6] = torch.where(mask, raw, converted[..., :6])
        else:
            import numpy as np

            low = np.asarray(low, dtype=converted.dtype)
            high = np.asarray(high, dtype=converted.dtype)
            mask = np.asarray(mask, dtype=bool)
            raw = (converted[..., :6] + 1.0) * 0.5 * (high - low) + low
            converted[..., :6] = np.where(mask, raw, converted[..., :6])
    converted[..., -1] = 1.0 - 2.0 * (converted[..., -1] >= 0.5)
    return converted


class VariablePrefixActionQueue:
    """Discard every unexecuted suffix and re-query as soon as a prefix ends."""

    def __init__(self, policy_fn, *, max_prefix_steps: int = 8) -> None:
        self.policy_fn = policy_fn
        self.max_prefix_steps = int(max_prefix_steps)
        if self.max_prefix_steps < 1:
            raise ValueError("max_prefix_steps must be positive.")
        self._queue = deque()
        self.query_id = 0
        self.last_metadata = None

    def reset(self) -> None:
        self._queue.clear()
        self.query_id = 0
        self.last_metadata = None

    def _query(self, observation) -> None:
        result = self.policy_fn(observation)
        if isinstance(result, tuple):
            actions, metadata = result
        else:
            actions, metadata = result, {}
        length = len(actions)
        if not 1 <= length <= self.max_prefix_steps:
            raise ValueError(
                f"Temporal policy must return a 1..{self.max_prefix_steps} step prefix."
            )
        self.query_id += 1
        self.last_metadata = dict(metadata or {})
        self.last_metadata["query_id"] = self.query_id
        for step, action in enumerate(actions):
            self._queue.append((self.query_id, step, action))

    def next_action(self, observation):
        if not self._queue:
            self._query(observation)
        query_id, prefix_step, action = self._queue.popleft()
        return action, {"query_id": query_id, "prefix_step": prefix_step, **(self.last_metadata or {})}

    @property
    def remaining(self) -> int:
        return len(self._queue)


class TemporalSkillPolicyAdapter:
    """Observe every step and query the policy exactly on predicted prefix boundaries."""

    def __init__(
        self,
        model,
        image_transform,
        *,
        action_statistics,
        history_length: int = 8,
        long_memory_slots: int = 4,
        flow_seed: int = 7,
    ) -> None:
        self.model = model
        self.image_transform = image_transform
        self.action_statistics = action_statistics
        self.flow_seed = int(flow_seed)
        self.memory = OnlineMemoryState(
            history_length=history_length,
            long_memory_slots=long_memory_slots,
        )
        self.instruction = None
        self.previous_action = None
        self.query_records = []
        self.queue = VariablePrefixActionQueue(
            self._query,
            max_prefix_steps=int(getattr(model, "execution_default_steps", 8)),
        )

    def reset(self, instruction: str) -> None:
        self.reset_episode()
        self.set_instruction(instruction)

    def reset_episode(self) -> None:
        """Clear online state before a new simulator episode/sequence."""

        self.instruction = None
        self.previous_action = None
        self.query_records.clear()
        self.memory.reset()
        self.queue.reset()

    def set_instruction(self, instruction: str) -> bool:
        """Update a subtask goal without erasing same-episode visual memory.

        Any unexecuted action suffix was conditioned on the previous goal and
        must be discarded immediately.
        """

        resolved = str(instruction)
        changed = self.instruction != resolved
        if changed:
            self.instruction = resolved
            self.queue.reset()
        return changed

    def _batch(self, primary_image, wrist_image, proprio=None):
        if self.instruction is None:
            raise RuntimeError("Call reset(instruction) before the first observation.")
        from PIL import Image

        import numpy as np
        import torch

        primary_rgb = Image.fromarray(np.asarray(primary_image, dtype=np.uint8)).convert("RGB")
        wrist_rgb = Image.fromarray(np.asarray(wrist_image, dtype=np.uint8)).convert("RGB")
        primary_pixels = self.image_transform(primary_rgb)
        wrist_pixels = self.image_transform(wrist_rgb)
        self.memory.append(primary_pixels, wrist_pixels, self.previous_action)
        values = self.memory.tensors(action_dim=7)
        batch = {key: value.unsqueeze(0) for key, value in values.items()}
        batch["language"] = [self.instruction]
        if proprio is not None:
            batch["proprio"] = torch.as_tensor(proprio, dtype=torch.float32).reshape(1, -1)
        return batch

    def _query(self, batch):
        prefixes, output = self.model.predict_actions(
            batch,
            flow_seed=self.flow_seed + self.queue.query_id,
        )
        prefix = prefixes[0].detach().cpu()

        def scalar(name, default):
            value = output.get(name)
            if value is None:
                return default
            value = value[0]
            return value.item() if hasattr(value, "item") else value

        record = {
            "prefix_length": len(prefix),
            "route_indices": output["route_indices"][0].detach().cpu().tolist(),
            "execution_reason_code": int(scalar("execution_reason_code", 0)),
            "execution_boundary_position": int(
                scalar("execution_boundary_position", -1)
            ),
            "execution_boundary_entropy": float(
                scalar("execution_boundary_entropy", 0.0)
            ),
            "execution_boundary_margin": float(
                scalar("execution_boundary_margin", 1.0)
            ),
            "execution_motion_jump_l2": float(
                scalar("execution_motion_jump_l2", 0.0)
            ),
            "execution_residual_l2": float(
                scalar("execution_residual_l2", 0.0)
            ),
            "execution_crosses_predicted_boundary": bool(
                scalar("execution_crosses_predicted_boundary", False)
            ),
            "current_view_weights": output["current_view_weights"][0].detach().float().cpu().tolist(),
            "view_order": list(output["view_order"]),
        }
        self.query_records.append(record)
        return list(prefix), record

    def next_canonical_action(self, primary_image, wrist_image, proprio=None):
        batch = self._batch(primary_image, wrist_image, proprio)
        action, metadata = self.queue.next_action(batch)
        self.previous_action = action.detach().cpu() if hasattr(action, "detach") else action
        return action, {**metadata, "queue_remaining": self.queue.remaining}

    def next_libero_action(self, primary_image, wrist_image, proprio=None):
        canonical, metadata = self.next_canonical_action(primary_image, wrist_image, proprio)
        converted = canonical_action_to_libero(canonical, self.action_statistics)
        if hasattr(converted, "detach"):
            converted = converted.detach().cpu().numpy()
        return converted, metadata
