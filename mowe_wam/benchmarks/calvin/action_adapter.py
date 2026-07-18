"""Explicit conversion between CALVIN Cartesian actions and MoWE actions."""

from __future__ import annotations

from typing import Any, Sequence


class CalvinActionAdapter:
    """Normalize six CALVIN motion dimensions and canonicalize the gripper.

    Statistics must be computed from the allowed CALVIN training split.  No
    LIBERO statistics or guessed controller ranges are accepted.
    """

    def __init__(
        self,
        *,
        motion_q01: Sequence[float],
        motion_q99: Sequence[float],
        motion_mask: Sequence[bool] = (True, True, True, True, True, True),
        action_mode: str = "relative_cartesian",
        rotation_representation: str = "euler_xyz",
        gripper_open_value: float,
        gripper_closed_value: float,
        clip_normalized_motion: bool = True,
    ) -> None:
        if action_mode not in {"relative_cartesian", "absolute_cartesian"}:
            raise ValueError("CALVIN v1 adapter supports relative or absolute Cartesian actions.")
        if rotation_representation != "euler_xyz":
            raise ValueError("CALVIN v1 adapter requires explicit euler_xyz rotation semantics.")
        if len(motion_q01) != 6 or len(motion_q99) != 6 or len(motion_mask) != 6:
            raise ValueError("CALVIN motion statistics and mask must contain six values.")
        if any(float(high) <= float(low) for low, high in zip(motion_q01, motion_q99)):
            raise ValueError("Each CALVIN motion q99 must be larger than q01.")
        if float(gripper_open_value) == float(gripper_closed_value):
            raise ValueError("CALVIN open and closed gripper values must differ.")
        self.motion_q01 = tuple(float(value) for value in motion_q01)
        self.motion_q99 = tuple(float(value) for value in motion_q99)
        self.motion_mask = tuple(bool(value) for value in motion_mask)
        self.action_mode = action_mode
        self.rotation_representation = rotation_representation
        self.gripper_open_value = float(gripper_open_value)
        self.gripper_closed_value = float(gripper_closed_value)
        self.clip_normalized_motion = bool(clip_normalized_motion)

    @classmethod
    def from_config(cls, config: dict[str, Any]):
        required = (
            "motion_q01",
            "motion_q99",
            "gripper_open_value",
            "gripper_closed_value",
        )
        missing = [
            name
            for name in required
            if config.get(name) is None or config.get(name) == "TBD"
        ]
        if missing:
            raise ValueError(
                "CALVIN action adapter requires ABC-train-derived values for: "
                f"{missing}."
            )
        return cls(
            motion_q01=config["motion_q01"],
            motion_q99=config["motion_q99"],
            motion_mask=config.get("motion_mask", [True] * 6),
            action_mode=config.get("action_mode", "relative_cartesian"),
            rotation_representation=config.get("rotation_representation", "euler_xyz"),
            gripper_open_value=config["gripper_open_value"],
            gripper_closed_value=config["gripper_closed_value"],
            clip_normalized_motion=bool(config.get("clip_normalized_motion", True)),
        )

    @staticmethod
    def _backend(action):
        try:
            import torch
        except ModuleNotFoundError:
            torch = None
        if torch is not None and torch.is_tensor(action):
            return "torch", torch
        try:
            import numpy as np
        except ModuleNotFoundError as exc:
            raise RuntimeError("NumPy is required for non-tensor CALVIN actions.") from exc
        return "numpy", np

    def to_shared_action(self, calvin_action):
        backend, module = self._backend(calvin_action)
        converted = calvin_action.clone() if backend == "torch" else module.array(calvin_action, copy=True)
        if converted.shape[-1] != 7:
            raise ValueError("CALVIN Cartesian action must have seven dimensions.")
        if backend == "torch":
            low = module.as_tensor(self.motion_q01, dtype=converted.dtype, device=converted.device)
            high = module.as_tensor(self.motion_q99, dtype=converted.dtype, device=converted.device)
            mask = module.as_tensor(self.motion_mask, dtype=module.bool, device=converted.device)
        else:
            low = module.asarray(self.motion_q01, dtype=converted.dtype)
            high = module.asarray(self.motion_q99, dtype=converted.dtype)
            mask = module.asarray(self.motion_mask, dtype=bool)
        normalized = 2.0 * (converted[..., :6] - low) / (high - low) - 1.0
        if self.clip_normalized_motion:
            normalized = normalized.clip(-1.0, 1.0)
        converted[..., :6] = module.where(mask, normalized, converted[..., :6])
        distance_open = module.abs(converted[..., 6] - self.gripper_open_value)
        distance_closed = module.abs(converted[..., 6] - self.gripper_closed_value)
        converted[..., 6] = (distance_closed < distance_open).to(converted.dtype) if backend == "torch" else (
            distance_closed < distance_open
        ).astype(converted.dtype)
        return converted

    def from_shared_action(self, shared_action):
        backend, module = self._backend(shared_action)
        converted = shared_action.clone() if backend == "torch" else module.array(shared_action, copy=True)
        if converted.shape[-1] != 7:
            raise ValueError("MoWE shared action must have seven dimensions.")
        if backend == "torch":
            low = module.as_tensor(self.motion_q01, dtype=converted.dtype, device=converted.device)
            high = module.as_tensor(self.motion_q99, dtype=converted.dtype, device=converted.device)
            mask = module.as_tensor(self.motion_mask, dtype=module.bool, device=converted.device)
        else:
            low = module.asarray(self.motion_q01, dtype=converted.dtype)
            high = module.asarray(self.motion_q99, dtype=converted.dtype)
            mask = module.asarray(self.motion_mask, dtype=bool)
        raw = (converted[..., :6].clip(-1.0, 1.0) + 1.0) * 0.5 * (high - low) + low
        converted[..., :6] = module.where(mask, raw, converted[..., :6])
        closed = converted[..., 6] >= 0.5
        if backend == "torch":
            converted[..., 6] = module.where(
                closed,
                converted.new_tensor(self.gripper_closed_value),
                converted.new_tensor(self.gripper_open_value),
            )
        else:
            converted[..., 6] = module.where(
                closed, self.gripper_closed_value, self.gripper_open_value
            )
        return converted

    def contract(self) -> dict[str, Any]:
        return {
            "action_mode": self.action_mode,
            "rotation_representation": self.rotation_representation,
            "motion_q01": list(self.motion_q01),
            "motion_q99": list(self.motion_q99),
            "motion_mask": list(self.motion_mask),
            "gripper_open_value": self.gripper_open_value,
            "gripper_closed_value": self.gripper_closed_value,
            "shared_gripper_contract": "0=open,1=closed",
        }
