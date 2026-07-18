"""Loss primitives for conditional 6D motion flow and binary gripper control."""

from __future__ import annotations

from mowe_wam.utils.optional import require_torch


def _masked_mean(values, timestep_mask=None):
    if timestep_mask is None:
        return values.mean()
    weights = timestep_mask.to(device=values.device, dtype=values.dtype)
    if weights.shape != values.shape:
        raise ValueError(f"mask shape {tuple(weights.shape)} does not match {tuple(values.shape)}.")
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def conditional_flow_matching_loss(pred_velocity, target_velocity, timestep_mask=None):
    if pred_velocity.shape != target_velocity.shape or pred_velocity.shape[-1] != 6:
        raise ValueError("Flow velocity tensors must have identical [B, T, 6] shapes.")
    values = (pred_velocity.float() - target_velocity.float()).square().mean(dim=-1)
    return _masked_mean(values, timestep_mask)


def gripper_binary_loss(gripper_logits, binary_targets, timestep_mask=None):
    torch = require_torch()
    if gripper_logits.shape != binary_targets.shape or gripper_logits.shape[-1] != 1:
        raise ValueError("Gripper logits and targets must have identical [B, T, 1] shapes.")
    target = binary_targets.to(device=gripper_logits.device, dtype=gripper_logits.dtype)
    if not torch.all((target == 0) | (target == 1)):
        raise ValueError("Gripper BCE targets must be canonical binary 0/1 values.")
    values = torch.nn.functional.binary_cross_entropy_with_logits(
        gripper_logits.float(), target.float(), reduction="none"
    ).squeeze(-1)
    return _masked_mean(values, timestep_mask)


def masked_route_cross_entropy(router_logits, labels, timestep_mask, class_weights=None):
    torch = require_torch()
    if router_logits.ndim != 3 or labels.shape != router_logits.shape[:2]:
        raise ValueError("Router logits/labels must have shapes [B, T, R] and [B, T].")
    valid = timestep_mask.bool() & labels.ge(0) & labels.lt(router_logits.shape[-1])
    if not bool(valid.any()):
        return router_logits.float().sum() * 0.0
    weights = None
    if class_weights is not None:
        weights = class_weights.to(device=router_logits.device, dtype=torch.float32)
    return torch.nn.functional.cross_entropy(
        router_logits.float()[valid], labels.long()[valid], weight=weights
    )


def residual_regularization(residuals, mask=None):
    if residuals.ndim != 3 or residuals.shape[-1] != 6:
        raise ValueError("residuals must have shape [B, T, 6].")
    values = residuals.float().square().mean(dim=-1)
    return _masked_mean(values, mask)
