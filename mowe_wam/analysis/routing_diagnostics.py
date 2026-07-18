"""Small routing diagnostics that do not claim task success."""

from __future__ import annotations

from mowe_wam.utils.optional import require_torch


def routing_summary(router_probs, expert_weights=None) -> dict:
    torch = require_torch()
    probabilities = router_probs.float()
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
    used = expert_weights.float() if expert_weights is not None else probabilities
    reduce_dims = tuple(range(probabilities.ndim - 1))
    summary = {
        "entropy_mean": float(entropy.mean()),
        "probability_mean": probabilities.mean(dim=reduce_dims).detach().cpu().tolist(),
        "expert_weight_mean": used.mean(dim=reduce_dims).detach().cpu().tolist(),
        "top1_counts": torch.bincount(
            probabilities.argmax(dim=-1).reshape(-1), minlength=probabilities.shape[-1]
        ).detach().cpu().tolist(),
    }
    if probabilities.ndim == 3:
        summary["probability_mean_by_position"] = probabilities.mean(dim=0).detach().cpu().tolist()
        summary["entropy_mean_by_position"] = entropy.mean(dim=0).detach().cpu().tolist()
    return summary


def temporal_route_metrics(route_indices, labels, mask) -> dict[str, object]:
    require_torch()
    if route_indices.shape != labels.shape or route_indices.shape != mask.shape:
        raise ValueError("route_indices, labels, and mask must have identical [B, T] shapes.")
    valid = mask.bool() & labels.ge(0)
    correct = route_indices.eq(labels) & valid
    accuracy_by_position = [
        float(correct[:, index].sum().float() / valid[:, index].sum().clamp_min(1))
        for index in range(labels.shape[1])
    ]
    transition_valid = valid[:, 1:] & valid[:, :-1]
    truth = labels[:, 1:].ne(labels[:, :-1]) & transition_valid
    predicted = route_indices[:, 1:].ne(route_indices[:, :-1]) & transition_valid
    true_positive = (truth & predicted).sum().float()
    precision = true_positive / predicted.sum().clamp_min(1)
    recall = true_positive / truth.sum().clamp_min(1)
    f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)
    return {
        "accuracy_by_position": accuracy_by_position,
        "current_skill_accuracy": accuracy_by_position[0],
        "boundary_precision": float(precision),
        "boundary_recall": float(recall),
        "boundary_f1": float(f1),
    }


def action_correction_summary(nominal, final, target) -> dict[str, float]:
    return {
        "nominal_target_l1": float((nominal.float() - target.float()).abs().mean()),
        "final_target_l1": float((final.float() - target.float()).abs().mean()),
        "residual_l1": float((final.float() - nominal.float()).abs().mean()),
    }


def motion_gripper_summary(nominal_motion, final_motion, target_motion, gripper_logits, target_gripper):
    require_torch()
    predicted_gripper = gripper_logits.ge(0)
    return {
        "nominal_motion_target_l1": float(
            (nominal_motion.float() - target_motion.float()).abs().mean()
        ),
        "final_motion_target_l1": float((final_motion.float() - target_motion.float()).abs().mean()),
        "motion_residual_l1": float((final_motion.float() - nominal_motion.float()).abs().mean()),
        "gripper_accuracy": float(predicted_gripper.eq(target_gripper.bool()).float().mean()),
    }
