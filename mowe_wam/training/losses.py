"""Loss terms for MoWE-WAM dry runs and future training."""

from __future__ import annotations

from mowe_wam.utils.optional import require_torch


def predicate_loss(pred, target, mask=None):
    torch = require_torch()
    loss = torch.nn.functional.binary_cross_entropy(pred, target, reduction="none")
    if mask is not None:
        loss = loss * mask
    return loss.mean()


def progress_loss(pred, target):
    torch = require_torch()
    return torch.nn.functional.mse_loss(pred, target)


def risk_loss(logits, target):
    torch = require_torch()
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, target)


def progress_delta_loss(pred, target):
    torch = require_torch()
    return torch.nn.functional.smooth_l1_loss(pred, target)


def phase_router_loss(router_logits, target):
    torch = require_torch()
    return torch.nn.functional.cross_entropy(router_logits, target.long().view(-1))


def switch_regularization(router_probs, previous_expert, event_target=None):
    """Penalize changing experts except at labeled event boundaries."""

    torch = require_torch()
    if previous_expert is None:
        return router_probs.new_tensor(0.0)
    previous = previous_expert.to(device=router_probs.device, dtype=torch.long).view(-1)
    valid = (previous >= 0) & (previous < router_probs.shape[-1])
    if not valid.any():
        return router_probs.new_tensor(0.0)
    stay = router_probs[valid].gather(1, previous[valid].unsqueeze(1)).squeeze(1)
    penalty = 1.0 - stay
    if event_target is not None:
        events = event_target.to(device=router_probs.device).view(-1)[valid]
        # Event 0 is ``none``. Switching at a real event should not be penalized.
        penalty = penalty * (events == 0).float()
    return penalty.mean()


def action_loss(pred_actions, target_actions, kind: str = "l1"):
    torch = require_torch()
    if kind == "l1":
        return torch.nn.functional.l1_loss(pred_actions, target_actions)
    if kind == "mse":
        return torch.nn.functional.mse_loss(pred_actions, target_actions)
    raise ValueError(f"Unknown action loss kind: {kind}")


def load_balance_loss(router_probs):
    torch = require_torch()
    mean_probs = router_probs.mean(dim=0)
    target = torch.full_like(mean_probs, 1.0 / router_probs.shape[-1])
    return torch.nn.functional.mse_loss(mean_probs, target)


def temporal_smoothness_loss(router_probs, event_mask=None):
    require_torch()
    if router_probs.dim() < 3 or router_probs.shape[1] < 2:
        return router_probs.new_tensor(0.0)
    diffs = (router_probs[:, 1:] - router_probs[:, :-1]).abs()
    if event_mask is not None:
        diffs = diffs * event_mask[:, 1:].unsqueeze(-1)
    return diffs.mean()


def weighted_training_losses(outputs, batch, weights: dict[str, float], action_kind: str = "l1"):
    """Compute configured MoWE-WAM training losses and their weighted total."""

    losses = {
        "action_loss": action_loss(outputs["actions"], batch["actions"], kind=action_kind),
        "load_balance_loss": load_balance_loss(outputs["router_probs"]),
        "temporal_smoothness_loss": temporal_smoothness_loss(outputs["router_probs"]),
    }
    if "future_predicate_logits" in outputs:
        losses.update(
            {
                "future_predicate_loss": risk_loss(outputs["future_predicate_logits"], batch["future_predicates"]),
                "progress_delta_loss": progress_delta_loss(outputs["progress_delta"], batch["progress_delta"]),
                "future_risk_loss": risk_loss(outputs["future_risk_logits"], batch["future_risk"]),
                "future_recovery_loss": risk_loss(outputs["future_recovery_logits"], batch["future_recovery"]),
                "phase_router_loss": phase_router_loss(outputs["router_logits"], batch["phase_target"]),
                "switch_loss": switch_regularization(
                    outputs["router_probs"], batch.get("previous_expert"), batch.get("event_target")
                ),
            }
        )
    else:
        losses.update(
            {
                "predicate_loss": predicate_loss(outputs["predicates"], batch["predicates"]),
                "progress_loss": progress_loss(outputs["progress"], batch["progress"]),
                "risk_loss": risk_loss(outputs["risk_logits"], batch["risk"]),
            }
        )
    total = outputs["actions"].new_tensor(0.0)
    key_map = {
        "action_loss": "action",
        "predicate_loss": "predicate",
        "progress_loss": "progress",
        "risk_loss": "risk",
        "load_balance_loss": "load_balance",
        "temporal_smoothness_loss": "temporal_smoothness",
        "future_predicate_loss": "future_predicate",
        "progress_delta_loss": "progress_delta",
        "future_risk_loss": "future_risk",
        "future_recovery_loss": "future_recovery",
        "phase_router_loss": "phase_router",
        "switch_loss": "switch",
    }
    for loss_name, loss_value in losses.items():
        total = total + float(weights.get(key_map[loss_name], 0.0)) * loss_value
    losses["total_loss"] = total
    return losses
