"""Mechanism diagnostics for multi-horizon latent prediction."""

from __future__ import annotations

from mowe_wam.utils.optional import require_torch


def horizon_latent_metrics(predicted, target, horizons=(1, 4, 8)) -> dict[str, dict[str, float]]:
    torch = require_torch()
    pred = predicted.float()
    truth = target.to(pred.device).float()
    if pred.shape != truth.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {truth.shape}")
    metrics = {}
    for index, horizon in enumerate(horizons):
        metrics[str(horizon)] = {
            "mse": float((pred[:, index] - truth[:, index]).square().mean()),
            "smooth_l1": float(torch.nn.functional.smooth_l1_loss(pred[:, index], truth[:, index])),
            "cosine_distance": float(
                1.0 - torch.nn.functional.cosine_similarity(pred[:, index], truth[:, index], dim=-1).mean()
            ),
        }
    return metrics


def current_copy_baseline(current_target, future_target, horizons=(1, 4, 8)):
    copied = current_target.unsqueeze(1).expand_as(future_target)
    return horizon_latent_metrics(copied, future_target, horizons=horizons)

