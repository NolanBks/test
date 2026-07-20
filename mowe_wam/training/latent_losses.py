"""Joint objectives for latent world prediction and residual action routing."""

from __future__ import annotations

import math

from mowe_wam.utils.optional import require_torch
from mowe_wam.training.flow_matching import (
    conditional_flow_matching_loss,
    gripper_binary_loss,
    masked_route_cross_entropy,
    residual_regularization,
)


def _masked_horizon_mean(
    values,
    mask,
    sample_weight=None,
    horizon_weight=None,
):
    weights = mask.to(device=values.device, dtype=values.dtype)
    if sample_weight is not None:
        weights = weights * sample_weight.to(device=values.device, dtype=values.dtype).unsqueeze(1)
    if horizon_weight is not None:
        weights = weights * horizon_weight.to(
            device=values.device, dtype=values.dtype
        ).reshape(1, -1)
    return (values * weights).sum() / weights.sum().clamp_min(1e-6)


def _configured_horizon_weights(reference, horizon_mask, config):
    """Return immutable loss weights aligned with the configured horizon order."""

    torch = require_torch()
    raw = dict(config or {}).get("horizon_weights")
    horizon_count = int(horizon_mask.shape[1])
    if raw is None:
        return torch.ones(horizon_count, device=reference.device, dtype=reference.dtype)
    if not isinstance(raw, (list, tuple)) or len(raw) != horizon_count:
        raise ValueError(
            "world_prediction_loss.horizon_weights must contain one positive "
            "value per configured future horizon."
        )
    weights = torch.as_tensor(raw, device=reference.device, dtype=reference.dtype)
    if not bool(torch.isfinite(weights).all()) or bool(weights.le(0).any()):
        raise ValueError("world_prediction_loss.horizon_weights must be finite and positive.")
    return weights


def latent_prediction_loss(
    prediction,
    target,
    horizon_mask,
    sample_weight=None,
    horizon_weight=None,
):
    """Cosine plus Smooth-L1 distance, averaged over valid horizons."""

    torch = require_torch()
    pred = prediction.float()
    truth = target.to(device=pred.device, dtype=pred.dtype)
    smooth_l1 = torch.nn.functional.smooth_l1_loss(pred, truth, reduction="none").mean(dim=(-1, -2))
    cosine = 1.0 - torch.nn.functional.cosine_similarity(pred, truth, dim=-1).mean(dim=-1)
    return _masked_horizon_mean(
        cosine + 0.5 * smooth_l1,
        horizon_mask,
        sample_weight=sample_weight,
        horizon_weight=horizon_weight,
    )


def latent_prediction_components(
    prediction,
    target,
    horizon_mask,
    sample_weight=None,
    horizon_weight=None,
):
    torch = require_torch()
    pred = prediction.float()
    truth = target.to(device=pred.device, dtype=pred.dtype)
    smooth_l1 = torch.nn.functional.smooth_l1_loss(pred, truth, reduction="none").mean(dim=(-1, -2))
    cosine = 1.0 - torch.nn.functional.cosine_similarity(pred, truth, dim=-1).mean(dim=-1)
    return (
        _masked_horizon_mean(
            cosine, horizon_mask, sample_weight, horizon_weight
        ),
        _masked_horizon_mean(
            smooth_l1, horizon_mask, sample_weight, horizon_weight
        ),
    )


def delta_prediction_components(
    prediction,
    target,
    horizon_mask,
    *,
    sample_weight=None,
    horizon_weight=None,
    config=None,
):
    """Magnitude-aware delta objective with bounded per-horizon RMS scaling.

    Delta directions are ill-defined when the frozen visual target barely
    changes.  Weighting token cosine by target magnitude prevents those nearly
    static tokens from dominating, while the Smooth-L1 branch remains active.
    Optional batch/horizon RMS normalization equalizes the raw scale of short
    and long horizons without changing the model's raw-delta output contract.
    """

    torch = require_torch()
    cfg = dict(config or {})
    pred = prediction.float()
    truth = target.to(device=pred.device, dtype=pred.dtype)

    raw_smooth = torch.nn.functional.smooth_l1_loss(
        pred, truth, reduction="none"
    ).mean(dim=(-1, -2))
    rms_cfg = dict(cfg.get("delta_rms_normalization") or {})
    rms_mode = str(rms_cfg.get("mode", "none"))
    per_sample_target_power = truth.square().mean(dim=(2, 3))
    valid_horizon = horizon_mask.to(
        device=truth.device, dtype=per_sample_target_power.dtype
    )
    target_rms = (
        (per_sample_target_power * valid_horizon).sum(dim=0)
        / valid_horizon.sum(dim=0).clamp_min(1.0)
    ).sqrt().detach()
    if rms_mode == "none":
        normalized_pred = pred
        normalized_truth = truth
    elif rms_mode == "batch_horizon":
        floor = float(rms_cfg.get("floor", 0.05))
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError(
                "world_prediction_loss.delta_rms_normalization.floor must be positive."
            )
        denominator = target_rms.clamp_min(floor).reshape(1, -1, 1, 1)
        normalized_pred = pred / denominator
        normalized_truth = truth / denominator
    else:
        raise ValueError(
            "world_prediction_loss.delta_rms_normalization.mode must be "
            "'none' or 'batch_horizon'."
        )
    normalized_smooth = torch.nn.functional.smooth_l1_loss(
        normalized_pred, normalized_truth, reduction="none"
    ).mean(dim=(-1, -2))

    token_cosine = 1.0 - torch.nn.functional.cosine_similarity(
        pred, truth, dim=-1
    )
    cosine_cfg = dict(cfg.get("delta_cosine") or {})
    cosine_mode = str(cosine_cfg.get("mode", "uniform"))
    if cosine_mode == "uniform":
        token_weight = torch.ones_like(token_cosine)
    elif cosine_mode == "magnitude_aware":
        scale = float(cosine_cfg.get("scale", 0.10))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(
                "world_prediction_loss.delta_cosine.scale must be positive."
            )
        token_rms = truth.square().mean(dim=-1).sqrt().detach()
        token_weight = token_rms / (token_rms + scale)
    else:
        raise ValueError(
            "world_prediction_loss.delta_cosine.mode must be 'uniform' or "
            "'magnitude_aware'."
        )
    cosine = (token_cosine * token_weight).sum(dim=-1) / token_weight.sum(
        dim=-1
    ).clamp_min(1e-6)

    cosine_loss = _masked_horizon_mean(
        cosine, horizon_mask, sample_weight, horizon_weight
    )
    smooth_loss = _masked_horizon_mean(
        normalized_smooth, horizon_mask, sample_weight, horizon_weight
    )
    diagnostics = {
        "delta_raw_smooth_l1_loss": _masked_horizon_mean(
            raw_smooth, horizon_mask, sample_weight, horizon_weight
        ),
        "delta_cosine_weight_mean": _masked_horizon_mean(
            token_weight.mean(dim=-1),
            horizon_mask,
            sample_weight,
            horizon_weight,
        ),
        "delta_target_rms_mean": _masked_horizon_mean(
            target_rms.reshape(1, -1).expand_as(horizon_mask),
            horizon_mask,
            sample_weight,
            horizon_weight,
        ),
    }
    return cosine_loss, smooth_loss, diagnostics


def router_load_balance_loss(router_probs):
    require_torch()
    reduce_dims = tuple(range(router_probs.ndim - 1))
    mean_probability = router_probs.float().mean(dim=reduce_dims)
    target = mean_probability.new_full(mean_probability.shape, 1.0 / mean_probability.numel())
    return (mean_probability - target).square().mean()


def latent_wam_training_losses(
    outputs,
    batch,
    weights: dict[str, float],
    stage: str = "joint",
    action_kind: str = "l1",
):
    """Compute the pretraining or joint objective from the resolved config."""

    torch = require_torch()
    if stage not in {"pretrain", "joint"}:
        raise ValueError("stage must be 'pretrain' or 'joint'.")
    target_actions = batch["target_actions"].to(outputs["nominal_actions"].device)
    if action_kind == "l1":
        action_fn = torch.nn.functional.l1_loss
    elif action_kind == "mse":
        action_fn = torch.nn.functional.mse_loss
    else:
        raise ValueError(f"Unsupported action loss: {action_kind}")

    losses = {"nominal_action_loss": action_fn(outputs["nominal_actions"].float(), target_actions.float())}
    if "future_latent_targets" not in outputs:
        raise RuntimeError("Training requires future latent targets from the frozen teacher or a target cache.")
    future_mask = batch["future_mask"].to(outputs["predicted_future_latents"].device)
    world_gate = outputs.get("action_distance_gate")
    losses["world_cosine_loss"], losses["world_smooth_l1_loss"] = latent_prediction_components(
        outputs["predicted_future_latents"],
        outputs["future_latent_targets"],
        future_mask,
        sample_weight=world_gate,
    )
    losses["world_loss"] = losses["world_cosine_loss"] + 0.5 * losses["world_smooth_l1_loss"]
    losses["delta_loss"] = latent_prediction_loss(
        outputs["predicted_delta_latents"],
        outputs["delta_latent_targets"],
        future_mask,
        sample_weight=world_gate,
    )

    if stage == "joint":
        losses.update(
            {
                "action_loss": action_fn(outputs["final_actions"].float(), target_actions.float()),
                "load_balance_loss": router_load_balance_loss(outputs["router_probs"]),
                "residual_loss": outputs["action_residual"].float().square().mean(),
            }
        )

    weight_keys = {
        "action_loss": "action",
        "nominal_action_loss": "nominal_action",
        "world_loss": "world",
        "delta_loss": "delta",
        "load_balance_loss": "load_balance",
        "residual_loss": "residual",
    }
    total = outputs["nominal_actions"].new_tensor(0.0, dtype=torch.float32)
    for name, weight_name in weight_keys.items():
        if name in losses:
            total = total + float(weights.get(weight_name, 0.0)) * losses[name]
    losses["total_loss"] = total
    return losses


def flow_wam_skill_losses(
    outputs,
    batch,
    weights: dict[str, float],
    schedule_state: dict | None = None,
    *,
    stage: str = "joint",
    class_weights=None,
    world_prediction_config: dict | None = None,
):
    """Loss contract for nominal pretrain, oracle warm-start, and joint routing."""

    torch = require_torch()
    if stage not in {"nominal_flow_pretrain", "expert_warmstart", "joint"}:
        raise ValueError("Unknown flow-WAM training stage.")
    schedule_state = dict(schedule_state or {})
    target_motion = batch.get("target_motion", batch["target_actions"][..., :6]).to(
        outputs["nominal_motion"].device
    )
    target_gripper = batch.get("target_gripper", batch["target_actions"][..., 6:7]).to(
        outputs["gripper_logits"].device
    )
    timestep_mask = batch.get("action_mask")
    if timestep_mask is not None:
        timestep_mask = timestep_mask.to(outputs["nominal_motion"].device).bool()

    nominal_flow = outputs.get("nominal_flow")
    if nominal_flow is None:
        raise RuntimeError("Training output is missing nominal_flow supervision tensors.")
    losses = {
        "nominal_flow_loss": conditional_flow_matching_loss(
            nominal_flow["predicted_velocity"], nominal_flow["target_velocity"], timestep_mask
        ),
        "gripper_bce_loss": gripper_binary_loss(
            outputs["gripper_logits"], target_gripper, timestep_mask
        ),
    }

    future_mask = batch.get("future_mask")
    if "future_latent_targets" in outputs and future_mask is not None:
        future_mask = future_mask.to(outputs["future_latents"].device)
        world_gate = outputs.get("action_distance_gate")
        horizon_weight = _configured_horizon_weights(
            outputs["future_latents"], future_mask, world_prediction_config
        )
        losses["world_cosine_loss"], losses["world_smooth_l1_loss"] = latent_prediction_components(
            outputs["future_latents"],
            outputs["future_latent_targets"],
            future_mask,
            sample_weight=world_gate,
            horizon_weight=horizon_weight,
        )
        losses["world_loss"] = losses["world_cosine_loss"] + 0.5 * losses["world_smooth_l1_loss"]
        delta_cosine, delta_smooth, delta_diagnostics = delta_prediction_components(
            outputs["delta_latents"],
            outputs["delta_latent_targets"],
            future_mask,
            sample_weight=world_gate,
            horizon_weight=horizon_weight,
            config=world_prediction_config,
        )
        losses["delta_cosine_loss"] = delta_cosine
        losses["delta_smooth_l1_loss"] = delta_smooth
        losses["delta_loss"] = delta_cosine + 0.5 * delta_smooth
        losses.update(delta_diagnostics)
        teacher_mask = outputs.get("teacher_forcing_mask")
        if teacher_mask is not None:
            gt_samples = teacher_mask.reshape(teacher_mask.shape[0], -1).all(dim=1)
            nominal_samples = ~gt_samples
            if bool(gt_samples.any()):
                gt_cosine, gt_smooth = latent_prediction_components(
                    outputs["future_latents"],
                    outputs["future_latent_targets"],
                    future_mask,
                    sample_weight=gt_samples,
                    horizon_weight=horizon_weight,
                )
                losses["world_loss_gt_conditioned"] = gt_cosine + 0.5 * gt_smooth
            if bool(nominal_samples.any()):
                nominal_weight = nominal_samples.to(world_gate.dtype) * world_gate
                nominal_cosine, nominal_smooth = latent_prediction_components(
                    outputs["future_latents"],
                    outputs["future_latent_targets"],
                    future_mask,
                    sample_weight=nominal_weight,
                    horizon_weight=horizon_weight,
                )
                losses["world_loss_nominal_conditioned"] = nominal_cosine + 0.5 * nominal_smooth

    labels = batch.get("expert_skill_labels")
    label_mask = batch.get("expert_skill_mask")
    if labels is not None:
        labels = labels.to(outputs["router_logits"].device)
    if label_mask is not None:
        label_mask = label_mask.to(outputs["router_logits"].device).bool()
    if stage != "nominal_flow_pretrain":
        if labels is None or label_mask is None:
            raise RuntimeError("Expert warm-start/joint training requires per-timestep skill labels.")
        losses["route_ce_loss"] = masked_route_cross_entropy(
            outputs["router_logits"], labels, label_mask, class_weights=class_weights
        )
        expert_flow = outputs.get("expert_flow")
        if expert_flow is None:
            raise RuntimeError("Expert warm-start/joint output is missing expert_flow tensors.")
        motor_mask = label_mask & labels.ge(0) & labels.lt(6)
        losses["expert_flow_loss"] = conditional_flow_matching_loss(
            expert_flow["predicted_velocity"], expert_flow["target_velocity"], motor_mask
        )
        losses["residual_loss"] = residual_regularization(outputs["residual_motion"], motor_mask)
        losses["endpoint_loss"] = (
            (outputs["motion_actions"].float() - target_motion.float()).abs().mean(dim=-1)
            * motor_mask.to(dtype=torch.float32)
        ).sum() / motor_mask.sum().clamp_min(1)
        if schedule_state.get("enable_load_balance", outputs.get("route_source") != "oracle"):
            motor_probabilities = outputs["router_probs"][..., :6]
            motor_probabilities = motor_probabilities / motor_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            losses["load_balance_loss"] = (
                router_load_balance_loss(motor_probabilities[motor_mask])
                if bool(motor_mask.any())
                else motor_probabilities.float().sum() * 0.0
            )

    predicted_gripper = outputs["gripper_logits"].ge(0)
    losses["gripper_accuracy"] = predicted_gripper.eq(target_gripper.bool()).float().mean()
    losses["nominal_motion_target_l1"] = (
        outputs["nominal_motion"].float() - target_motion.float()
    ).abs().mean()
    losses["motion_endpoint_l1"] = (
        outputs["motion_actions"].float() - target_motion.float()
    ).abs().mean()

    weight_keys = {
        "nominal_flow_loss": "flow_nominal",
        "expert_flow_loss": "flow_expert",
        "gripper_bce_loss": "gripper_bce",
        "route_ce_loss": "route",
        "world_loss": "world",
        "delta_loss": "delta",
        "load_balance_loss": "load_balance",
        "residual_loss": "residual",
        "endpoint_loss": "endpoint",
    }
    total = outputs["nominal_motion"].new_tensor(0.0, dtype=torch.float32)
    for name, weight_key in weight_keys.items():
        if name in losses:
            total = total + float(weights.get(weight_key, 0.0)) * losses[name]
    losses["total_loss"] = total
    return losses
