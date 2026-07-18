"""Schedules for action conditioning and sparse routing."""

from __future__ import annotations


def progress_ratio(step: int, max_steps: int) -> float:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")
    return min(1.0, max(0.0, float(step) / float(max_steps)))


def action_condition_probability(
    step: int,
    max_steps: int,
    nominal_start_ratio: float = 0.30,
    nominal_end_ratio: float = 0.70,
    final_nominal_probability: float = 0.80,
) -> float:
    """Return the probability of conditioning the WAM on nominal actions."""

    ratio = progress_ratio(step, max_steps)
    if ratio <= nominal_start_ratio:
        return 0.0
    if ratio >= nominal_end_ratio:
        return float(final_nominal_probability)
    width = max(nominal_end_ratio - nominal_start_ratio, 1e-8)
    return float(final_nominal_probability * (ratio - nominal_start_ratio) / width)


def teacher_forcing_probability(step: int, max_steps: int, **kwargs) -> float:
    """Return GT-condition probability (the complement of nominal use)."""

    return 1.0 - action_condition_probability(step, max_steps, **kwargs)


def action_distance_gate(nominal, target, beta: float = 2.0):
    """Detached confidence for pairing nominal actions with demonstration futures."""

    if beta <= 0:
        return nominal.new_ones((nominal.shape[0],))
    distance = (nominal.detach().float() - target.detach().float()).abs().mean(dim=(1, 2))
    return (-float(beta) * distance).exp().clamp_min(0.05).to(nominal.dtype)


def temporal_router_schedule(
    step: int,
    max_steps: int,
    predicted_start_ratio: float = 0.20,
    predicted_end_ratio: float = 0.70,
    start_temperature: float = 1.0,
    end_temperature: float = 0.1,
) -> dict[str, float]:
    """Oracle-to-ST-Gumbel schedule for per-position Top-1 routing."""

    ratio = progress_ratio(step, max_steps)
    if ratio <= predicted_start_ratio:
        predicted_probability = 0.0
        temperature = start_temperature
    elif ratio >= predicted_end_ratio:
        predicted_probability = 1.0
        temperature = end_temperature
    else:
        width = max(predicted_end_ratio - predicted_start_ratio, 1e-8)
        fraction = (ratio - predicted_start_ratio) / width
        predicted_probability = fraction
        temperature = start_temperature + fraction * (end_temperature - start_temperature)
    return {
        "oracle_route_probability": float(1.0 - predicted_probability),
        "st_gumbel_probability": float(predicted_probability),
        "gumbel_temperature": float(temperature),
    }


def router_schedule(
    step: int,
    max_steps: int,
    soft_warmup_ratio: float = 0.10,
    anneal_end_ratio: float = 0.30,
    start_temperature: float = 1.0,
    end_temperature: float = 0.2,
) -> dict[str, float | bool]:
    """Use dense soft routing, sharpen it, then switch to hard Top-k."""

    ratio = progress_ratio(step, max_steps)
    if ratio <= soft_warmup_ratio:
        temperature = start_temperature
    elif ratio < anneal_end_ratio:
        width = max(anneal_end_ratio - soft_warmup_ratio, 1e-8)
        fraction = (ratio - soft_warmup_ratio) / width
        temperature = start_temperature + fraction * (end_temperature - start_temperature)
    else:
        temperature = end_temperature
    return {"hard_topk": ratio >= anneal_end_ratio, "temperature": float(temperature)}
