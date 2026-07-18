"""Complete motion-flow, latent-WAM, temporal-skill policy."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.models.action_flow import rectified_flow_path
from mowe_wam.training.schedules import action_distance_gate
from mowe_wam.utils.optional import require_torch


_CONTEXT_ALLOWED_FIELDS = {
    "pixel_values_primary",
    "pixel_values_wrist",
    "history_pixel_values_primary",
    "history_pixel_values_wrist",
    "long_history_pixel_values_primary",
    "long_history_pixel_values_wrist",
    "language",
    "proprio",
    "proprio_mask",
    "current_visual_views",
    "history_visual_views",
    "long_history_visual_views",
    "precomputed_language",
}


def first_skill_segment_steps(route_indices, max_steps: int = 8):
    """Return the first same-skill segment length for offline diagnostics.

    Deployment no longer stops at every predicted skill boundary.  This helper
    is retained to measure boundary locations and oracle-label crossings.
    """

    torch_mod = require_torch()
    if route_indices.ndim != 2 or route_indices.shape[1] < 1:
        raise ValueError("route_indices must have shape [B, chunk].")
    cap = min(int(max_steps), route_indices.shape[1])
    first = route_indices[:, :1]
    same = route_indices[:, :cap].eq(first)
    # cumprod turns all positions at/after the first boundary to zero.
    prefix = same.to(torch_mod.long).cumprod(dim=1).sum(dim=1)
    return prefix.clamp(min=1, max=cap)


def execution_steps(route_indices, max_steps: int = 8):
    """Backward-compatible alias for :func:`first_skill_segment_steps`."""

    return first_skill_segment_steps(route_indices, max_steps)


def risk_gated_execution(
    route_indices,
    route_probabilities,
    motion_actions,
    residual_motion,
    *,
    default_steps: int = 8,
    caution_steps: int = 4,
    normalized_entropy_caution: float = 0.55,
    normalized_entropy_high: float = 0.75,
    top2_margin_caution: float = 0.20,
    top2_margin_high: float = 0.10,
    motion_jump_l2_caution: float = 0.60,
    motion_jump_l2_high: float = 0.90,
    residual_l2_caution: float = 0.35,
    residual_l2_high: float = 0.45,
):
    """Choose a synchronous committed prefix without hard-capping every boundary.

    A confident, smooth per-token expert transition is allowed inside the
    default prefix.  A caution boundary shortens the prefix to four actions;
    a high-risk boundary stops before the first token assigned to the new
    skill.  The returned tensors are deployment diagnostics, not training
    targets, and are intentionally detached from the gradient graph.
    """

    torch_mod = require_torch()
    if route_indices.ndim != 2:
        raise ValueError("route_indices must have shape [B, chunk].")
    batch_size, chunk_size = route_indices.shape
    if route_probabilities.shape[:2] != (batch_size, chunk_size):
        raise ValueError("route_probabilities must align with route_indices.")
    if motion_actions.shape != (batch_size, chunk_size, 6):
        raise ValueError("motion_actions must have shape [B, chunk, 6].")
    if residual_motion.shape != motion_actions.shape:
        raise ValueError("residual_motion must match motion_actions.")
    default_steps = int(default_steps)
    caution_steps = int(caution_steps)
    if not 1 <= caution_steps <= default_steps <= chunk_size:
        raise ValueError("Execution steps must satisfy 1 <= caution <= default <= chunk.")
    if not 0.0 <= normalized_entropy_caution < normalized_entropy_high <= 1.0:
        raise ValueError("Normalized entropy thresholds must satisfy 0 <= caution < high <= 1.")
    if not 0.0 <= top2_margin_high < top2_margin_caution <= 1.0:
        raise ValueError("Top-2 margin thresholds must satisfy 0 <= high < caution <= 1.")
    if not 0.0 <= motion_jump_l2_caution < motion_jump_l2_high:
        raise ValueError("Motion-jump thresholds must satisfy 0 <= caution < high.")
    if not 0.0 <= residual_l2_caution < residual_l2_high:
        raise ValueError("Residual thresholds must satisfy 0 <= caution < high.")

    with torch_mod.no_grad():
        probabilities = route_probabilities.detach().float()
        probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        num_routes = probabilities.shape[-1]
        entropy = -(probabilities * probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / max(float(torch_mod.log(torch_mod.tensor(float(num_routes)))), 1e-8)
        top2 = probabilities.topk(min(2, num_routes), dim=-1).values
        margin = top2[..., 0] - (top2[..., 1] if num_routes > 1 else 0.0)
        motion = motion_actions.detach().float()
        residual = residual_motion.detach().float()

        steps = route_indices.new_full((batch_size,), default_steps)
        reason_code = route_indices.new_zeros((batch_size,))  # 0=default, 1=caution, 2=high
        boundary_position = route_indices.new_full((batch_size,), -1)
        boundary_entropy = motion.new_zeros((batch_size,))
        boundary_margin = motion.new_ones((batch_size,))
        boundary_motion_jump = motion.new_zeros((batch_size,))
        boundary_residual_l2 = motion.new_zeros((batch_size,))
        crosses_boundary = torch_mod.zeros((batch_size,), dtype=torch_mod.bool, device=route_indices.device)

        for row in range(batch_size):
            candidates = []
            for position in range(1, default_steps):
                if int(route_indices[row, position]) == int(route_indices[row, position - 1]):
                    continue
                local_entropy = float(entropy[row, position - 1 : position + 1].max())
                local_margin = float(margin[row, position - 1 : position + 1].min())
                motion_jump = float((motion[row, position] - motion[row, position - 1]).norm())
                residual_l2 = float(
                    residual[row, position - 1 : position + 1].norm(dim=-1).max()
                )
                high = (
                    local_entropy >= normalized_entropy_high
                    or local_margin <= top2_margin_high
                    or motion_jump >= motion_jump_l2_high
                    or residual_l2 >= residual_l2_high
                )
                caution = high or (
                    local_entropy >= normalized_entropy_caution
                    or local_margin <= top2_margin_caution
                    or motion_jump >= motion_jump_l2_caution
                    or residual_l2 >= residual_l2_caution
                )
                candidates.append(
                    (position, 2 if high else 1 if caution else 0, local_entropy, local_margin, motion_jump, residual_l2)
                )

            if not candidates:
                continue
            high_candidates = [item for item in candidates if item[1] == 2]
            caution_candidates = [item for item in candidates if item[1] == 1]
            if high_candidates:
                selected = high_candidates[0]
                steps[row] = max(1, selected[0])
                reason_code[row] = 2
            elif caution_candidates:
                selected = caution_candidates[0]
                steps[row] = caution_steps
                reason_code[row] = 1
            else:
                selected = candidates[0]
            boundary_position[row] = selected[0]
            boundary_entropy[row] = selected[2]
            boundary_margin[row] = selected[3]
            boundary_motion_jump[row] = selected[4]
            boundary_residual_l2[row] = selected[5]
            crosses_boundary[row] = any(item[0] < int(steps[row]) for item in candidates)

    return {
        "execution_steps": steps,
        "execution_reason_code": reason_code,
        "execution_boundary_position": boundary_position,
        "execution_boundary_entropy": boundary_entropy,
        "execution_boundary_margin": boundary_margin,
        "execution_motion_jump_l2": boundary_motion_jump,
        "execution_residual_l2": boundary_residual_l2,
        "execution_crosses_predicted_boundary": crosses_boundary,
    }


class FlowWAMSkillPolicy(nn.Module if nn is not None else object):
    """Compose nominal motion flow, WAM, temporal router, and residual flow."""

    def __init__(
        self,
        backbone,
        memory_encoder,
        nominal_action_head,
        world_model,
        router,
        residual_experts,
        view_fusion,
        visual_teacher=None,
        context_dim: int = 4096,
        memory_dim: int = 512,
        world_dim: int = 512,
        expert_condition_dim: int = 512,
        flow_steps: int = 4,
        execution_config: dict | None = None,
        action_distance_beta: float = 2.0,
        max_residual_l2: float = 0.5,
        ablation: dict | None = None,
    ) -> None:
        require_torch()
        super().__init__()
        self.backbone = backbone
        self.memory_encoder = memory_encoder
        self.nominal_action_head = nominal_action_head
        self.world_model = world_model
        self.router = router
        self.residual_experts = residual_experts
        self.view_fusion = view_fusion
        self.visual_teacher = visual_teacher
        self.flow_steps = int(flow_steps)
        execution = dict(execution_config or {})
        risk = dict(execution.get("risk", {}))
        self.execution_default_steps = int(execution.get("default_steps", 8))
        self.execution_caution_steps = int(execution.get("caution_steps", 4))
        self.execution_risk = {
            "normalized_entropy_caution": float(risk.get("normalized_entropy_caution", 0.55)),
            "normalized_entropy_high": float(risk.get("normalized_entropy_high", 0.75)),
            "top2_margin_caution": float(risk.get("top2_margin_caution", 0.20)),
            "top2_margin_high": float(risk.get("top2_margin_high", 0.10)),
            "motion_jump_l2_caution": float(risk.get("motion_jump_l2_caution", 0.60)),
            "motion_jump_l2_high": float(risk.get("motion_jump_l2_high", 0.90)),
            "residual_l2_caution": float(risk.get("residual_l2_caution", 0.35)),
            "residual_l2_high": float(risk.get("residual_l2_high", 0.45)),
        }
        self.action_distance_beta = float(action_distance_beta)
        self.max_residual_l2 = float(max_residual_l2)
        if not 0.0 < self.max_residual_l2 <= 1.0:
            raise ValueError("max_residual_l2 must be within (0,1].")
        self.ablation = dict(ablation or {})
        self.expert_context = nn.Sequential(
            nn.Linear(context_dim + memory_dim + world_dim, expert_condition_dim),
            nn.LayerNorm(expert_condition_dim),
            nn.SiLU(),
        )

    def _project_residual(self, residual):
        norms = residual.float().norm(dim=-1, keepdim=True)
        scale = (self.max_residual_l2 / norms.clamp_min(1e-8)).clamp(max=1.0)
        bounded = residual * scale.to(dtype=residual.dtype)
        clipped = norms.squeeze(-1).gt(self.max_residual_l2)
        return bounded, clipped

    def _execution_policy(self, routing, motion_actions, residual_motion):
        return risk_gated_execution(
            routing["route_indices"],
            routing["router_probs"],
            motion_actions,
            residual_motion,
            default_steps=self.execution_default_steps,
            caution_steps=self.execution_caution_steps,
            **self.execution_risk,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        keep_eval = getattr(self.backbone, "keep_frozen_backbone_eval", None)
        if keep_eval is not None:
            keep_eval()
        if self.visual_teacher is not None:
            self.visual_teacher.eval()
        return self

    def _teacher_targets(self, batch, device):
        if "future_latent_targets" in batch and "current_latent_target" in batch:
            return (
                batch["current_latent_target"].to(device=device),
                batch["future_latent_targets"].to(device=device),
            )
        if self.visual_teacher is None:
            return None, None
        current_raw = batch["current_raw_pixel_values"]
        future_raw = batch["future_raw_pixel_values"]
        batch_size, horizons = future_raw.shape[:2]
        current = self.visual_teacher.encode(current_raw)
        future = self.visual_teacher.encode(future_raw.flatten(0, 1)).reshape(
            batch_size,
            horizons,
            self.visual_teacher.spatial_tokens,
            self.visual_teacher.target_dim,
        )
        return current.to(device=device), future.to(device=device)

    @staticmethod
    def _target_parts(batch, reference):
        if "target_motion" in batch and "target_gripper" in batch:
            motion = batch["target_motion"]
            gripper = batch["target_gripper"]
        elif "target_actions" in batch:
            motion = batch["target_actions"][..., :6]
            gripper = batch["target_actions"][..., 6:7]
        else:
            return None, None
        motion = motion.to(device=reference.device, dtype=reference.dtype)
        gripper = gripper.to(device=reference.device, dtype=reference.dtype)
        if not torch.all((gripper == 0) | (gripper == 1)):
            raise ValueError("target_gripper must use canonical absolute binary 0/1 targets.")
        return motion, gripper

    @staticmethod
    def _select_action_condition(nominal_actions, target_motion, target_gripper, mode, probability):
        nominal = nominal_actions.detach()
        if mode == "nominal":
            mask = torch.zeros((nominal.shape[0], 1, 1), dtype=torch.bool, device=nominal.device)
            return nominal, mask
        if target_motion is None or target_gripper is None:
            raise ValueError(f"action condition mode {mode!r} requires action targets.")
        target = torch.cat([target_motion, target_gripper], dim=-1)
        if mode == "ground_truth":
            mask = torch.ones((nominal.shape[0], 1, 1), dtype=torch.bool, device=nominal.device)
            return target, mask
        if mode == "scheduled":
            probability = min(1.0, max(0.0, float(probability)))
            mask = torch.rand((nominal.shape[0], 1, 1), device=nominal.device) < probability
            return torch.where(mask, target, nominal), mask
        raise ValueError("action_condition_mode must be nominal, ground_truth, or scheduled.")

    def forward(
        self,
        batch: dict,
        *,
        action_condition_mode: str = "scheduled",
        teacher_forcing_probability: float = 0.0,
        route_mode: str = "predicted",
        gumbel_temperature: float = 1.0,
        flow_seed: int | None = None,
        flow_steps: int | None = None,
        compute_teacher_targets: bool = True,
        compute_residual: bool = True,
        compute_route_diagnostics: bool = False,
    ):
        steps = self.flow_steps if flow_steps is None else int(flow_steps)
        # Do not merely rely on the backbone ignoring supervision keys: remove
        # every action/teacher/skill target before the call boundary.
        context_batch = {
            key: value
            for key, value in batch.items()
            if key in _CONTEXT_ALLOWED_FIELDS or key.startswith("synthetic_")
        }
        context = self.backbone.extract_context_features(context_batch)
        language = context["language"]
        current_view = self.view_fusion(context["current_visual_views"], language)
        history_view = self.view_fusion(context["history_visual_views"], language)
        long_view = self.view_fusion(context["long_history_visual_views"], language)
        current = current_view["fused"]
        language = language.to(device=current.device, dtype=current.dtype)
        memory = self.memory_encoder(
            history_visual=history_view["fused"],
            current_visual=current,
            history_actions=batch["history_actions"].to(current.device),
            history_mask=batch["history_mask"].to(current.device),
            long_history_visual=long_view["fused"],
            long_history_actions=batch["long_history_actions"].to(current.device),
            long_history_mask=batch["long_history_mask"].to(current.device),
            language=language,
        )
        nominal = self.nominal_action_head.sample(
            current,
            memory["memory_context"],
            seed=flow_seed,
            num_steps=steps,
        )
        target_motion, target_gripper = self._target_parts(batch, nominal["nominal_motion"])

        nominal_flow = None
        if target_motion is not None:
            nominal_flow = rectified_flow_path(target_motion)
            nominal_flow["predicted_velocity"] = self.nominal_action_head.motion_velocity(
                nominal_flow["noisy_motion"],
                nominal_flow["flow_time"],
                current,
                memory["memory_context"],
                condition=nominal["flow_condition"],
            )

        action_condition, teacher_mask = self._select_action_condition(
            nominal["nominal_actions"],
            target_motion,
            target_gripper,
            action_condition_mode,
            teacher_forcing_probability,
        )
        world = self.world_model(
            current_context=current,
            short_memory=memory["short_memory_tokens"],
            long_memory=memory["long_memory_tokens"],
            action_condition=action_condition,
            short_memory_mask=memory["short_memory_mask"],
            long_memory_mask=memory["long_memory_mask"],
            horizon_mask=batch.get("future_mask"),
        )
        router_world_belief = world["world_belief"]
        router_future = world["future_latents"]
        router_delta = world["delta_latents"]
        router_tokens = world["route_world_tokens"]
        router_memory = memory["memory_context"]
        unshuffled_router_inputs = (
            router_world_belief,
            router_future,
            router_delta,
            router_tokens,
            router_memory,
        )
        if self.ablation.get("shuffle_future_before_router", False):
            if router_future.shape[0] > 1:
                # Cross-sample permutation is the clean control when a batch has
                # multiple tasks/windows.
                router_future = router_future.roll(1, dims=0)
                router_delta = router_delta.roll(1, dims=0)
                router_tokens = router_tokens.roll(1, dims=0)
            else:
                # The default real batch size is one.  A batch-axis roll would
                # silently be an identity ablation, so permute latent channels
                # and the stepwise h-token schedule instead.
                router_future = router_future.roll(1, dims=-1)
                router_delta = router_delta.roll(1, dims=-1)
                router_tokens = router_tokens.roll(1, dims=1)
        if self.ablation.get("history_only_router", False):
            router_future = torch.zeros_like(router_future)
            router_delta = torch.zeros_like(router_delta)
            router_tokens = torch.zeros_like(router_tokens)
        if self.ablation.get("behavior_prior_router", False):
            router_world_belief = torch.zeros_like(router_world_belief)
            router_future = torch.zeros_like(router_future)
            router_delta = torch.zeros_like(router_delta)
            router_tokens = torch.zeros_like(router_tokens)
            router_memory = torch.zeros_like(router_memory)
        labels = batch.get("expert_skill_labels")
        label_mask = batch.get("expert_skill_mask")
        if labels is not None:
            labels = labels.to(device=current.device)
        if label_mask is not None:
            label_mask = label_mask.to(device=current.device)
        routing = self.router(
            world_belief=router_world_belief,
            future_latents=router_future,
            delta_latents=router_delta,
            route_world_tokens=router_tokens,
            memory_context=router_memory,
            nominal_action_tokens=nominal["nominal_actions"],
            uncertainty=world.get("log_variance"),
            route_mode=route_mode,
            oracle_labels=labels,
            oracle_mask=label_mask,
            gumbel_temperature=gumbel_temperature,
        )
        if self.ablation.get("shuffle_future_before_router", False):
            with torch.no_grad():
                reference_routing = self.router(
                    world_belief=unshuffled_router_inputs[0],
                    future_latents=unshuffled_router_inputs[1],
                    delta_latents=unshuffled_router_inputs[2],
                    route_world_tokens=unshuffled_router_inputs[3],
                    memory_context=unshuffled_router_inputs[4],
                    nominal_action_tokens=nominal["nominal_actions"],
                    uncertainty=world.get("log_variance"),
                    route_mode="predicted",
                )
            routing["future_shuffle_router_change_rate"] = (
                routing["router_logits"].argmax(dim=-1)
                .ne(reference_routing["route_indices"])
                .float()
                .mean()
            )
            routing["future_shuffle_router_logit_l1"] = (
                routing["router_logits"].float() - reference_routing["router_logits"].float()
            ).abs().mean()
        expert_condition = self.expert_context(
            torch.cat([current, memory["memory_context"], world["world_belief"]], dim=-1)
        )

        expert_flow = None
        residual_target_clipped = nominal["nominal_motion"].new_zeros(
            nominal["nominal_motion"].shape[:2], dtype=torch.bool
        )
        if target_motion is not None and compute_residual:
            residual_target_raw = target_motion - nominal["nominal_motion"].detach()
            residual_target, residual_target_clipped = self._project_residual(
                residual_target_raw
            )
            expert_flow = rectified_flow_path(residual_target)
            motor_gate = routing["route_gates"][..., :6].sum(dim=-1, keepdim=True)
            expert_flow["noisy_motion"] = motor_gate * expert_flow["noisy_motion"]
            expert_flow["predicted_velocity"] = self.residual_experts.velocity(
                expert_condition,
                nominal["nominal_motion"].detach(),
                expert_flow["noisy_motion"],
                expert_flow["flow_time"],
                routing["route_gates"],
            )
            expert_flow["residual_target"] = residual_target
            expert_flow["residual_target_raw"] = residual_target_raw

        if compute_residual:
            residual_motion_raw = self.residual_experts.sample(
                expert_condition,
                nominal["nominal_motion"].detach(),
                routing["route_gates"],
                seed=None if flow_seed is None else int(flow_seed) + 1,
                num_steps=steps,
            )
            residual_motion, residual_clipped = self._project_residual(residual_motion_raw)
        else:
            residual_motion_raw = torch.zeros_like(nominal["nominal_motion"])
            residual_motion = torch.zeros_like(nominal["nominal_motion"])
            residual_clipped = nominal["nominal_motion"].new_zeros(
                nominal["nominal_motion"].shape[:2], dtype=torch.bool
            )
        motion_actions = (nominal["nominal_motion"] + residual_motion).clamp(-1.0, 1.0)
        gripper_actions = (nominal["gripper_probability"] >= 0.5).to(motion_actions.dtype)
        actions = torch.cat([motion_actions, gripper_actions], dim=-1)
        nominal_distance_gate = (
            action_distance_gate(
                nominal["nominal_motion"], target_motion, beta=self.action_distance_beta
            )
            if target_motion is not None
            else motion_actions.new_ones((motion_actions.shape[0],))
        )
        # Ground-truth-conditioned samples are exactly aligned to the
        # demonstration future and should retain unit weight.  Only nominal-
        # conditioned samples need the detached action-distance confidence.
        world_gate = torch.where(
            teacher_mask.reshape(teacher_mask.shape[0], -1).all(dim=1),
            torch.ones_like(nominal_distance_gate),
            nominal_distance_gate,
        )

        route_mode_diagnostics = {}
        if compute_route_diagnostics and compute_residual and target_motion is not None:
            with torch.no_grad():
                diagnostic_gates = {
                    "hard_predicted": torch.nn.functional.one_hot(
                        routing["router_logits"].argmax(dim=-1),
                        routing["router_logits"].shape[-1],
                    ).to(routing["router_logits"].dtype)
                }
                if labels is not None:
                    valid_labels = labels.ge(0) & labels.lt(routing["router_logits"].shape[-1])
                    if label_mask is not None:
                        valid_labels = valid_labels & label_mask.bool()
                    safe_labels = torch.where(valid_labels, labels, labels.new_full(labels.shape, 6))
                    diagnostic_gates["oracle"] = torch.nn.functional.one_hot(
                        safe_labels.long(), routing["router_logits"].shape[-1]
                    ).to(routing["router_logits"].dtype)
                generator = torch.Generator(device=routing["router_logits"].device)
                generator.manual_seed(int(flow_seed or 0) + 90_001)
                uniform = torch.rand(
                    routing["router_logits"].shape,
                    device=routing["router_logits"].device,
                    dtype=routing["router_logits"].dtype,
                    generator=generator,
                ).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(uniform))
                st_indices = (
                    routing["router_logits"] / max(float(gumbel_temperature), 1e-4) + gumbel
                ).argmax(dim=-1)
                diagnostic_gates["st_gumbel"] = torch.nn.functional.one_hot(
                    st_indices, routing["router_logits"].shape[-1]
                ).to(routing["router_logits"].dtype)
                valid_actions = batch.get("action_mask")
                if valid_actions is None:
                    valid_actions = torch.ones(target_motion.shape[:2], dtype=torch.bool, device=target_motion.device)
                else:
                    valid_actions = valid_actions.to(target_motion.device).bool()
                for diagnostic_index, (name, gates) in enumerate(diagnostic_gates.items()):
                    diagnostic_residual = self.residual_experts.sample(
                        expert_condition,
                        nominal["nominal_motion"].detach(),
                        gates,
                        seed=int(flow_seed or 0) + 10_000 + diagnostic_index,
                        num_steps=steps,
                    )
                    diagnostic_residual, _ = self._project_residual(diagnostic_residual)
                    diagnostic_motion = (
                        nominal["nominal_motion"].detach() + diagnostic_residual
                    ).clamp(-1.0, 1.0)
                    diagnostic_execution = risk_gated_execution(
                        gates.argmax(dim=-1),
                        gates,
                        diagnostic_motion,
                        diagnostic_residual,
                        default_steps=self.execution_default_steps,
                        caution_steps=self.execution_caution_steps,
                        **self.execution_risk,
                    )
                    endpoint = (
                        diagnostic_motion.float() - target_motion.float()
                    ).abs().mean(dim=-1)
                    record = {
                        "motion_endpoint_l1": (
                            endpoint * valid_actions.float()
                        ).sum()
                        / valid_actions.sum().clamp_min(1),
                        "execution_steps_mean": diagnostic_execution[
                            "execution_steps"
                        ].float().mean(),
                    }
                    if labels is not None and label_mask is not None:
                        valid_route = label_mask.bool() & labels.ge(0)
                        record["route_accuracy"] = (
                            gates.argmax(dim=-1).eq(labels) & valid_route
                        ).float().sum() / valid_route.sum().clamp_min(1)
                    route_mode_diagnostics[name] = record

        execution = self._execution_policy(routing, motion_actions, residual_motion)
        output = {
            **nominal,
            **world,
            **routing,
            "residual_motion": residual_motion,
            "residual_motion_raw": residual_motion_raw,
            "residual_clip_fraction": residual_clipped.float().mean(),
            "residual_target_clip_fraction": residual_target_clipped.float().mean(),
            "max_residual_l2": residual_motion.new_tensor(self.max_residual_l2),
            "motion_actions": motion_actions,
            "gripper_actions": gripper_actions,
            "actions": actions,
            "final_actions": actions,
            "memory_context": memory["memory_context"],
            "action_condition": action_condition,
            "teacher_forcing_mask": teacher_mask,
            "action_distance_gate": world_gate,
            "nominal_action_distance_gate": nominal_distance_gate,
            **execution,
            "execution_default_steps": motion_actions.new_tensor(
                self.execution_default_steps, dtype=torch.long
            ),
            "null_motion_zero_violation_count": (
                (routing["route_indices"].eq(6).unsqueeze(-1) & residual_motion.ne(0)).sum()
            ),
            "context_input_keys": sorted(context_batch),
            "route_mode_diagnostics": route_mode_diagnostics,
            "current_view_weights": current_view["weights"],
            "history_view_weights": history_view["weights"],
            "long_history_view_weights": long_view["weights"],
            "current_view_entropy": current_view["entropy"],
            "history_view_entropy": history_view["entropy"],
            "long_history_view_entropy": long_view["entropy"],
            "view_order": self.view_fusion.view_order,
        }
        if nominal_flow is not None:
            output["nominal_flow"] = nominal_flow
        if expert_flow is not None:
            output["expert_flow"] = expert_flow
        if compute_teacher_targets:
            current_target, future_target = self._teacher_targets(batch, actions.device)
            if current_target is not None and future_target is not None:
                output["current_latent_target"] = current_target
                output["future_latent_targets"] = future_target
                output["delta_latent_targets"] = future_target - current_target.unsqueeze(1)
        return output

    def predict_actions(self, batch: dict, *, flow_seed: int | None = None):
        forbidden = {"expert_skill_labels", "expert_skill_mask", "expert_label_source"} & set(batch)
        if forbidden:
            raise ValueError(f"Deployment batch must not contain training labels: {sorted(forbidden)}")
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(
                batch,
                action_condition_mode="nominal",
                route_mode="predicted",
                flow_seed=flow_seed,
                compute_teacher_targets=False,
            )
        self.train(was_training)
        prefixes = [
            output["actions"][index, : int(length)]
            for index, length in enumerate(output["execution_steps"].tolist())
        ]
        return prefixes, output
