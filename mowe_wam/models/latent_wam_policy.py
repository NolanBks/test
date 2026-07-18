"""Complete nominal-action-conditioned latent WAM residual-MoE policy."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch
from mowe_wam.training.schedules import action_distance_gate


class LatentWAMPolicy(nn.Module if nn is not None else object):
    """Compose frozen context/teacher modules with trainable WAM and MoE heads."""

    def __init__(
        self,
        backbone,
        memory_encoder,
        nominal_action_head,
        world_model,
        router,
        residual_experts,
        visual_teacher=None,
        context_dim: int = 4096,
        memory_dim: int = 512,
        world_dim: int = 512,
        expert_hidden_dim: int = 512,
        residual_gate_threshold: float = 0.05,
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
        self.visual_teacher = visual_teacher
        self.residual_gate_threshold = float(residual_gate_threshold)
        self.ablation = dict(ablation or {})
        self.expert_context = nn.Sequential(
            nn.Linear(context_dim + memory_dim + world_dim, expert_hidden_dim),
            nn.LayerNorm(expert_hidden_dim),
            nn.GELU(),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        keep_eval = getattr(self.backbone, "keep_frozen_backbone_eval", None)
        if keep_eval is not None:
            keep_eval()
        if self.visual_teacher is not None:
            self.visual_teacher.eval()
        return self

    def trainable_parameters(self):
        return (parameter for parameter in self.parameters() if parameter.requires_grad)

    def _select_action_condition(self, nominal_actions, target_actions, mode: str, teacher_forcing_probability: float):
        nominal = nominal_actions.detach()
        if mode == "nominal":
            mask = torch.zeros((nominal.shape[0], 1, 1), dtype=torch.bool, device=nominal.device)
            return nominal, mask
        if target_actions is None:
            raise ValueError(f"action condition mode {mode!r} requires target_actions.")
        target = target_actions.to(device=nominal.device, dtype=nominal.dtype)
        if mode == "ground_truth":
            mask = torch.ones((nominal.shape[0], 1, 1), dtype=torch.bool, device=nominal.device)
            return target, mask
        if mode == "scheduled":
            probability = min(1.0, max(0.0, float(teacher_forcing_probability)))
            mask = torch.rand((nominal.shape[0], 1, 1), device=nominal.device) < probability
            return torch.where(mask, target, nominal), mask
        raise ValueError(f"Unknown action condition mode: {mode}")

    def _teacher_targets(self, batch, device):
        if "future_latent_targets" in batch and "current_latent_target" in batch:
            current = batch["current_latent_target"].to(device=device)
            future = batch["future_latent_targets"].to(device=device)
            return current, future
        if self.visual_teacher is None:
            return None, None
        current_raw = batch["current_raw_pixel_values"]
        future_raw = batch["future_raw_pixel_values"]
        batch_size, horizons = future_raw.shape[:2]
        current = self.visual_teacher.encode(current_raw)
        future = self.visual_teacher.encode(future_raw.flatten(0, 1)).reshape(
            batch_size, horizons, self.visual_teacher.spatial_tokens, self.visual_teacher.target_dim
        )
        return current.to(device=device), future.to(device=device)

    def forward(
        self,
        batch,
        action_condition_mode: str = "nominal",
        teacher_forcing_probability: float = 0.0,
        router_hard_topk: bool = True,
        router_temperature: float = 1.0,
        compute_teacher_targets: bool | None = None,
    ):
        context = self.backbone.extract_context_features(batch)
        memory = self.memory_encoder(
            history_visual=context["history_visual"],
            current_visual=context["current_visual"],
            history_actions=batch["history_actions"].to(context["current_visual"].device),
            history_mask=batch["history_mask"].to(context["current_visual"].device),
            long_history_visual=context["long_history_visual"],
            long_history_actions=batch["long_history_actions"].to(context["current_visual"].device),
            long_history_mask=batch["long_history_mask"].to(context["current_visual"].device),
            language=context["language"],
        )
        disable_short = bool(self.ablation.get("disable_short_memory", False))
        disable_long = bool(self.ablation.get("disable_long_memory", False))
        if disable_short:
            memory["short_memory_tokens"] = torch.zeros_like(memory["short_memory_tokens"])
            memory["short_memory_mask"] = torch.zeros_like(memory["short_memory_mask"])
        if disable_long:
            memory["long_memory_tokens"] = torch.zeros_like(memory["long_memory_tokens"])
            memory["long_memory_mask"] = torch.zeros_like(memory["long_memory_mask"])
        if disable_short or disable_long:
            short_summary = (
                torch.zeros_like(memory["short_memory_summary"])
                if disable_short
                else memory["short_memory_summary"]
            )
            long_summary = (
                torch.zeros_like(memory["long_memory_summary"])
                if disable_long
                else memory["long_memory_summary"]
            )
            memory["memory_context"] = self.memory_encoder.output_norm(short_summary + long_summary)
        nominal_actions = self.nominal_action_head(context["current_visual"], memory["memory_context"])
        target_actions = batch.get("target_actions")
        action_condition, teacher_forcing_mask = self._select_action_condition(
            nominal_actions,
            target_actions,
            mode=action_condition_mode,
            teacher_forcing_probability=teacher_forcing_probability,
        )
        if self.ablation.get("action_condition") == "zeros":
            action_condition = torch.zeros_like(action_condition)
        world = self.world_model(
            current_context=context["current_visual"],
            short_memory_tokens=memory["short_memory_tokens"],
            short_memory_mask=memory["short_memory_mask"],
            long_memory_tokens=memory["long_memory_tokens"],
            long_memory_mask=memory["long_memory_mask"],
            action_condition=action_condition,
        )
        router_future = world["predicted_future_latents"]
        router_delta = world["predicted_delta_latents"]
        if self.ablation.get("shuffle_future_before_router", False):
            router_future = router_future.roll(1, dims=0)
            router_delta = router_delta.roll(1, dims=0)
        if self.ablation.get("force_dense_routing", False):
            router_hard_topk = False
        routing = self.router(
            world_belief=world["world_belief"],
            predicted_future_latents=router_future,
            predicted_delta_latents=router_delta,
            memory_context=memory["memory_context"],
            predicted_uncertainty=world.get("predicted_uncertainty"),
            hard_topk=router_hard_topk,
            temperature=router_temperature,
        )
        expert_features = self.expert_context(
            torch.cat([context["current_visual"], memory["memory_context"], world["world_belief"]], dim=-1)
        )
        experts = self.residual_experts(expert_features, routing["router_probs"], routing["topk_experts"])

        if target_actions is not None and self.training and self.residual_gate_threshold > 0:
            target = target_actions.to(device=nominal_actions.device, dtype=nominal_actions.dtype)
            distance = (nominal_actions.detach() - target).abs().mean(dim=(1, 2))
            residual_gate = (distance > self.residual_gate_threshold).to(nominal_actions.dtype)
            world_loss_gate = action_distance_gate(nominal_actions, target, self.residual_gate_threshold)
        else:
            distance = nominal_actions.new_full((nominal_actions.shape[0],), float("nan"))
            residual_gate = nominal_actions.new_ones((nominal_actions.shape[0],))
            world_loss_gate = nominal_actions.new_ones((nominal_actions.shape[0],))
        gated_residual = experts["action_residual"] * residual_gate[:, None, None]
        final_actions = nominal_actions + gated_residual

        should_compute_teacher = self.training if compute_teacher_targets is None else bool(compute_teacher_targets)
        current_target, future_targets = (None, None)
        if should_compute_teacher:
            current_target, future_targets = self._teacher_targets(batch, final_actions.device)

        output = {
            **world,
            **routing,
            **experts,
            "actions": final_actions,
            "final_actions": final_actions,
            "nominal_actions": nominal_actions,
            "gated_action_residual": gated_residual,
            "residual_gate": residual_gate,
            "nominal_target_distance": distance,
            "action_distance_gate": world_loss_gate,
            "action_condition": action_condition,
            "teacher_forcing_mask": teacher_forcing_mask,
            "memory_context": memory["memory_context"],
        }
        if current_target is not None and future_targets is not None:
            output["current_latent_target"] = current_target
            output["future_latent_targets"] = future_targets
            output["delta_latent_targets"] = future_targets - current_target.unsqueeze(1)
        return output

    def predict_actions(self, batch, router_hard_topk: bool = True):
        was_training = self.training
        self.eval()
        with torch.no_grad():
            output = self.forward(
                batch,
                action_condition_mode="nominal",
                teacher_forcing_probability=0.0,
                router_hard_topk=router_hard_topk,
                compute_teacher_targets=False,
            )
        self.train(was_training)
        return output["final_actions"], output
