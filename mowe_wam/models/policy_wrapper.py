"""Wrapper that connects a frozen backbone to MoWE-WAM heads."""

from __future__ import annotations

try:
    from torch import nn
except ModuleNotFoundError:
    nn = None

from mowe_wam.utils.optional import require_torch


class MoWEPolicyWrapper(nn.Module if nn is not None else object):
    def __init__(self, backbone, world_head, router, experts, memory_encoder=None, predictive: bool = False) -> None:
        require_torch()
        super().__init__()
        self.backbone = backbone
        self.world_head = world_head
        self.router = router
        self.experts = experts
        self.memory_encoder = memory_encoder
        self.predictive = bool(predictive)

    def train(self, mode: bool = True):
        """Keep a frozen OpenVLA backbone in inference mode during head training."""

        super().train(mode)
        if bool(getattr(self.backbone, "freeze_backbone", False)):
            keep_eval = getattr(self.backbone, "keep_frozen_backbone_eval", None)
            if callable(keep_eval):
                keep_eval()
            else:
                self.backbone.eval()
        return self

    def extract_features(self, batch: dict):
        if "features" in batch:
            features = batch["features"]
        elif hasattr(self.backbone, "extract_features"):
            features = self.backbone.extract_features(batch)
        else:
            raise NotImplementedError(
                "Backbone must expose extract_features(batch). Pass batch['features'] only for synthetic checks."
            )
        if getattr(features, "dim", lambda: 0)() == 3:
            features = features[:, -1]
        # The frozen backbone may emit bf16 features while newly initialized
        # local heads are fp32. Autocast handles the fast path, but explicit
        # conversion keeps the float32 fallback configuration valid as well.
        try:
            head_dtype = next(self.experts.parameters()).dtype
            if getattr(features, "is_floating_point", lambda: False)():
                features = features.to(dtype=head_dtype)
        except StopIteration:
            pass
        return features

    def _predictive_forward(self, features, batch: dict, use_oracle_future: bool = False):
        if self.memory_encoder is None:
            raise RuntimeError("Predictive MoWE wrapper requires an EventMemoryEncoder.")
        required = ("history_actions", "history_predicates", "memory_state")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"Predictive MoWE batch is missing: {', '.join(missing)}")
        memory_state = batch["memory_state"].to(device=features.device, dtype=features.dtype)
        memory_context = self.memory_encoder(memory_state)
        transition_outputs = self.world_head(
            features,
            batch["history_actions"].to(device=features.device, dtype=features.dtype),
            batch["history_predicates"].to(device=features.device, dtype=features.dtype),
            memory_context,
        )
        if use_oracle_future:
            future_predicates = batch["future_predicates"].to(device=features.device, dtype=features.dtype)
            progress_delta = batch["progress_delta"].to(device=features.device, dtype=features.dtype)
            future_risk = batch["future_risk"].to(device=features.device, dtype=features.dtype)
            future_recovery = batch["future_recovery"].to(device=features.device, dtype=features.dtype)
        else:
            future_predicates = transition_outputs["future_predicates"]
            progress_delta = transition_outputs["progress_delta"]
            future_risk = transition_outputs["future_risk"]
            future_recovery = transition_outputs["future_recovery"]
        router_outputs = self.router(
            features,
            future_predicates,
            progress_delta,
            future_risk,
            future_recovery,
            memory_context,
            previous_expert=batch.get("previous_expert"),
        )
        expert_outputs = self.experts(features, router_outputs["router_probs"], router_outputs["topk_experts"])
        return {
            **transition_outputs,
            "memory_context": memory_context,
            "event_memory_state": memory_state,
            **router_outputs,
            **expert_outputs,
        }

    def forward(self, batch: dict, use_oracle_predicates: bool = False, use_oracle_future: bool = False):
        features = self.extract_features(batch)
        if self.predictive:
            return self._predictive_forward(features, batch, use_oracle_future=use_oracle_future)
        world_outputs = self.world_head(features)
        predicates = batch["predicates"] if use_oracle_predicates and "predicates" in batch else world_outputs["predicates"]
        router_outputs = self.router(features, predicates)
        expert_outputs = self.experts(features, router_outputs["router_probs"], router_outputs["topk_experts"])
        return {**world_outputs, **router_outputs, **expert_outputs}
