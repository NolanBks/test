"""World-predicate prediction head."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:
    torch = None
    nn = None

from mowe_wam.utils.optional import require_torch
from mowe_wam.predicates.schema import predicate_index


class WorldPredicateHead(nn.Module if nn is not None else object):
    def __init__(self, hidden_dim: int, predicate_dim: int, hidden_layers: list[int] | None = None) -> None:
        require_torch()
        super().__init__()
        layers = []
        dims = [hidden_dim] + list(hidden_layers or [512]) + [predicate_dim]
        for idx in range(len(dims) - 2):
            layers.append(nn.Linear(dims[idx], dims[idx + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        logits = self.net(features)
        predicates = torch.sigmoid(logits)
        progress_idx = predicate_index("progress_score")
        risk_idx = predicate_index("failure_risk")
        return {
            "predicate_logits": logits,
            "predicates": predicates,
            "progress": predicates[:, progress_idx : progress_idx + 1],
            "risk_logits": logits[:, risk_idx : risk_idx + 1],
            "risk": predicates[:, risk_idx : risk_idx + 1],
        }
