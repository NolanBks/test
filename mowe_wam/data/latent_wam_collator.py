"""Collation for fixed-size latent-WAM episode windows."""

from __future__ import annotations

from typing import Any

from mowe_wam.utils.optional import require_torch


class LatentWAMCollator:
    """Stack tensor fields while preserving text and episode identifiers."""

    _TEXT_FIELDS = {"episode_id", "dataset_name", "language"}

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        torch = require_torch()
        if not instances:
            raise ValueError("Cannot collate an empty list.")
        keys = set(instances[0])
        if any(set(instance) != keys for instance in instances):
            common = set.intersection(*(set(instance) for instance in instances))
            required = keys - {"proprio"}
            if not required.issubset(common):
                raise ValueError("Sequence samples have inconsistent required fields.")

        batch: dict[str, Any] = {}
        for key in sorted(keys - {"proprio"}):
            values = [instance[key] for instance in instances]
            if key in self._TEXT_FIELDS:
                batch[key] = [str(value) for value in values]
            elif key == "step_id":
                batch[key] = torch.tensor(values, dtype=torch.long)
            elif torch.is_tensor(values[0]):
                batch[key] = torch.stack(values, dim=0)
            else:
                batch[key] = values

        proprio_values = [instance.get("proprio") for instance in instances]
        if any(value is not None for value in proprio_values):
            reference = next(value for value in proprio_values if value is not None)
            batch["proprio"] = torch.stack(
                [torch.zeros_like(reference) if value is None else value for value in proprio_values], dim=0
            )
            batch["proprio_mask"] = torch.tensor(
                [value is not None for value in proprio_values], dtype=torch.bool
            )
        return batch

