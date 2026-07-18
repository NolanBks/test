"""Optional dependency helpers."""

from __future__ import annotations


def require_torch():
    """Import torch or raise an actionable error."""

    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for this command. Install torch in the active "
            "environment before running model forward or training smoke tests."
        ) from exc
    return torch
