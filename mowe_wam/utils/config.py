"""Small config loader used before the full training stack is installed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _merge_config(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a JSON-compatible YAML config.

    The project configs are written as JSON-compatible YAML so they can be read
    without PyYAML during local planning and smoke checks. If PyYAML is
    available, we use it first.
    """

    config_path = Path(path).resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"Config inheritance cycle at {config_path}")
    seen.add(config_path)
    text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except ModuleNotFoundError:
        data = json.loads(text)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping config in {config_path}, got {type(data).__name__}")
    base_path = data.pop("base_config", None)
    if base_path is None:
        return data
    candidate = Path(base_path)
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return _merge_config(load_config(candidate, seen), data)


def dump_resolved_config(config: dict[str, Any]) -> str:
    """Return a stable, readable config string."""

    return json.dumps(config, indent=2, sort_keys=True)
