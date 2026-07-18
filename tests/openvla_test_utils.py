from __future__ import annotations

import hashlib
import json


def synthetic_openvla_identity(tag: str = "test") -> dict[str, object]:
    """Build a deterministic, structurally valid identity without model weights."""

    def digest(suffix: str) -> str:
        return hashlib.sha256(f"{tag}:{suffix}".encode("utf-8")).hexdigest()

    semantic = {
        "repo_id": "openvla/openvla-7b",
        "revision": hashlib.sha1(tag.encode("utf-8")).hexdigest(),
        "config_fingerprint_sha256": digest("config"),
        "processor_fingerprint_sha256": digest("processor"),
        "weight_fingerprint_sha256": digest("weight"),
    }
    identity_sha256 = hashlib.sha256(
        json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "format": "openvla_backbone_identity_v1",
        **semantic,
        "identity_sha256": identity_sha256,
        "checkpoint": f"/synthetic/{tag}",
        "local_snapshot": True,
    }
