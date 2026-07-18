"""Immutable identity contract for the frozen original OpenVLA backbone."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


ORIGINAL_OPENVLA_REPO_ID = "openvla/openvla-7b"
OPENVLA_IDENTITY_FORMAT = "openvla_backbone_identity_v1"
_FULL_REVISION = re.compile(r"^[0-9a-f]{40}$")
_HEX_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DISALLOWED_REFERENCE_MARKERS = (
    "oft-finetuned",
    "openvla-7b-oft",
    "finetuned-libero",
    "libero-all",
)


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_digest(path: Path) -> str:
    """Use a Hugging Face blob digest when available, otherwise hash bytes."""

    if path.is_symlink():
        target = path.resolve()
        if _HEX_DIGEST.fullmatch(target.name.lower()):
            return target.name.lower()
    return _sha256_file(path)


def _fingerprint_files(root: Path, files: list[Path]) -> tuple[str, list[dict[str, Any]]]:
    records = []
    unique = {str(value.absolute()): value.absolute() for value in files}
    for path in sorted(unique.values(), key=lambda value: str(value)):
        try:
            relative = path.relative_to(root)
        except ValueError:
            relative = Path(path.name)
        records.append(
            {
                "path": relative.as_posix(),
                "size": int(path.stat().st_size),
                "sha256": _content_digest(path),
            }
        )
    return _json_sha256(records), records


def _infer_snapshot_revision(path: Path) -> str | None:
    resolved = path.resolve()
    parts = resolved.parts
    for index, part in enumerate(parts[:-1]):
        if part == "snapshots" and index + 1 < len(parts):
            candidate = parts[index + 1].lower()
            if _FULL_REVISION.fullmatch(candidate):
                return candidate
    candidate = resolved.name.lower()
    return candidate if _FULL_REVISION.fullmatch(candidate) else None


def _validate_revision(revision: str | None) -> str:
    normalized = str(revision or "").strip().lower()
    if not _FULL_REVISION.fullmatch(normalized):
        raise ValueError(
            "Original OpenVLA requires an immutable 40-character Hugging Face commit revision."
        )
    return normalized


def validate_original_openvla_reference(
    checkpoint: str | Path,
    *,
    repo_id: str = ORIGINAL_OPENVLA_REPO_ID,
    revision: str | None = None,
) -> tuple[str, str]:
    """Reject benchmark-finetuned references and return repo/revision."""

    normalized_repo = str(repo_id).strip()
    if normalized_repo != ORIGINAL_OPENVLA_REPO_ID:
        raise ValueError(
            f"Frozen backbone repo must be {ORIGINAL_OPENVLA_REPO_ID!r}, got {normalized_repo!r}."
        )
    rendered = str(checkpoint)
    lowered = rendered.lower()
    if any(marker in lowered for marker in _DISALLOWED_REFERENCE_MARKERS):
        raise ValueError(
            "LIBERO-finetuned OpenVLA-OFT checkpoints are forbidden in the original-backbone mainline."
        )
    path = Path(rendered).expanduser()
    inferred = _infer_snapshot_revision(path) if path.exists() else None
    requested_revision = None if revision in {None, "", "TBD"} else str(revision)
    return normalized_repo, _validate_revision(requested_revision or inferred)


def resolve_original_openvla_identity(
    checkpoint: str | Path,
    *,
    revision: str | None = None,
    repo_id: str = ORIGINAL_OPENVLA_REPO_ID,
    require_local: bool = True,
) -> dict[str, Any]:
    """Resolve and fingerprint one immutable local snapshot of original OpenVLA.

    Weight files are hashed once during conversion/evaluation identity resolution.
    Training from precomputed features consumes the recorded identity and never
    reopens the 7B snapshot.
    """

    normalized_repo, normalized_revision = validate_original_openvla_reference(
        checkpoint, repo_id=repo_id, revision=revision
    )
    path = Path(checkpoint).expanduser()
    if not path.exists():
        if require_local:
            raise FileNotFoundError(
                "Formal OpenVLA conversion/evaluation requires a local immutable snapshot: "
                f"{path}"
            )
        return {
            "format": OPENVLA_IDENTITY_FORMAT,
            "repo_id": normalized_repo,
            "revision": normalized_revision,
            "checkpoint": str(checkpoint),
            "local_snapshot": False,
            "config_fingerprint_sha256": None,
            "processor_fingerprint_sha256": None,
            "weight_fingerprint_sha256": None,
            "identity_sha256": _json_sha256(
                {"repo_id": normalized_repo, "revision": normalized_revision}
            ),
        }
    if not path.is_dir():
        raise ValueError(f"OpenVLA checkpoint must be a snapshot directory, got {path}.")
    path = path.resolve()
    config_path = path / "config.json"
    if not config_path.is_file():
        raise ValueError(f"OpenVLA snapshot is missing config.json: {path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"OpenVLA config.json is unreadable: {config_path}") from exc
    if config.get("model_type") != "openvla":
        raise ValueError(
            f"Expected config.model_type='openvla', got {config.get('model_type')!r}."
        )

    weight_files = sorted(path.glob("*.safetensors")) + sorted(path.glob("*.bin"))
    if not weight_files:
        raise ValueError(f"OpenVLA snapshot contains no .safetensors or .bin weights: {path}")
    processor_files = []
    for pattern in (
        "preprocessor*",
        "processor*",
        "tokenizer*",
        "special_tokens*",
        "*.model",
    ):
        processor_files.extend(value for value in path.glob(pattern) if value.is_file())
    if not processor_files:
        raise ValueError(f"OpenVLA snapshot contains no processor/tokenizer artifacts: {path}")

    config_fingerprint, config_records = _fingerprint_files(path, [config_path])
    processor_fingerprint, processor_records = _fingerprint_files(path, processor_files)
    weight_fingerprint, weight_records = _fingerprint_files(path, weight_files)
    semantic = {
        "repo_id": normalized_repo,
        "revision": normalized_revision,
        "config_fingerprint_sha256": config_fingerprint,
        "processor_fingerprint_sha256": processor_fingerprint,
        "weight_fingerprint_sha256": weight_fingerprint,
    }
    return {
        "format": OPENVLA_IDENTITY_FORMAT,
        **semantic,
        "identity_sha256": _json_sha256(semantic),
        "checkpoint": str(path),
        "local_snapshot": True,
        "files": {
            "config": config_records,
            "processor": processor_records,
            "weights": weight_records,
        },
    }


def validate_openvla_identity(identity: dict[str, Any], *, require_fingerprints: bool = True) -> dict[str, Any]:
    if not isinstance(identity, dict) or identity.get("format") != OPENVLA_IDENTITY_FORMAT:
        raise ValueError("Missing or unsupported original OpenVLA identity contract.")
    if identity.get("repo_id") != ORIGINAL_OPENVLA_REPO_ID:
        raise ValueError("Backbone identity is not the original openvla/openvla-7b repository.")
    _validate_revision(identity.get("revision"))
    required = (
        "config_fingerprint_sha256",
        "processor_fingerprint_sha256",
        "weight_fingerprint_sha256",
    )
    if require_fingerprints and any(not identity.get(key) for key in required):
        raise ValueError("Original OpenVLA identity is missing immutable artifact fingerprints.")
    for key in required:
        value = identity.get(key)
        if value is not None and not _HEX_DIGEST.fullmatch(str(value).lower()):
            raise ValueError(f"Original OpenVLA identity has an invalid {key} digest.")
    semantic = {
        "repo_id": identity.get("repo_id"),
        "revision": identity.get("revision"),
        **{key: identity.get(key) for key in required},
    }
    expected = _json_sha256(semantic)
    if not _HEX_DIGEST.fullmatch(str(identity.get("identity_sha256", "")).lower()):
        raise ValueError("Original OpenVLA identity has an invalid identity_sha256 digest.")
    if identity.get("identity_sha256") != expected:
        raise ValueError("Original OpenVLA identity fingerprint is internally inconsistent.")
    return identity


def openvla_identities_match(left: dict[str, Any], right: dict[str, Any]) -> bool:
    try:
        validate_openvla_identity(left)
        validate_openvla_identity(right)
    except ValueError:
        return False
    return str(left["identity_sha256"]) == str(right["identity_sha256"])
