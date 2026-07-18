"""Backbone adapters for MoWE-WAM."""

from mowe_wam.backbones.openvla_identity import (
    OPENVLA_IDENTITY_FORMAT,
    ORIGINAL_OPENVLA_REPO_ID,
    openvla_identities_match,
    resolve_original_openvla_identity,
    validate_openvla_identity,
    validate_original_openvla_reference,
)
from mowe_wam.backbones.openvla_oft_adapter import OpenVLAContextAdapter, OpenVLAOFTAdapter
from mowe_wam.backbones.precomputed_features import PrecomputedFeatureBackbone
from mowe_wam.backbones.visual_target_encoder import VisualTargetEncoder, teacher_transform_metadata

__all__ = [
    "OPENVLA_IDENTITY_FORMAT",
    "ORIGINAL_OPENVLA_REPO_ID",
    "OpenVLAContextAdapter",
    "OpenVLAOFTAdapter",
    "PrecomputedFeatureBackbone",
    "VisualTargetEncoder",
    "teacher_transform_metadata",
    "openvla_identities_match",
    "resolve_original_openvla_identity",
    "validate_openvla_identity",
    "validate_original_openvla_reference",
]
