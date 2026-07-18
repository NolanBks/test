#!/usr/bin/env python3
"""Inspect real same-episode latent-WAM windows without loading the 7B model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mowe_wam.backbones.openvla_oft_adapter import _ensure_openvla_path
from mowe_wam.backbones import validate_original_openvla_reference
from mowe_wam.data import LatentWAMCollator, LiberoSequenceDataset
from mowe_wam.utils.optional import require_torch


def load_processor(checkpoint: str, revision: str, openvla_root: str):
    _ensure_openvla_path(openvla_root)
    validate_original_openvla_reference(checkpoint, revision=revision)
    from transformers import AutoConfig, AutoImageProcessor, AutoProcessor
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    except ValueError:
        pass
    kwargs = {"trust_remote_code": True}
    if not Path(checkpoint).expanduser().exists():
        kwargs["revision"] = revision
    return AutoProcessor.from_pretrained(checkpoint, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-revision", required=True)
    parser.add_argument("--openvla-root", default="external/openvla-oft")
    parser.add_argument("--dataset-name", action="append")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--skill-sidecar", default="datasets/libero_cot_rlds/cot_file.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch = require_torch()
    processor = load_processor(
        args.checkpoint, args.backbone_revision, args.openvla_root
    )
    resolution = tuple(processor.image_processor.input_sizes[0][-2:])
    dataset = LiberoSequenceDataset(
        args.data_root,
        processor,
        dataset_names=args.dataset_name or ["libero_spatial_no_noops"],
        resize_resolution=resolution,
        limit=args.limit,
        openvla_root=args.openvla_root,
        skill_sidecar_path=args.skill_sidecar,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=LatentWAMCollator(), num_workers=0)
    for batch in loader:
        report = {
            "episode_id": batch["episode_id"][0],
            "step_id": int(batch["step_id"][0]),
            "dataset_name": batch["dataset_name"][0],
            "history_pixel_values_primary": list(batch["history_pixel_values_primary"].shape),
            "history_pixel_values_wrist": list(batch["history_pixel_values_wrist"].shape),
            "history_actions": list(batch["history_actions"].shape),
            "history_mask": batch["history_mask"][0].tolist(),
            "long_history_pixel_values_primary": list(batch["long_history_pixel_values_primary"].shape),
            "long_history_pixel_values_wrist": list(batch["long_history_pixel_values_wrist"].shape),
            "long_history_mask": batch["long_history_mask"][0].tolist(),
            "future_raw_pixel_values": list(batch["future_raw_pixel_values"].shape),
            "future_horizons": batch["future_horizons"][0].tolist(),
            "target_actions": list(batch["target_actions"].shape),
            "target_motion": list(batch["target_motion"].shape),
            "target_gripper": list(batch["target_gripper"].shape),
            "gripper_values": sorted(set(batch["target_gripper"].flatten().tolist())),
            "expert_skill_labels": batch["expert_skill_labels"][0].tolist(),
            "expert_skill_mask": batch["expert_skill_mask"][0].tolist(),
            "expert_label_source": batch["expert_label_source"][0],
        }
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
