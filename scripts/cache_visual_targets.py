#!/usr/bin/env python3
"""Build a sharded, deduplicated DINO feature cache for Flow-WAM training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inspect_latent_sequence_dataset import load_processor

from mowe_wam.backbones import VisualTargetEncoder, teacher_transform_metadata
from mowe_wam.data import (
    LatentWAMCollator,
    LiberoSequenceDataset,
    ShardedVisualTargetCacheWriter,
    feature_cache_key,
    rlds_manifest_fingerprint,
)
from mowe_wam.training.flow_runtime import deep_update, resolve_device
from mowe_wam.utils.config import load_config
from mowe_wam.utils.optional import require_torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/mowe_wam/train_nominal_flow_wam.yaml")
    parser.add_argument("--data-root")
    parser.add_argument("--processor-checkpoint")
    parser.add_argument("--teacher-checkpoint")
    parser.add_argument("--output", required=True, help="New cache directory; must not already contain a cache.")
    parser.add_argument("--limit", type=int, help="Optional number of Flow-WAM windows for a smoke cache.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encode-batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=4096)
    return parser.parse_args()


def _metadata(cfg, resolution, sidecar_metadata) -> dict:
    transform = teacher_transform_metadata(
        cfg["teacher"]["checkpoint"],
        resolution,
        int(cfg["teacher"].get("spatial_grid", 4)),
    )
    return {
        "teacher_checkpoint": cfg["teacher"]["checkpoint"],
        "spatial_tokens": int(cfg["teacher"].get("spatial_tokens", 16)),
        "target_dim": int(cfg["teacher"].get("target_dim", 384)),
        "future_horizons": list(cfg["data"]["future_horizons"]),
        "context_views": list(cfg["data"]["observation_views"]),
        "teacher_target_views": list(cfg["teacher"]["target_views"]),
        "dataset_names": list(cfg["data"]["dataset_names"]),
        "dataset_fingerprint": rlds_manifest_fingerprint(
            cfg["data"]["data_root"], cfg["data"]["dataset_names"]
        ),
        "skill_sidecar_fingerprint": sidecar_metadata.get("fingerprint_sha256"),
        "image_resolution": transform["image_resolution"],
        "transform_id": transform["transform_id"],
        "transform_hash": transform["transform_hash"],
        "storage_contract": "one_float16_feature_per_episode_timestep",
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.encode_batch_size < 1:
        raise SystemExit("--batch-size and --encode-batch-size must be positive.")
    torch = require_torch()
    cfg = load_config(args.config)
    deep_update(
        cfg,
        {
            "data": {
                "data_root": args.data_root,
                "limit_batches": args.limit,
                "window_shuffle_buffer_size": 0,
            },
            "backbone": {"checkpoint": args.processor_checkpoint},
            "teacher": {"checkpoint": args.teacher_checkpoint, "cache_path": None},
            "training": {"batch_size": args.batch_size},
        },
    )
    if cfg["data"].get("data_root") in {None, "TBD"}:
        raise SystemExit("Teacher caching requires --data-root or a concrete config data root.")
    processor_checkpoint = cfg["backbone"].get("checkpoint") or cfg["backbone"].get("vla_path")
    if processor_checkpoint in {None, "TBD"}:
        raise SystemExit("Teacher caching requires --processor-checkpoint or a concrete backbone checkpoint.")
    processor = load_processor(processor_checkpoint, cfg["backbone"].get("openvla_root"))
    resolution = tuple(processor.image_processor.input_sizes[0][-2:])
    cfg["backbone"]["image_resolution"] = list(resolution)
    dataset = LiberoSequenceDataset(
        dataset_root=cfg["data"]["data_root"],
        processor=processor,
        dataset_names=cfg["data"]["dataset_names"],
        history_length=int(cfg["data"].get("history_length", 8)),
        long_memory_slots=int(cfg["data"].get("long_memory_slots", 4)),
        future_horizons=cfg["data"].get("future_horizons", [1, 4, 8]),
        split=cfg["data"].get("split", "train"),
        resize_resolution=resolution,
        image_aug=False,
        use_proprio=False,
        openvla_root=cfg["backbone"].get("openvla_root", "external/openvla-oft"),
        limit=cfg["data"].get("limit_batches"),
        joint_action_normalization=bool(cfg["data"].get("joint_action_normalization", True)),
        skill_sidecar_path=cfg["data"].get("skill_sidecar_path"),
        assume_sidecar_timestep_aligned=bool(
            cfg["data"].get("assume_sidecar_timestep_aligned", True)
        ),
        window_shuffle_buffer_size=0,
        cache_only=args.limit is None,
        action_chunk_size=int(cfg["data"].get("action_chunk_size", 8)),
    )
    loader = None
    if args.limit is not None:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            collate_fn=LatentWAMCollator(),
            num_workers=0,
        )
    device = resolve_device(cfg)
    teacher_dtype = (
        "float32" if device == "cpu" else cfg["teacher"].get("dtype", cfg["training"].get("precision", "bf16"))
    )
    teacher = VisualTargetEncoder(
        checkpoint=cfg["teacher"]["checkpoint"],
        spatial_grid=int(cfg["teacher"].get("spatial_grid", 4)),
        target_dim=int(cfg["teacher"].get("target_dim", 384)),
        num_spatial_tokens=int(cfg["teacher"].get("spatial_tokens", 16)),
        device=device,
        dtype=teacher_dtype,
    )
    writer = ShardedVisualTargetCacheWriter(
        args.output,
        _metadata(cfg, resolution, dataset.skill_sidecar_metadata or {}),
        shard_size=args.shard_size,
    )
    windows = 0
    if loader is not None:
        for batch in loader:
            candidates = {}
            for batch_index, episode_id in enumerate(batch["episode_id"]):
                step_id = int(batch["step_id"][batch_index])
                current_key = feature_cache_key(episode_id, step_id)
                if current_key not in writer:
                    candidates.setdefault(current_key, batch["current_raw_pixel_values"][batch_index])
                for horizon_index, horizon in enumerate(batch["future_horizons"][batch_index].tolist()):
                    future_key = feature_cache_key(episode_id, step_id + int(horizon))
                    if future_key not in writer:
                        candidates.setdefault(
                            future_key,
                            batch["future_raw_pixel_values"][batch_index, horizon_index],
                        )
            candidate_items = list(candidates.items())
            for start in range(0, len(candidate_items), args.encode_batch_size):
                chunk = candidate_items[start : start + args.encode_batch_size]
                images = torch.stack([image for _, image in chunk], dim=0)
                features = teacher.encode(images)
                for (key, _), feature in zip(chunk, features):
                    writer.add(key, feature)
            windows += len(batch["episode_id"])
            if windows % 1000 < len(batch["episode_id"]):
                print(f"processed_windows={windows}", flush=True)
    else:
        pending = []
        next_progress = 10_000

        def encode_pending():
            if not pending:
                return
            images = torch.stack([image for _, image in pending], dim=0)
            features = teacher.encode(images)
            for (key, _), feature in zip(pending, features):
                writer.add(key, feature)
            pending.clear()

        for record in dataset.iter_episode_timesteps():
            key = feature_cache_key(record["episode_id"], record["step_id"])
            if key not in writer:
                pending.append((key, record["raw_pixel_values"]))
            if len(pending) >= args.encode_batch_size:
                encode_pending()
            if writer.record_count >= next_progress:
                print(f"processed_timesteps={writer.record_count}", flush=True)
                next_progress += 10_000
        encode_pending()
    manifest = writer.close()
    print(
        f"saved {writer.record_count} unique timestep features from "
        f"{windows if loader is not None else 'direct episode traversal'} to {manifest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
