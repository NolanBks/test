from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

from openvla_test_utils import synthetic_openvla_identity


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _distributed_checksum_worker(rank: int, world_size: int, output_dir: str, port: int) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    from mowe_wam.training.distributed import initialize_distributed

    cfg = {
        "training": {
            "device": "auto",
            "distributed": {"enabled": "auto", "backend": "gloo"},
        }
    }
    context = initialize_distributed(cfg)
    try:
        torch.manual_seed(11)
        model = torch.nn.Linear(3, 2, bias=False)
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        sample = torch.full((2, 3), float(rank + 1))
        wrapped(sample).square().mean().backward()
        optimizer.step()
        checksum = sum(float(parameter.detach().sum()) for parameter in model.parameters())
        checksums = context.all_gather_objects(checksum)
        if context.is_main:
            Path(output_dir, "rank0.json").write_text(
                json.dumps({"checksums": checksums, "world_size": context.world_size}),
                encoding="utf-8",
            )
    finally:
        context.close()


def _flow_checkpoint_worker(
    rank: int,
    world_size: int,
    root: str,
    port: int,
    stop_step: int,
    resume_path: str | None,
) -> None:
    os.environ.update(
        {
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(port),
            "RANK": str(rank),
            "LOCAL_RANK": str(rank),
            "WORLD_SIZE": str(world_size),
        }
    )
    from mowe_wam.training.flow_runtime import run_flow_training
    from mowe_wam.utils.config import load_config

    root_path = Path(root)
    cfg = load_config("configs/mowe_wam/train_nominal_flow_wam.yaml")
    cfg["output_dir"] = str(root_path / "output")
    cfg["skill_expert_config"] = str(root_path / "skills.json")
    cfg["backbone"].update(
        {
            "mode": "precomputed_features",
            "feature_source": "pre_action_context_cache",
            "checkpoint": "synthetic-openvla",
            "dtype": "float32",
        }
    )
    cfg["teacher"].update(
        {"spatial_tokens": 2, "target_dim": 3, "cache_path": None}
    )
    cfg["data"].update(
        {
            "backend": "mowe_feature_store_v1",
            "feature_store_path": str(root_path / "store"),
            "history_length": 8,
            "long_memory_slots": 4,
            "future_horizons": [1, 4, 8, 16],
            "action_chunk_size": 16,
            "num_workers": 0,
            "pin_memory": False,
        }
    )
    cfg["memory"].update({"hidden_dim": 16, "heads": 4})
    cfg["flow"].update(
        {"hidden_dim": 16, "depth": 2, "num_inference_steps": 2}
    )
    cfg["world_model"].update(
        {"hidden_dim": 16, "layers": 1, "heads": 4, "route_world_dim": 4}
    )
    cfg["router"]["hidden_dim"] = 8
    cfg["view_fusion"]["score_hidden_dim"] = 8
    cfg["training"].update(
        {
            "device": "cpu",
            "precision": "float32",
            "batch_size": 1,
            "grad_accumulation_steps": 1,
            "max_steps": 2,
            "stop_step": stop_step,
            "save_freq": 1,
            "log_freq": 1,
            "distributed": {
                "enabled": "auto",
                "backend": "gloo",
                "timeout_seconds": 120,
                "broadcast_buffers": False,
                "find_unused_parameters": False,
                "memory_guard_fraction": 0.99,
                "gpu_memory_guard_fraction": 0.99,
            },
        }
    )
    cfg["validation"]["enabled"] = False
    with redirect_stdout(io.StringIO()):
        run_flow_training(
            cfg,
            stage="nominal_flow_pretrain",
            resume=resume_path,
        )


def _build_flow_resume_fixture(root: Path) -> None:
    import numpy as np

    from mowe_wam.data.feature_store import MoWEFeatureStoreWriter
    from mowe_wam.utils.config import load_config

    writer = MoWEFeatureStoreWriter(
        root / "store",
        source_contract={
            "rlds_manifest_fingerprint": "ddp-rlds",
            "skill_sidecar_fingerprint": "ddp-sidecar",
            "openvla_checkpoint": "synthetic-openvla",
            "openvla_identity": synthetic_openvla_identity("ddp-resume"),
            "joint_action_statistics": {
                "q01": [-1.0] * 6 + [0.0],
                "q99": [1.0] * 6 + [1.0],
                "mask": [True] * 6 + [False],
            },
        },
        history_length=8,
        long_memory_slots=4,
        future_horizons=(1, 4, 8, 16),
        action_chunk_size=16,
        episodes_per_shard=2,
    )
    for episode_index in range(2):
        length = 18
        steps = np.arange(length, dtype=np.float32)
        views = np.stack(
            [np.full((2, 8), value + episode_index, dtype=np.float32) for value in steps]
        )
        targets = np.stack(
            [np.full((2, 3), value, dtype=np.float32) for value in steps]
        )
        actions = np.zeros((length, 7), dtype=np.float32)
        actions[:, 0] = steps / length
        actions[:, -1] = (steps.astype(np.int64) % 2).astype(np.float32)
        writer.add_episode(
            episode_id=f"ddp-episode-{episode_index}",
            dataset_name="suite-a",
            partition="train",
            language=f"task-{episode_index}",
            language_feature=np.full(8, episode_index, dtype=np.float32),
            openvla_views=views,
            dino_tokens=targets,
            actions=actions,
            skills=(steps.astype(np.int8) % 7),
        )
    writer.finalize()
    skill_cfg = load_config("configs/mowe_wam/skill_experts.yaml")
    skill_cfg["audit"].update(
        {
            "dataset_manifest_fingerprint_sha256": "ddp-rlds",
            "sidecar_fingerprint_sha256": "ddp-sidecar",
            "alignment_verified": False,
            "transitions": 2,
            "label_counts": {"pick_grasp": 2},
        }
    )
    (root / "skills.json").write_text(json.dumps(skill_cfg), encoding="utf-8")


@unittest.skipIf(torch is None, "PyTorch is not installed")
class DistributedRuntimeTests(unittest.TestCase):
    def test_two_rank_flow_checkpoint_resumes_from_step_one_to_two(self):
        if not torch.distributed.is_available():
            self.skipTest("torch.distributed is unavailable")
        from mowe_wam.training.flow_runtime import read_flow_checkpoint_metadata

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _build_flow_resume_fixture(root)
            torch.multiprocessing.spawn(
                _flow_checkpoint_worker,
                args=(2, str(root), _free_port(), 1, None),
                nprocs=2,
                join=True,
            )
            checkpoint = root / "output/checkpoint_latest.pt"
            first = read_flow_checkpoint_metadata(checkpoint)
            self.assertEqual(first["step"], 1)
            self.assertEqual(first["distributed_contract"]["world_size"], 2)
            self.assertEqual(len(first["sampler_state_by_rank"]), 2)

            torch.multiprocessing.spawn(
                _flow_checkpoint_worker,
                args=(2, str(root), _free_port(), 2, str(checkpoint)),
                nprocs=2,
                join=True,
            )
            second = read_flow_checkpoint_metadata(checkpoint)
            self.assertEqual(second["step"], 2)
            self.assertEqual(len(second["sampler_state_by_rank"]), 2)
            state = torch.load(checkpoint, map_location="cpu")
            self.assertEqual(len(state["rng_state_by_rank"]), 2)
            self.assertEqual(
                [item["cursor"] for item in state["sampler_state_by_rank"]],
                [2, 2],
            )
            rows = [
                json.loads(line)
                for line in (root / "output/train_log.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual([row["step"] for row in rows], [1, 2])

    def test_two_rank_parameters_match_and_only_rank_zero_writes(self):
        if not torch.distributed.is_available():
            self.skipTest("torch.distributed is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(
                _distributed_checksum_worker,
                args=(2, directory, _free_port()),
                nprocs=2,
                join=True,
            )
            outputs = list(Path(directory).glob("*.json"))
            self.assertEqual([path.name for path in outputs], ["rank0.json"])
            payload = json.loads(outputs[0].read_text(encoding="utf-8"))
            self.assertEqual(payload["world_size"], 2)
            self.assertAlmostEqual(payload["checksums"][0], payload["checksums"][1], places=7)


if __name__ == "__main__":
    unittest.main()
