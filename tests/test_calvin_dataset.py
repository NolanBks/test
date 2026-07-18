from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mowe_wam.benchmarks.calvin import CalvinActionAdapter
from mowe_wam.benchmarks.calvin.dataset import (
    CalvinLanguageSegmentDataset,
    resolve_calvin_abc_training_root,
)
from scripts.convert_calvin_to_mowe_store import _encode_segment
from scripts.audit_calvin_feature_store_equivalence import (
    _window_from_encoded_segment,
)


class CalvinDatasetTests(unittest.TestCase):
    @staticmethod
    def _build_dataset(root: Path) -> Path:
        training = root / "task_ABC_D/training"
        annotations = training / "lang_annotations"
        annotations.mkdir(parents=True)
        verbs = ["pick", "place", "move", "open", "turn", "push"]
        spans = []
        languages = []
        tasks = []
        frame_index = 0
        for segment_index, verb in enumerate(verbs):
            start = frame_index
            for local_index in range(9):
                motion = np.asarray(
                    [
                        -0.8 + 1.6 * (frame_index / 53.0) + dimension * 0.001
                        for dimension in range(6)
                    ],
                    dtype=np.float32,
                )
                action = np.concatenate(
                    [motion, np.asarray([1.0 if frame_index % 2 else -1.0], dtype=np.float32)]
                )
                np.savez(
                    training / f"episode_{frame_index:07d}.npz",
                    rgb_static=np.full(
                        (200, 200, 3), segment_index + local_index, dtype=np.uint8
                    ),
                    rgb_gripper=np.full(
                        (84, 84, 3), segment_index + local_index, dtype=np.uint8
                    ),
                    robot_obs=np.full(15, frame_index, dtype=np.float32),
                    rel_actions=action,
                )
                frame_index += 1
            spans.append([start, frame_index - 1])
            languages.append(f"{verb} the target object")
            tasks.append(f"task_{segment_index}")
        np.save(
            annotations / "auto_lang_ann.npy",
            {
                "info": {"indx": np.asarray(spans, dtype=np.int64)},
                "language": {
                    "ann": np.asarray(languages, dtype=object),
                    "task": np.asarray(tasks, dtype=object),
                    "emb": np.zeros((len(spans), 1), dtype=np.float32),
                },
            },
        )
        return training

    def test_official_schema_audit_and_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = self._build_dataset(root)
            dataset = CalvinLanguageSegmentDataset(root / "task_ABC_D")
            self.assertEqual(dataset.root, training.resolve())
            self.assertEqual(len(dataset.valid_segments), 6)
            report = dataset.audit()
            self.assertTrue(report["passed"], report)
            self.assertEqual(report["transitions"], 54)
            self.assertEqual(report["valid_windows_h8"], 6)
            self.assertEqual(report["unknown_ratio"], 0.0)
            self.assertTrue(report["all_motor_classes_present"])
            self.assertEqual(report["action_statistics"]["frame_count"], 54)
            self.assertEqual(
                report["action_statistics"]["gripper_counts"],
                {"-1.0": 27, "1.0": 27},
            )
            segment = next(iter(dataset.iter_segments(limit=1)))
            self.assertEqual(segment["rgb_static"].shape, (9, 200, 200, 3))
            self.assertEqual(segment["rgb_gripper"].shape, (9, 84, 84, 3))
            self.assertEqual(segment["rel_actions"].shape, (9, 7))
            self.assertTrue(np.all(segment["skill_ids"] == 0))
            config = dataset.skill_config(report, audit_path="audit.json")
            self.assertEqual(
                sum(config["audit"]["label_counts"].values()),
                config["audit"]["transitions"],
            )
            self.assertTrue(all(value > 0 for value in config["class_weights_inverse_sqrt"]))

            validation = root / "task_ABC_D/validation"
            validation.mkdir()
            with self.assertRaisesRegex(ValueError, "validation/D"):
                resolve_calvin_abc_training_root(validation)

    def test_rejects_non_abc_split_root(self):
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory) / "task_D_D/training"
            other.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "task_ABC_D"):
                resolve_calvin_abc_training_root(other.parent)
            with self.assertRaisesRegex(ValueError, "task_ABC_D"):
                resolve_calvin_abc_training_root(other)

    def test_segment_encoding_handles_official_camera_shapes(self):
        class ImageProcessor:
            @staticmethod
            def apply_transform(image):
                array = np.asarray(image, dtype=np.uint8).copy()
                return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

        class Processor:
            image_processor = ImageProcessor()

        class Backbone:
            processor = Processor()

            @staticmethod
            def encode_pooled_views(primary, wrist):
                values = torch.stack(
                    [primary.mean(dim=(1, 2, 3)), wrist.mean(dim=(1, 2, 3))], dim=1
                )
                return values.unsqueeze(-1).repeat(1, 1, 4)

            @staticmethod
            def encode_language_tokens(languages):
                return torch.ones(len(languages), 2, 4), torch.ones(
                    len(languages), 2, dtype=torch.bool
                )

        class Teacher:
            @staticmethod
            def encode(images):
                values = images.float().mean(dim=(1, 2, 3))
                return values[:, None, None].repeat(1, 2, 3)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._build_dataset(root)
            dataset = CalvinLanguageSegmentDataset(root / "task_ABC_D")
            report = dataset.audit()
            adapter = CalvinActionAdapter.from_config(report["action_statistics"])
            segment = next(iter(dataset.iter_segments(limit=1)))
            encoded = _encode_segment(
                torch, Backbone(), Teacher(), adapter, segment, batch_size=4
            )
            self.assertEqual(encoded["openvla_views"].shape, (9, 2, 4))
            self.assertEqual(encoded["dino_tokens"].shape, (9, 2, 3))
            self.assertEqual(encoded["language_feature"].shape, (4,))
            self.assertEqual(encoded["actions"].shape, (9, 7))
            self.assertTrue(np.isin(encoded["actions"][:, 6], [0.0, 1.0]).all())

            segment["episode_id"] = dataset.segment_episode_id(
                next(iter(dataset.iter_segment_records(limit=1)))
            )
            window = _window_from_encoded_segment(
                torch,
                encoded,
                segment,
                0,
                {
                    "history_length": 8,
                    "long_memory_slots": 4,
                    "future_horizons": [1, 4, 8],
                    "action_chunk_size": 8,
                },
            )
            self.assertEqual(window["episode_id"], segment["episode_id"])
            self.assertEqual(tuple(window["history_visual_views"].shape), (7, 2, 4))
            self.assertEqual(window["history_mask"].tolist(), [False] * 7 + [True])
            self.assertEqual(tuple(window["future_latent_targets"].shape), (3, 2, 3))
            self.assertEqual(tuple(window["target_actions"].shape), (8, 7))


if __name__ == "__main__":
    unittest.main()
