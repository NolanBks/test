from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mowe_wam.data.canonical_archive import (
    MoWECanonicalArchiveWriter,
    audit_canonical_archive,
    canonical_conversion_environment,
    load_canonical_archive_manifest,
)


class CanonicalArchiveTests(unittest.TestCase):
    @staticmethod
    def _episode(
        offset: int,
        length: int = 3,
        primary_shape: tuple[int, int] = (16, 16),
        wrist_shape: tuple[int, int] = (16, 16),
    ):
        primary = np.stack(
            [
                np.full((*primary_shape, 3), offset + index, dtype=np.uint8)
                for index in range(length)
            ]
        )
        wrist = np.stack(
            [
                np.full((*wrist_shape, 3), 100 + offset + index, dtype=np.uint8)
                for index in range(length)
            ]
        )
        actions = np.zeros((length, 7), dtype=np.float32)
        actions[:, 0] = np.arange(length, dtype=np.float32) + offset
        actions[:, 6] = np.arange(length) % 2
        skills = np.asarray([0, 1, -1], dtype=np.int8)[:length]
        proprio = np.full((length, 4), offset, dtype=np.float32)
        return primary, wrist, actions, skills, proprio

    @staticmethod
    def _source_contract():
        return {
            "rlds_manifest_fingerprint": "rlds-test",
            "skill_sidecar_fingerprint": "sidecar-test",
            "joint_action_statistics": {
                "q01": [-1.0] * 7,
                "q99": [1.0] * 7,
                "mask": [True] * 6 + [False],
            },
        }

    def _add(self, writer, index: int):
        primary, wrist, actions, skills, proprio = self._episode(index * 10)
        return writer.add_episode(
            episode_id=f"suite:episode-{index}",
            dataset_name="libero_spatial_no_noops",
            partition="train" if index == 0 else "validation",
            language="pick up the block" if index == 0 else "put down the block",
            actions=actions,
            skills=skills,
            primary_frames=primary,
            wrist_frames=wrist,
            proprio=proprio,
            source_traj_index=index,
            source_file_key="/data/source.hdf5",
        )

    def test_real_parquet_mp4_resume_and_audit(self):
        environment = canonical_conversion_environment()
        self.assertIn("pyarrow", environment)
        self.assertTrue(Path(environment["ffmpeg"]).exists())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "canonical"
            writer = MoWECanonicalArchiveWriter(
                root,
                source_contract=self._source_contract(),
                fps=10.0,
                episodes_per_chunk=2,
                video_crf=18,
                video_preset="ultrafast",
            )
            self.assertTrue(self._add(writer, 0))
            self.assertTrue(writer.has_episode("suite:episode-0"))
            self.assertEqual(len(writer.pending), 1)
            staging = next((root / ".staging").glob("episode-*.npz"))
            original = staging.read_bytes()
            staging.write_bytes(original[:-1] + bytes([original[-1] ^ 0xFF]))
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                MoWECanonicalArchiveWriter(
                    root,
                    source_contract=self._source_contract(),
                    fps=10.0,
                    episodes_per_chunk=2,
                    video_crf=18,
                    video_preset="ultrafast",
                )
            staging.write_bytes(original)

            # A restart before chunk publication keeps the staged episode and
            # must not rescan/re-encode it as a new episode.
            resumed = MoWECanonicalArchiveWriter(
                root,
                source_contract=self._source_contract(),
                fps=10.0,
                episodes_per_chunk=2,
                video_crf=18,
                video_preset="ultrafast",
            )
            self.assertTrue(resumed.has_episode("suite:episode-0"))
            self.assertEqual(
                resumed.source_episode_identities(),
                {("/data/source.hdf5", 0)},
            )
            self.assertFalse(self._add(resumed, 0))
            self.assertTrue(self._add(resumed, 1))
            manifest = resumed.finalize()

            self.assertEqual(manifest["episode_count"], 2)
            self.assertEqual(manifest["frame_count"], 6)
            self.assertEqual(manifest["task_count"], 2)
            self.assertEqual(manifest["chunk_count"], 1)
            self.assertEqual(manifest["video_shape"], [16, 16, 3])
            self.assertEqual(
                manifest["video_shapes"],
                {"primary": [16, 16, 3], "wrist": [16, 16, 3]},
            )
            self.assertEqual(
                manifest["chunks"][0]["episodes"][1]["global_frame_start"], 3
            )
            for name in ("primary_video", "wrist_video"):
                video = root / manifest["chunks"][0]["files"][name]["path"]
                self.assertGreater(video.stat().st_size, 0)
                self.assertEqual(video.suffix, ".mp4")

            report = audit_canonical_archive(root, verify_checksums=True)
            self.assertTrue(report["passed"], report)
            loaded = load_canonical_archive_manifest(root)
            self.assertEqual(loaded["frame_count"], 6)
            stats = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["action_count"], 6)
            self.assertEqual(stats["skill_counts"], {"-1": 2, "0": 2, "1": 2})
            self.assertFalse(any(path.name.endswith(".tmp") for path in root.rglob("*")))

            completed = MoWECanonicalArchiveWriter(
                root,
                source_contract=self._source_contract(),
                fps=10.0,
                episodes_per_chunk=2,
                video_crf=18,
                video_preset="ultrafast",
            )
            self.assertTrue(completed.has_episode("suite:episode-0"))
            self.assertEqual(len(completed.pending), 0)
            with self.assertRaisesRegex(ValueError, "different contract"):
                MoWECanonicalArchiveWriter(
                    root,
                    source_contract={"rlds_manifest_fingerprint": "other"},
                    fps=10.0,
                    episodes_per_chunk=2,
                    video_crf=18,
                    video_preset="ultrafast",
                )

    def test_heterogeneous_calvin_camera_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "calvin-canonical"
            writer = MoWECanonicalArchiveWriter(
                root,
                source_contract={"dataset_fingerprint": "calvin-test"},
                fps=30.0,
                episodes_per_chunk=1,
                video_preset="ultrafast",
            )
            primary, wrist, actions, skills, proprio = self._episode(
                0,
                length=2,
                primary_shape=(200, 200),
                wrist_shape=(84, 84),
            )
            writer.add_episode(
                episode_id="calvin:0",
                dataset_name="calvin_abc_language_segments",
                partition="train",
                language="open the drawer",
                actions=actions,
                skills=skills,
                primary_frames=primary,
                wrist_frames=wrist,
                proprio=proprio,
            )
            manifest = writer.finalize()
            self.assertIsNone(manifest["video_shape"])
            self.assertEqual(
                manifest["video_shapes"],
                {"primary": [200, 200, 3], "wrist": [84, 84, 3]},
            )
            self.assertEqual(manifest["chunks"][0]["video_shapes"], manifest["video_shapes"])
            self.assertTrue(audit_canonical_archive(root, verify_checksums=True)["passed"])

            resumed = MoWECanonicalArchiveWriter(
                root,
                source_contract={"dataset_fingerprint": "calvin-test"},
                fps=30.0,
                episodes_per_chunk=1,
                video_preset="ultrafast",
            )
            self.assertEqual(resumed.video_shapes["primary"], (200, 200, 3))
            self.assertEqual(resumed.video_shapes["wrist"], (84, 84, 3))

    def test_module_import_does_not_eagerly_import_offline_dependencies(self):
        code = (
            "import json,sys; import mowe_wam.data; "
            "print(json.dumps({name: name in sys.modules for name in "
            "['pyarrow','imageio_ffmpeg','tensorflow']}))"
        )
        observed = json.loads(
            subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        )
        self.assertEqual(
            observed,
            {"pyarrow": False, "imageio_ffmpeg": False, "tensorflow": False},
        )

    def test_rejects_noncanonical_gripper(self):
        with tempfile.TemporaryDirectory() as directory:
            writer = MoWECanonicalArchiveWriter(
                directory,
                source_contract=self._source_contract(),
                fps=10.0,
                episodes_per_chunk=2,
            )
            primary, wrist, actions, skills, proprio = self._episode(0)
            actions[:, 6] = -1.0
            with self.assertRaisesRegex(ValueError, "binary 0/1"):
                writer.add_episode(
                    episode_id="bad",
                    dataset_name="suite",
                    partition="train",
                    language="bad",
                    actions=actions,
                    skills=skills,
                    primary_frames=primary,
                    wrist_frames=wrist,
                    proprio=proprio,
                )
            primary, wrist, actions, skills, proprio = self._episode(
                0, primary_shape=(15, 16)
            )
            with self.assertRaisesRegex(ValueError, "must be even"):
                writer.add_episode(
                    episode_id="odd-camera",
                    dataset_name="suite",
                    partition="train",
                    language="bad camera",
                    actions=actions,
                    skills=skills,
                    primary_frames=primary,
                    wrist_frames=wrist,
                    proprio=proprio,
                )


if __name__ == "__main__":
    unittest.main()
