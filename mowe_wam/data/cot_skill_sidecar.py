"""TensorFlow-side exact join for the training-only LIBERO CoT skill labels."""

from __future__ import annotations

import base64
import json
import hashlib
from copy import deepcopy
from pathlib import Path

from mowe_wam.data.expert_skill_labels import label_directive


COT_SKILL_MARKER = "<|mowe_skill_label|>"
SOURCE_EPISODE_MARKER = "<|mowe_source_episode|>"
SOURCE_PAYLOAD_SEPARATOR = "."
SOURCE_EPISODE_KEY_SEPARATOR = "\x1f"


def source_episode_key(file_key: str, trajectory_index: int) -> str:
    """Stable pre-frame-transform identity used by resumable converters."""

    return f"{str(file_key)}{SOURCE_EPISODE_KEY_SEPARATOR}{int(trajectory_index)}"


def tensorflow_source_episode_key(trajectory):
    """Build the same identity from one joined episodic TensorFlow record."""

    import tensorflow as tf

    if {"_mowe_source_file_key", "_mowe_source_traj_index"}.issubset(trajectory):
        file_key = tf.reshape(trajectory["_mowe_source_file_key"], [-1])[0]
        trajectory_index = tf.reshape(trajectory["_mowe_source_traj_index"], [-1])[0]
    elif "task" in trajectory and "language_instruction" in trajectory["task"]:
        text = tf.reshape(trajectory["task"]["language_instruction"], [-1])[0]
        payload = tf.strings.split(text, SOURCE_EPISODE_MARKER)[-1]
        parts = tf.strings.split(payload, SOURCE_PAYLOAD_SEPARATOR)
        file_key = tf.io.decode_base64(parts[0])
        trajectory_index = tf.strings.to_number(parts[1], out_type=tf.int64)
    else:
        raise KeyError(
            "Source episode filter requires transport fields or a marked language instruction."
        )
    return tf.strings.join(
        [
            file_key,
            tf.constant(SOURCE_EPISODE_KEY_SEPARATOR),
            tf.strings.as_string(trajectory_index),
        ]
    )


def exclude_source_episodes(dataset, excluded_keys):
    """Filter committed episodes before CPU-heavy frame transforms."""

    keys = sorted({str(value) for value in excluded_keys})
    if not keys:
        return dataset
    import tensorflow as tf

    initializer = tf.lookup.KeyValueTensorInitializer(
        tf.constant(keys, dtype=tf.string),
        tf.ones(len(keys), dtype=tf.int32),
    )
    table = tf.lookup.StaticHashTable(initializer, default_value=0)

    def keep(trajectory):
        return tf.equal(table.lookup(tensorflow_source_episode_key(trajectory)), 0)

    return dataset.filter(keep)


def shard_episodic_dataset(dataset, *, rank: int, world_size: int):
    """Shard post-overlay trajectories without changing their global indices."""

    rank = int(rank)
    world_size = int(world_size)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed dataset identity rank={rank}, world_size={world_size}.")
    return dataset if world_size == 1 else dataset.shard(world_size, rank)


def split_cot_skill_marker(value) -> tuple[str, int]:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if SOURCE_EPISODE_MARKER in text:
        text = text.rsplit(SOURCE_EPISODE_MARKER, 1)[0]
    if COT_SKILL_MARKER not in text:
        return text, -1
    instruction, encoded = text.rsplit(COT_SKILL_MARKER, 1)
    try:
        return instruction, int(encoded)
    except ValueError as exc:
        raise ValueError(f"Invalid CoT skill marker: {encoded!r}") from exc


def split_mowe_transport_markers(value) -> tuple[str, int, str | None, int | None]:
    """Remove training-only skill/source markers from one instruction."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    source_file = None
    trajectory_index = None
    if SOURCE_EPISODE_MARKER in text:
        text, payload = text.rsplit(SOURCE_EPISODE_MARKER, 1)
        try:
            encoded_file, encoded_index = payload.split(
                SOURCE_PAYLOAD_SEPARATOR, 1
            )
            padding = "=" * (-len(encoded_file) % 4)
            source_file = base64.urlsafe_b64decode(encoded_file + padding).decode(
                "utf-8"
            )
            trajectory_index = int(encoded_index)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"Invalid source episode marker: {payload!r}") from exc
    instruction, skill_label = split_cot_skill_marker(text)
    return instruction, skill_label, source_file, trajectory_index


class TensorFlowCotSkillOverlay:
    """Lookup labels before upstream standardization drops episode metadata."""

    def __init__(self, sidecar_path: str | Path) -> None:
        path = Path(sidecar_path)
        if not path.exists():
            raise FileNotFoundError(f"CoT skill sidecar not found: {path}")
        raw_bytes = path.read_bytes()
        payload = json.loads(raw_bytes)
        keys = []
        labels = []
        for key, annotation in payload.items():
            keys.append(str(key))
            labels.append(int(label_directive(str(annotation))[0]))
        if not keys:
            raise ValueError(f"CoT skill sidecar is empty: {path}")
        try:
            import tensorflow as tf
        except ModuleNotFoundError as exc:
            raise RuntimeError("TensorFlow is required for the exact RLDS skill-sidecar join.") from exc
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            # The upstream RLDS module applies the same CPU-only setting at
            # import time. Building rank-0 validation after the training
            # pipeline may repeat this call after TF is initialized; that is
            # safe only when no GPU remains visible.
            if tf.config.get_visible_devices("GPU"):
                raise RuntimeError(
                    "TensorFlow GPU visibility was initialized before the Flow-WAM CPU-only RLDS pipeline."
                ) from exc
        initializer = tf.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.string),
            tf.constant(labels, dtype=tf.int64),
        )
        self.table = tf.lookup.StaticHashTable(initializer, default_value=-1)
        self.metadata = {
            "format": "tensorflow_cot_skill_overlay_v1",
            "label_version": "cot_final_directive_leading_verb_v1",
            "fingerprint_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "records": len(keys),
            "assume_timestep_aligned": True,
            "alignment_verified": False,
            "join_key": "episode_metadata.file_path + dlimp._traj_index + timestep",
            "requires_deterministic_rlds_order": True,
            "required_num_parallel_reads": 16,
        }

    def wrap_standardizer(self, standardize_fn):
        import tensorflow as tf

        def standardize_with_overlay(trajectory):
            # dlimp broadcasts RLDS episode metadata into ``traj_metadata``
            # before the OpenVLA standardizer runs.  Raw TFDS trajectories use
            # the top-level ``episode_metadata`` form, so retain that fallback
            # for direct tests and for compatibility with other RLDS readers.
            if (
                "traj_metadata" in trajectory
                and "episode_metadata" in trajectory["traj_metadata"]
                and "file_path" in trajectory["traj_metadata"]["episode_metadata"]
            ):
                metadata = trajectory["traj_metadata"]["episode_metadata"]["file_path"]
            elif "episode_metadata" in trajectory and "file_path" in trajectory["episode_metadata"]:
                metadata = trajectory["episode_metadata"]["file_path"]
            else:
                raise KeyError(
                    "LIBERO CoT overlay requires episode_metadata.file_path, either directly or under "
                    "dlimp traj_metadata."
                )
            length = tf.shape(trajectory["action"])[0]
            steps = tf.strings.as_string(tf.range(length))
            # Raw TFDS provides one scalar path while dlimp broadcasts it to
            # one path per timestep.  broadcast_to handles both contracts and
            # fails loudly if an unexpected metadata length is encountered.
            prefix = tf.broadcast_to(tf.reshape(metadata, [-1]), [length])
            if "_traj_index" in trajectory:
                trajectory_ids = tf.strings.as_string(
                    tf.broadcast_to(tf.reshape(trajectory["_traj_index"], [-1]), [length])
                )
                indexed_keys = tf.strings.join(
                    [prefix, tf.repeat("_", length), trajectory_ids, tf.repeat("_", length), steps]
                )
                labels = self.table.lookup(indexed_keys)
            else:
                # Direct raw-TFDS tests do not carry dlimp's global trajectory
                # index.  Only trajectory zero can be addressed unambiguously.
                zero_keys = tf.strings.join([prefix, tf.repeat("_0_", length), steps])
                labels = self.table.lookup(zero_keys)
            marker = tf.strings.join(
                [tf.repeat(COT_SKILL_MARKER, length), tf.strings.as_string(labels)]
            )
            source_marker = tf.strings.join(
                [
                    tf.repeat(SOURCE_EPISODE_MARKER, length),
                    tf.io.encode_base64(prefix),
                    tf.repeat(SOURCE_PAYLOAD_SEPARATOR, length),
                    trajectory_ids
                    if "_traj_index" in trajectory
                    else tf.repeat("0", length),
                ]
            )
            copied = dict(trajectory)
            copied["language_instruction"] = tf.strings.join(
                [trajectory["language_instruction"], marker, source_marker]
            )
            standardized = dict(standardize_fn(copied))
            # Preserve the exact pre-frame-transform join identity for
            # feature-store/canonical provenance. These transport fields are
            # never passed to the model and do not alter sidecar labels.
            standardized["_mowe_source_file_key"] = prefix
            standardized["_mowe_source_traj_index"] = (
                tf.strings.to_number(trajectory_ids, out_type=tf.int64)
                if "_traj_index" in trajectory
                else tf.zeros([length], dtype=tf.int64)
            )
            return standardized

        return standardize_with_overlay


def make_sidecar_episodic_dataset(
    base_class,
    sidecar,
    *,
    distributed_rank: int = 0,
    distributed_world_size: int = 1,
    frame_num_parallel_calls: int | None = None,
):
    """Create an upstream-compatible episodic dataset without patching upstream."""

    rank = int(distributed_rank)
    world_size = int(distributed_world_size)
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed dataset identity rank={rank}, world_size={world_size}.")

    class SidecarEpisodicRLDSDataset(base_class):
        def __init__(self, *args, **kwargs):
            self._skill_overlay = (
                sidecar if isinstance(sidecar, TensorFlowCotSkillOverlay) else TensorFlowCotSkillOverlay(sidecar)
            )
            super().__init__(*args, **kwargs)

        def make_dataset(self, rlds_config):
            from prismatic.vla.datasets.rlds.dataset import (
                apply_frame_transforms,
                apply_trajectory_transforms,
                make_dataset_from_rlds,
            )

            per_dataset_kwargs = rlds_config["dataset_kwargs_list"]
            if len(per_dataset_kwargs) != 1:
                raise ValueError("Skill-sidecar episodic loader supports one dataset per instance.")
            dataset_kwargs = deepcopy(per_dataset_kwargs[0])
            # The sidecar's middle key component is the deterministic global
            # RLDS trajectory index.  File-order shuffling would renumber
            # dlimp._traj_index and silently attach labels to the wrong demo.
            # Window-level shuffling may be applied after this exact join.
            dataset_kwargs["shuffle"] = False
            # cot_file.json was generated from the deterministic 16-way TFDS
            # shard interleave used by OpenVLA-OFT.  AUTOTUNE is host-dependent
            # (for example it resolves to 4 on Apple Silicon), which renumbers
            # _traj_index after the first shard block and corrupts the join.
            dataset_kwargs["num_parallel_reads"] = 16
            dataset_kwargs["standardize_fn"] = self._skill_overlay.wrap_standardizer(
                dataset_kwargs["standardize_fn"]
            )
            # Preserve every raw episode timestep.  Flow-WAM constructs its
            # eight-action chunk locally together with the H=8 future-image
            # target; asking dlimp to pre-chunk actions would trim seven steps
            # before the local horizon boundary and double-truncate episodes.
            traj_transform_kwargs = deepcopy(rlds_config["traj_transform_kwargs"])
            traj_transform_kwargs["future_action_window_size"] = 0
            frame_transform_kwargs = deepcopy(rlds_config["frame_transform_kwargs"])
            if frame_num_parallel_calls is not None:
                frame_transform_kwargs["num_parallel_calls"] = int(frame_num_parallel_calls)

            # `_traj_index` and the skill overlay are created by
            # make_dataset_from_rlds.  Shard only after that exact join, but
            # before CPU-heavy image decoding/resizing, so ranks neither
            # renumber sidecar keys nor decode each other's episodes.
            dataset, statistics = make_dataset_from_rlds(
                **dataset_kwargs,
                train=rlds_config["train"],
            )
            dataset = apply_trajectory_transforms(
                dataset,
                **traj_transform_kwargs,
                train=rlds_config["train"],
            )
            if world_size > 1:
                dataset = shard_episodic_dataset(dataset, rank=rank, world_size=world_size)
            # Retain the joined, sharded, pre-frame dataset so an offline
            # converter can install a completed-episode filter after it loads
            # writer resume state but before the first image is decoded.
            self._mowe_pre_frame_dataset = dataset
            self._mowe_frame_transform_kwargs = frame_transform_kwargs
            self._mowe_frame_transform_train = rlds_config["train"]
            framed = apply_frame_transforms(
                dataset,
                **frame_transform_kwargs,
                train=rlds_config["train"],
            )
            return framed.with_ram_budget(1), statistics["num_trajectories"], statistics

        def set_excluded_source_episode_keys(self, excluded_keys) -> None:
            from prismatic.vla.datasets.rlds.dataset import apply_frame_transforms

            filtered = exclude_source_episodes(
                self._mowe_pre_frame_dataset, excluded_keys
            )
            framed = apply_frame_transforms(
                filtered,
                **self._mowe_frame_transform_kwargs,
                train=self._mowe_frame_transform_train,
            )
            self.dataset = framed.with_ram_budget(1)

    return SidecarEpisodicRLDSDataset
