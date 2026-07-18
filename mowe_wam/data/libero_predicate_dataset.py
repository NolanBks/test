"""LIBERO-style predicate dataset adapter with mock and RLDS modes."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from mowe_wam.memory import build_memory_snapshots
from mowe_wam.predicates.labeler import build_mock_trajectory, label_trajectory
from mowe_wam.predicates.schema import PREDICATE_NAMES
from mowe_wam.predicates.schema import predicate_dict_to_vector

try:
    import torch

    _DatasetBase = torch.utils.data.IterableDataset
except ModuleNotFoundError:
    torch = None
    _DatasetBase = object

LIBERO_RLDS_DATASETS = {
    "libero_spatial_no_noops",
    "libero_object_no_noops",
    "libero_goal_no_noops",
    "libero_10_no_noops",
    "libero_4_task_suites_no_noops",
}


def _as_float_list(value: Any) -> list[float]:
    """Convert numpy/tensor/list values into a JSON-label-compatible list."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    return [float(x) for x in value]


def _phase_target(predicate: list[float]) -> int:
    """Weak expert phase label derived from one predicate vector."""

    values = {name: float(predicate[idx]) for idx, name in enumerate(PREDICATE_NAMES)}
    if values["needs_recovery"] >= 0.5 or values["failure_risk"] >= 0.7:
        return 4
    if values["near_goal_region"] >= 0.7 or values["alignment_required"] >= 0.5:
        return 3
    if values["object_grasped"] >= 0.5 and values["object_moving_with_gripper"] >= 0.3:
        return 2
    if values["contact_likely"] >= 0.5:
        return 1
    return 0


class TransitionLabelStore:
    """Read offline future/event labels keyed by episode and timestep.

    Supported JSONL formats are one episode per row:
    ``{"episode_id": "...", "steps": [{...}, ...]}``, or one timestep per
    row with ``episode_id`` and ``step_id``. The store intentionally rejects
    unkeyed labels in real predictive mode.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Transition label cache not found: {self.path}")
        self.episodes: dict[str, list[dict[str, Any]]] = {}
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            episode_id = str(row["episode_id"])
            if "steps" in row:
                self.episodes[episode_id] = list(row["steps"])
            else:
                step_id = int(row["step_id"])
                steps = self.episodes.setdefault(episode_id, [])
                while len(steps) <= step_id:
                    steps.append({})
                steps[step_id] = row

    def episode(self, episode_id: str) -> list[dict[str, Any]]:
        try:
            return self.episodes[episode_id]
        except KeyError as exc:
            raise KeyError(
                f"No transition labels for episode {episode_id!r}. "
                "Build labels with scripts/build_transition_labels.py before predictive training."
            ) from exc


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_openvla_path(openvla_root: str | Path) -> Path:
    root = Path(openvla_root)
    if not root.is_absolute():
        root = _repo_root() / root
    if not root.exists():
        raise FileNotFoundError(f"OpenVLA-OFT checkout not found: {root}")
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def _decode_language(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode().lower()
    return str(value).lower()


def _episode_id_from_rlds_batch(rlds_batch: dict[str, Any], fallback: str | None = None) -> str | None:
    """Best-effort extraction of an upstream episode identifier without mutation."""

    for key in ("episode_id", "trajectory_id", "traj_id"):
        if key in rlds_batch:
            return _decode_language(rlds_batch[key])
    metadata = rlds_batch.get("traj_metadata")
    if isinstance(metadata, dict):
        for key in ("episode_id", "trajectory_id", "file_path"):
            if key in metadata:
                return _decode_language(metadata[key])
    return fallback


def fallback_predicates_from_action(action: Any) -> dict[str, float]:
    """Create conservative trainable predicates when simulator state is absent."""

    import numpy as np

    arr = np.asarray(action, dtype=np.float32)
    current = arr[0] if arr.ndim > 1 else arr
    motion_norm = float(np.linalg.norm(current[:6])) if current.size >= 6 else float(np.linalg.norm(current))
    gripper = float(current[-1]) if current.size else 0.0
    motion_score = float(np.clip(motion_norm / 0.05, 0.0, 1.0))
    closing_score = float(np.clip((gripper + 1.0) / 2.0, 0.0, 1.0))
    values = {name: 0.0 for name in PREDICATE_NAMES}
    values.update(
        {
            "contact_likely": closing_score,
            "object_moving_with_gripper": motion_score * closing_score,
            "progress_score": motion_score,
            "failure_risk": 0.0,
            "needs_recovery": 0.0,
        }
    )
    return values


class MoWERLDSBatchTransform:
    """Convert one upstream RLDS sample into the OpenVLA + MoWE training contract."""

    def __init__(
        self,
        action_tokenizer,
        base_tokenizer,
        image_transform,
        prompt_builder_fn,
        predict_stop_token: bool = True,
        use_wrist_image: bool = False,
        use_proprio: bool = False,
    ) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.predict_stop_token = predict_stop_token
        self.use_wrist_image = use_wrist_image
        self.use_proprio = use_proprio

    def __call__(self, rlds_batch: dict[str, Any]) -> dict[str, Any]:
        import numpy as np
        from PIL import Image
        from prismatic.vla.constants import IGNORE_INDEX

        dataset_name = rlds_batch["dataset_name"]
        actions = rlds_batch["action"]
        current_action = actions[0]
        image = Image.fromarray(rlds_batch["observation"]["image_primary"][0])
        language = _decode_language(rlds_batch["task"]["language_instruction"])

        prompt_builder = self.prompt_builder_fn("openvla")
        future_actions = actions[1:]
        action_chunk_string = self.action_tokenizer(current_action) + "".join(self.action_tokenizer(future_actions))
        action_chunk_len = len(action_chunk_string)
        conversation = [
            {"from": "human", "value": f"What action should the robot take to {language}?"},
            {"from": "gpt", "value": action_chunk_string},
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        labels = list(input_ids)
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        pixel_values = self.image_transform(image)
        labels[: -(action_chunk_len + 1)] = IGNORE_INDEX
        if not self.predict_stop_token:
            labels[-1] = IGNORE_INDEX

        predicate_dict = fallback_predicates_from_action(actions)
        predicates = np.asarray(predicate_dict_to_vector(predicate_dict), dtype=np.float32)

        out = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            "dataset_name": dataset_name,
            "actions": actions,
            "language": language,
            "predicates": predicates,
            "progress": np.asarray([predicate_dict["progress_score"]], dtype=np.float32),
            "risk": np.asarray([predicate_dict["failure_risk"]], dtype=np.float32),
        }
        episode_id = _episode_id_from_rlds_batch(rlds_batch)
        if episode_id is not None:
            out["episode_id"] = episode_id
        if self.use_wrist_image:
            wrist_pixels = []
            for key, value in rlds_batch["observation"].items():
                if "wrist" in key:
                    wrist_pixels.append(self.image_transform(Image.fromarray(value[0])))
            if wrist_pixels:
                out["pixel_values_wrist"] = torch.cat(wrist_pixels, dim=0)
        if self.use_proprio and "proprio" in rlds_batch["observation"]:
            out["proprio"] = rlds_batch["observation"]["proprio"]
        return out


class MoWEPaddedCollator:
    """Pad OpenVLA tensors and keep MoWE supervision tensors in the batch."""

    def __init__(self, processor) -> None:
        from prismatic.util.data_utils import PaddedCollatorForActionPrediction

        self.base_collator = PaddedCollatorForActionPrediction(
            processor.tokenizer.model_max_length,
            processor.tokenizer.pad_token_id,
            padding_side="right",
        )

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(instances)
        batch["language"] = [instance.get("language", "") for instance in instances]
        if "dataset_names" in batch:
            batch["dataset_name"] = batch["dataset_names"]
        batch["predicates"] = torch.stack([torch.as_tensor(instance["predicates"]).float() for instance in instances])
        batch["progress"] = torch.stack([torch.as_tensor(instance["progress"]).float() for instance in instances])
        batch["risk"] = torch.stack([torch.as_tensor(instance["risk"]).float() for instance in instances])
        if "proprio" in instances[0]:
            # Upstream's generic collator uses ``squeeze`` here, which drops
            # the batch axis when B=1.  MoWE keeps a stable [B, proprio_dim]
            # contract for the optional OpenVLA proprio projector.
            batch["proprio"] = torch.stack(
                [torch.as_tensor(instance["proprio"]).float().reshape(-1) for instance in instances]
            )
        predictive_tensor_fields = (
            "history_actions",
            "history_predicates",
            "current_predicates",
            "future_predicates",
            "progress_delta",
            "future_risk",
            "future_recovery",
            "memory_state",
            "event_target",
            "phase_target",
            "previous_expert",
        )
        for key in predictive_tensor_fields:
            if key in instances[0]:
                batch[key] = torch.stack([torch.as_tensor(instance[key]) for instance in instances])
        if "episode_id" in instances[0]:
            batch["episode_id"] = [str(instance["episode_id"]) for instance in instances]
        if "step_id" in instances[0]:
            batch["step_id"] = torch.tensor([int(instance["step_id"]) for instance in instances], dtype=torch.long)
        return batch


class LiberoPredicateDataset(_DatasetBase):
    """Dataset wrapper for local mock data or upstream LIBERO RLDS data."""

    def __init__(
        self,
        dataset_root: str = "MOCK",
        split: str = "train",
        predicate_label_path: str | None = None,
        limit: int | None = None,
        cfg: dict[str, Any] | None = None,
        processor: Any | None = None,
        resize_resolution: tuple[int, int] | None = None,
        openvla_root: str | Path = "external/openvla-oft",
        image_aug: bool = True,
        shuffle_buffer_size: int = 100_000,
    ) -> None:
        if torch is not None:
            super().__init__()
        self.dataset_root = dataset_root
        self.split = split
        self.predicate_label_path = predicate_label_path
        self.limit = limit
        self.cfg = cfg or {}
        self.mock = dataset_root.upper() == "MOCK"
        self.dataset_name = self.cfg.get("dataset_name", "libero_spatial_no_noops")
        self.rlds_dataset = None
        self.predictive = bool(self.cfg.get("predictive", False))
        self.history_steps = int(self.cfg.get("history_steps", 4))
        self.prediction_horizon = int(self.cfg.get("prediction_horizon", 8))
        self.transition_label_store = None
        if self.history_steps < 1 or self.prediction_horizon < 1:
            raise ValueError("history_steps and prediction_horizon must be positive.")

        if self.mock:
            self.samples = self._load_mock_samples(predicate_label_path)
            if limit is not None:
                self.samples = self.samples[:limit]
        else:
            root = Path(dataset_root)
            if dataset_root == "TBD" or not root.exists():
                raise FileNotFoundError(
                    f"Real LIBERO dataset root is not available: {dataset_root}. "
                    "Use dataset_root='MOCK' for local smoke tests."
                )
            if processor is None:
                raise ValueError("Real LIBERO RLDS mode requires an OpenVLA processor.")
            if self.dataset_name not in LIBERO_RLDS_DATASETS:
                known = ", ".join(sorted(LIBERO_RLDS_DATASETS))
                raise ValueError(f"Unsupported LIBERO RLDS dataset {self.dataset_name!r}. Known: {known}")
            _ensure_openvla_path(openvla_root)
            from prismatic.models.backbones.llm.prompting import PurePromptBuilder
            from prismatic.vla.action_tokenizer import ActionTokenizer
            from prismatic.vla.datasets import EpisodicRLDSDataset, RLDSDataset

            batch_transform = MoWERLDSBatchTransform(
                ActionTokenizer(processor.tokenizer),
                processor.tokenizer,
                image_transform=processor.image_processor.apply_transform,
                prompt_builder_fn=PurePromptBuilder,
                use_wrist_image=bool(self.cfg.get("num_images_in_input", 1) > 1),
                use_proprio=bool(self.cfg.get("use_proprio", False)),
            )
            dataset_cls = EpisodicRLDSDataset if self.predictive else RLDSDataset
            if self.predictive:
                label_path = self.cfg.get("transition_label_path")
                if not label_path:
                    raise ValueError(
                        "Predictive training requires data.transition_label_path with trajectory-level future/event labels. "
                        "Run scripts/build_transition_labels.py after inspecting real RLDS episodes."
                    )
                self.transition_label_store = TransitionLabelStore(label_path)
            self.rlds_dataset = dataset_cls(
                root,
                self.dataset_name,
                batch_transform,
                resize_resolution=resize_resolution or (224, 224),
                shuffle_buffer_size=shuffle_buffer_size,
                train=split == "train",
                image_aug=image_aug,
            )

    def _load_mock_samples(self, predicate_label_path: str | None) -> list[dict[str, Any]]:
        if predicate_label_path and Path(predicate_label_path).exists():
            rows = [json.loads(line) for line in Path(predicate_label_path).read_text().splitlines() if line.strip()]
            labels = [row.get("predicates", row) for row in rows]
        else:
            trajectory, task_meta = build_mock_trajectory()
            labels = label_trajectory(trajectory, task_meta=task_meta)

        if self.predictive:
            return self._build_predictive_samples(labels)

        samples = []
        for idx, label_dict in enumerate(labels):
            predicate_vec = predicate_dict_to_vector(label_dict)
            samples.append(
                {
                    "images": [[[0.0]]],
                    "language": "mock place the block into the goal region",
                    "proprio": [0.0] * 8,
                    "actions": [[0.0] * 7 for _ in range(8)],
                    "predicates": predicate_vec,
                    "progress": [float(label_dict["progress_score"])],
                    "risk": [float(label_dict["failure_risk"])],
                    "task_meta": {"task_name": "mock_place_object", "step_index": idx},
                }
            )
        return samples

    def _build_predictive_samples(self, labels: list[dict[str, float]]) -> list[dict[str, Any]]:
        """Build deterministic trajectory windows for mock and unit smoke checks."""

        predicate_vectors = [predicate_dict_to_vector(label) for label in labels]
        progress = [float(label["progress_score"]) for label in labels]
        risk = [float(label["failure_risk"]) for label in labels]
        phase_targets = [_phase_target(vector) for vector in predicate_vectors]
        snapshots, event_targets = build_memory_snapshots(predicate_vectors, progress, risk, phase_targets)
        action_dim = int(self.cfg.get("action_dim", 7))
        chunk_size = int(self.cfg.get("chunk_size", 8))
        zero_action = [0.0] * action_dim
        samples = []
        for idx, predicate_vec in enumerate(predicate_vectors):
            future_idx = min(idx + self.prediction_horizon, len(predicate_vectors) - 1)
            history_indices = [idx - self.history_steps + offset for offset in range(self.history_steps)]
            history_predicates = [predicate_vectors[item] if item >= 0 else [0.0] * len(PREDICATE_NAMES) for item in history_indices]
            history_actions = [zero_action for _ in history_indices]
            samples.append(
                {
                    "images": [[[0.0]]],
                    "language": "mock place the block into the goal region",
                    "proprio": [0.0] * 8,
                    "actions": [zero_action for _ in range(chunk_size)],
                    "predicates": predicate_vec,
                    "progress": [progress[idx]],
                    "risk": [risk[idx]],
                    "episode_id": "mock_episode_0",
                    "step_id": idx,
                    "history_actions": history_actions,
                    "history_predicates": history_predicates,
                    "current_predicates": predicate_vec,
                    "future_predicates": predicate_vectors[future_idx],
                    "progress_delta": [progress[future_idx] - progress[idx]],
                    "future_risk": [risk[future_idx]],
                    "future_recovery": [predicate_vectors[future_idx][PREDICATE_NAMES.index("needs_recovery")]],
                    "memory_state": snapshots[idx],
                    "event_target": event_targets[idx],
                    "phase_target": phase_targets[idx],
                    "previous_expert": phase_targets[idx - 1] if idx > 0 else -1,
                    "task_meta": {"task_name": "mock_place_object", "step_index": idx},
                }
            )
        return samples

    def __len__(self) -> int:
        if self.mock:
            return len(self.samples)
        if self.rlds_dataset is None:
            return 0
        return len(self.rlds_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if not self.mock:
            raise NotImplementedError("Real LIBERO RLDS mode is iterable; use DataLoader iteration.")
        return self.samples[idx]

    def __iter__(self):
        if self.mock:
            yield from self.samples
            return
        iterator = iter(self.rlds_dataset)
        if self.predictive:
            for episode_idx, episode in enumerate(iterator):
                yield from self._iter_predictive_episode(episode_idx, episode)
            return
        if self.limit is None:
            yield from iterator
            return
        for idx, sample in enumerate(iterator):
            if idx >= self.limit:
                break
            yield sample

    @staticmethod
    def _label_predicate_vector(step_label: dict[str, Any]) -> list[float]:
        values = step_label.get("predicates", step_label.get("current_predicates"))
        if values is None:
            raise KeyError("Transition label step must include 'predicates' or 'current_predicates'.")
        if isinstance(values, dict):
            return predicate_dict_to_vector(values)
        return _as_float_list(values)

    def _iter_predictive_episode(self, episode_idx: int, episode: list[dict[str, Any]]):
        if not episode:
            return
        assert self.transition_label_store is not None
        raw_episode_id = episode[0].get("episode_id")
        if raw_episode_id is None and bool(self.cfg.get("strict_episode_id", True)):
            raise RuntimeError(
                "Predictive RLDS mode requires a stable upstream episode_id (or traj_metadata episode_id) "
                "to align future labels. Export an episode identifier during label generation, or set "
                "strict_episode_id=false only for a deterministic preflight run."
            )
        episode_id = str(raw_episode_id or f"{self.dataset_name}:{episode_idx}")
        labels = self.transition_label_store.episode(episode_id)
        if len(labels) != len(episode):
            raise ValueError(
                f"Episode {episode_id!r} has {len(episode)} RLDS steps but {len(labels)} label steps. "
                "Rebuild the transition label cache with matching trajectory transforms."
            )
        predicate_vectors = [self._label_predicate_vector(item) for item in labels]
        if any(len(vector) != len(PREDICATE_NAMES) for vector in predicate_vectors):
            raise ValueError(f"Episode {episode_id!r} contains malformed predicate labels.")
        progress = [float(item.get("progress", vector[PREDICATE_NAMES.index("progress_score")])) for item, vector in zip(labels, predicate_vectors)]
        risk = [float(item.get("risk", vector[PREDICATE_NAMES.index("failure_risk")])) for item, vector in zip(labels, predicate_vectors)]
        phase_targets = [_phase_target(vector) for vector in predicate_vectors]
        snapshots, event_targets = build_memory_snapshots(predicate_vectors, progress, risk, phase_targets)
        last_valid = len(episode) - self.prediction_horizon
        for step_idx in range(max(0, last_valid)):
            future_idx = step_idx + self.prediction_horizon
            history_indices = [step_idx - self.history_steps + offset for offset in range(self.history_steps)]
            action_dim = len(_as_float_list(episode[step_idx]["actions"][0]))
            history_actions = [
                _as_float_list(episode[item]["actions"][0]) if item >= 0 else [0.0] * action_dim for item in history_indices
            ]
            out = dict(episode[step_idx])
            out.update(
                {
                    "episode_id": episode_id,
                    "step_id": step_idx,
                    "predicates": predicate_vectors[step_idx],
                    "progress": [progress[step_idx]],
                    "risk": [risk[step_idx]],
                    "history_actions": history_actions,
                    "history_predicates": [
                        predicate_vectors[item] if item >= 0 else [0.0] * len(PREDICATE_NAMES) for item in history_indices
                    ],
                    "current_predicates": predicate_vectors[step_idx],
                    "future_predicates": predicate_vectors[future_idx],
                    "progress_delta": [progress[future_idx] - progress[step_idx]],
                    "future_risk": [risk[future_idx]],
                    "future_recovery": [predicate_vectors[future_idx][PREDICATE_NAMES.index("needs_recovery")]],
                    "memory_state": snapshots[step_idx],
                    "event_target": event_targets[step_idx],
                    "phase_target": phase_targets[future_idx],
                    "previous_expert": phase_targets[step_idx - 1] if step_idx > 0 else -1,
                }
            )
            yield out


def infer_shape(value: Any) -> list[int]:
    """Infer nested list shape for readable smoke-test output."""

    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(x) for x in shape]
    dims = []
    current = value
    while isinstance(current, list):
        dims.append(len(current))
        current = current[0] if current else []
    return dims
