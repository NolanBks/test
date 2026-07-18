from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mowe_wam.backbones.openvla_identity import (
    openvla_identities_match,
    resolve_original_openvla_identity,
    validate_openvla_identity,
    validate_original_openvla_reference,
)


class OriginalOpenVLAIdentityTests(unittest.TestCase):
    revision = "0123456789abcdef0123456789abcdef01234567"

    @staticmethod
    def _snapshot(root: Path, *, weight: bytes = b"original-openvla") -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(
            json.dumps({"model_type": "openvla", "hidden_size": 4096}),
            encoding="utf-8",
        )
        (root / "preprocessor_config.json").write_text(
            json.dumps({"image_size": 224}), encoding="utf-8"
        )
        (root / "tokenizer_config.json").write_text(
            json.dumps({"model_max_length": 2048}), encoding="utf-8"
        )
        (root / "model.safetensors").write_bytes(weight)
        return root

    def test_snapshot_identity_is_stable_and_weight_sensitive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_path = self._snapshot(root / "first")
            copy_path = self._snapshot(root / "copy")
            changed_path = self._snapshot(root / "changed", weight=b"different")
            first = resolve_original_openvla_identity(
                first_path, revision=self.revision
            )
            copied = resolve_original_openvla_identity(
                copy_path, revision=self.revision
            )
            changed = resolve_original_openvla_identity(
                changed_path, revision=self.revision
            )
            validate_openvla_identity(first)
            self.assertTrue(openvla_identities_match(first, copied))
            self.assertFalse(openvla_identities_match(first, changed))

    def test_rejects_mutable_revision_and_libero_finetuned_reference(self):
        with self.assertRaisesRegex(ValueError, "40-character"):
            validate_original_openvla_reference(
                "openvla/openvla-7b", revision="main"
            )
        with self.assertRaisesRegex(ValueError, "forbidden"):
            validate_original_openvla_reference(
                "/models/openvla-7b-oft-libero-all", revision=self.revision
            )

    def test_adapter_disables_snapshot_remote_code_for_local_multi_image_loader(self):
        """The immutable base snapshot must not override the registered OFT loader."""

        from mowe_wam.backbones.openvla_oft_adapter import OpenVLAOFTAdapter

        with patch("mowe_wam.backbones.openvla_oft_adapter._ensure_openvla_path", return_value=Path(".")), \
             patch("mowe_wam.backbones.openvla_oft_adapter.validate_original_openvla_reference"), \
             patch("transformers.AutoConfig.register"), \
             patch("transformers.AutoImageProcessor.register"), \
             patch("transformers.AutoProcessor.register"), \
             patch("transformers.AutoModelForVision2Seq.register"), \
             patch("transformers.AutoProcessor.from_pretrained") as processor_from_pretrained, \
             patch("transformers.AutoModelForVision2Seq.from_pretrained") as model_from_pretrained:
            class Vision:
                def set_num_images_in_input(self, value):
                    self.value = value

            class Model:
                llm_dim = 8
                config = type("Config", (), {"image_sizes": (224, 224)})()
                vision_backbone = Vision()

                def to(self, device):
                    return self

                def eval(self):
                    return self

                def parameters(self):
                    return []

            model_from_pretrained.return_value = Model()
            adapter = OpenVLAOFTAdapter(
                checkpoint="/snapshot", revision=self.revision, device="cpu", dtype="float32", num_images_in_input=2
            )

        self.assertEqual(adapter.model.vision_backbone.value, 2)
        self.assertFalse(processor_from_pretrained.call_args.kwargs["trust_remote_code"])
        self.assertFalse(model_from_pretrained.call_args.kwargs["trust_remote_code"])
