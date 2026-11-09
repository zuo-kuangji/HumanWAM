import json
import tempfile
import unittest
from pathlib import Path

from humanwam.assets import prepare_gr00t_offline_view


class AssetViewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.checkpoint = self.root / "checkpoint"
        self.backbone = self.root / "backbone"
        self.checkpoint.mkdir()
        self.backbone.mkdir()
        (self.checkpoint / "config.json").write_text(
            json.dumps({"model_name": "nvidia/Cosmos-Reason2-2B", "action_horizon": 40})
        )
        (self.checkpoint / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_kwargs": {
                        "model_name": "nvidia/Cosmos-Reason2-2B",
                        "max_action_dim": 132,
                    }
                }
            )
        )
        (self.checkpoint / "model.safetensors").write_bytes(b"untouched test weights")
        (self.checkpoint / "statistics.json").write_text('{"state": [0.1]}')
        (self.backbone / "config.json").write_text('{"model_type": "qwen3_vl"}')
        self.output = self.root / "offline-view"

    def test_relocates_only_metadata_and_links_original_weights(self):
        before = {p.name: p.read_bytes() for p in self.checkpoint.iterdir()}
        report = prepare_gr00t_offline_view(self.checkpoint, self.backbone, self.output)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.checkpoint.iterdir()})
        alias = self.output / "assets/nvidia/Cosmos-Reason2-2B"
        self.assertEqual(alias.resolve(), self.backbone)
        self.assertTrue((self.output / "model.safetensors").is_symlink())
        self.assertEqual(
            (self.output / "model.safetensors").resolve(), self.checkpoint / "model.safetensors"
        )
        self.assertEqual(
            (self.output / "statistics.json").resolve(), self.checkpoint / "statistics.json"
        )
        self.assertFalse((self.output / "config.json").is_symlink())
        model = json.loads((self.output / "config.json").read_text())
        self.assertEqual(model, {"model_name": str(alias), "action_horizon": 40})
        self.assertEqual(
            report["metadata_only_asset_locator_overrides"]["config.json"]["field_path"],
            "model_name",
        )
        self.assertFalse(report["original_cache_modified"])
        self.assertFalse(report["model_weights_loaded"])

    def test_native_processor_subdirectory_kept_real_for_metadata_copy(self):
        processor = self.checkpoint / "processor"
        processor.mkdir()
        (self.checkpoint / "processor_config.json").rename(processor / "processor_config.json")
        (processor / "statistics.json").write_text('{"unchanged": true}')
        report = prepare_gr00t_offline_view(self.checkpoint, self.backbone, self.output)
        self.assertIn(
            "processor/processor_config.json", report["metadata_only_asset_locator_overrides"]
        )
        self.assertFalse((self.output / "processor/processor_config.json").is_symlink())
        self.assertTrue((self.output / "processor/statistics.json").is_symlink())

    def test_existing_output_and_output_in_model_cache_rejected(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            prepare_gr00t_offline_view(self.checkpoint, self.backbone, self.output)
        for parent in (self.checkpoint, self.backbone):
            with self.subTest(parent=parent):
                with self.assertRaises(ValueError):
                    prepare_gr00t_offline_view(self.checkpoint, self.backbone, parent / "view")
                self.assertFalse((parent / "view").exists())

    def test_mismatched_backbone_locator_rejected_without_output(self):
        path = self.checkpoint / "config.json"
        path.write_text('{"model_name": "another/model"}')
        original = path.read_bytes()
        with self.assertRaises(ValueError):
            prepare_gr00t_offline_view(self.checkpoint, self.backbone, self.output)
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(self.output.exists())

    def test_dangling_output_symlink_is_not_overwritten(self):
        self.output.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            prepare_gr00t_offline_view(self.checkpoint, self.backbone, self.output)
        self.assertTrue(self.output.is_symlink())


if __name__ == "__main__":
    unittest.main()
