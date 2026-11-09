"""CPU contract tests with injected native backend; not GPU numerical tests."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from humanwam.backends import dit4dit as module
from humanwam.backends import dit4dit_commands as commands


class Model:
    def __init__(self):
        self.config = SimpleNamespace(
            framework=SimpleNamespace(
                action_model=SimpleNamespace(action_dim=32, state_dim=64, action_horizon=16)
            ),
            datasets=SimpleNamespace(vla_data=SimpleNamespace(image_size=[224, 224])),
        )
        self.seen = None

    def predict_action(self, *, examples):
        self.seen = examples
        return {"normalized_actions": np.full((1, 16, 32), 0.25, dtype=np.float32)}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        run = Path(self.tmp.name)
        (run / "checkpoints").mkdir()
        self.checkpoint = run / "checkpoints/step_1.pt"
        self.checkpoint.write_bytes(b"fixture-not-model")
        for filename in ["config.yaml", "dataset_statistics.json"]:
            (run / filename).write_text("{}")
        self.backend = Model()
        self.config = {"checkpoint": str(self.checkpoint)}
        self.policy = module.DiT4DiTPolicy(self.config, model_factory=lambda path: self.backend)
        self.obs = {
            "images": [np.zeros((480, 640, 3), dtype=np.uint8)],
            "prompt": "put the object down",
            "state_normalized": np.zeros(64),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_native_boundary_preserves_units_and_shapes(self):
        out = self.policy.predict(self.obs)
        self.assertEqual(np.asarray(out["actions"]).shape, (16, 32))
        self.assertTrue(out["normalized"])
        self.assertEqual(out["action_space"], "dit4dit_native_normalized")
        ex = self.backend.seen[0]
        self.assertEqual(ex["state"].shape, (1, 64))
        self.assertEqual(ex["image"][0].size, (224, 224))
        self.assertEqual(ex["lang"], self.obs["prompt"])

    def test_never_treat_second_camera_as_future_frame(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            self.policy.predict(dict(self.obs, images=self.obs["images"] * 2))

    def test_common_npz_offline_image_array(self):
        output = self.policy.predict(dict(self.obs, images=np.stack(self.obs["images"])))
        self.assertEqual(output["horizon"], 16)
        with self.assertRaisesRegex(ValueError, "NPZ images"):
            self.policy.predict(dict(self.obs, images=np.stack(self.obs["images"] * 2)))

    def test_raw_state_and_history_rejected(self):
        for field, value in [("state", np.zeros(44)), ("past_actions", np.zeros((16, 76)))]:
            with self.assertRaises(ValueError):
                self.policy.predict(dict(self.obs, **{field: value}))

    def test_sonic76_and_dimension_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "SONIC76"):
            module.DiT4DiTPolicy(
                dict(self.config, action_schema="g1_sonic76"),
                model_factory=lambda path: self.backend,
            )
        with self.assertRaisesRegex(ValueError, "action_dim"):
            module.DiT4DiTPolicy(
                dict(self.config, action_dim=76), model_factory=lambda path: self.backend
            )

    def test_nonfinite_and_wrong_native_output_rejected(self):
        with self.assertRaises(ValueError):
            self.policy.predict(dict(self.obs, state_normalized=np.full(64, np.nan)))
        self.backend.predict_action = lambda **kwargs: {"normalized_actions": np.zeros((1, 40, 32))}
        with self.assertRaisesRegex(ValueError, "output shape"):
            self.policy.predict(self.obs)

    def test_missing_checkpoint_metadata_rejected(self):
        (self.checkpoint.parents[1] / "dataset_statistics.json").unlink()
        with self.assertRaises(FileNotFoundError):
            module.DiT4DiTPolicy(self.config, model_factory=lambda path: self.backend)

    def test_native_commands_do_not_use_shell_or_change_working_directory(self):
        root = Path(self.tmp.name) / "upstream"
        for name in [
            "DiT4DiT/training/train.py",
            "DiT4DiT/config/deepseeds/deepspeed_zero2.yaml",
            "DiT4DiT/config/real_robot/dit4dit_g1.yaml",
        ]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        native_config = root / "DiT4DiT/config/real_robot/dit4dit_g1.yaml"
        cmd = commands.train_command(
            {
                "upstream_root": str(root),
                "native_config": str(native_config),
                "overrides": {
                    "datasets.vla_data.data_root_dir": "/space in path/data",
                    "trainer.enable_mixed_precision_training": True,
                },
            }
        )
        self.assertEqual(cmd["cwd"], str(root.resolve()))
        self.assertIn("/space in path/data", cmd["argv"])
        self.assertIn("--config_yaml", cmd["argv"])
        serve = commands.serve_command(
            {"upstream_root": str(root), "checkpoint": str(self.checkpoint)}
        )
        self.assertIn("deployment.model_server.server_policy", serve["argv"])
        self.assertIn("--ckpt_path", serve["argv"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
