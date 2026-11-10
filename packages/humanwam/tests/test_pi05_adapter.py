"""CPU boundaries only. These do not claim a full model or robot rollout passed."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from humanwam.backends.pi05 import Pi05Policy, pad_executed_history
from humanwam.backends.pi05_commands import build_serve_command, build_train_command


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / "params").mkdir()
        (self.root / "assets").mkdir()
        self.config = dict(
            method="delta_flow",
            config_name="pi05_libero_a2a_jax_lora_warmup_cosine",
            checkpoint_dir=str(self.root),
            device="cuda:0",
        )
        self.obs = {
            "observation/image": np.zeros((224, 224, 3), np.uint8),
            "observation/wrist_image": np.full((224, 224, 3), 27, np.uint8),
            "observation/state": np.arange(8, dtype=np.float32),
            "prompt": "put the bowl on the plate",
            "past_actions": np.full((10, 7), 8.25, np.float32),
            "past_action_valid_mask": np.ones(10, dtype=bool),
        }
        self.native = mock.Mock()
        self.native.infer.return_value = dict(
            actions=np.ones((10, 7), np.float32), policy_timing={"infer_ms": 1.0}
        )
        self.factory = mock.Mock(return_value=self.native)
        self.training = SimpleNamespace(
            model=SimpleNamespace(pi05=True, a2a_enabled=True, action_horizon=10, action_dim=32)
        )
        modules = {
            "openpi.training.config": SimpleNamespace(get_config=lambda name: self.training),
            "openpi.policies.policy_config": SimpleNamespace(
                create_trained_policy=self.factory, input_norm_stats_for_model=lambda: None
            ),
        }
        self.modules = mock.patch(
            "humanwam.backends.pi05.importlib.import_module", side_effect=modules.__getitem__
        )

    def test_lazy_load_uses_native_factory_and_raw_history_once(self):
        policy = Pi05Policy(self.config)
        with self.modules:
            first = policy.predict(self.obs)
            policy.predict(self.obs)
        self.factory.assert_called_once()
        self.assertEqual(self.factory.call_args.kwargs["sample_kwargs"], {"num_steps": 1})
        np.testing.assert_array_equal(
            self.native.infer.call_args.args[0]["past_actions"], self.obs["past_actions"]
        )
        self.assertEqual(first["actions"].shape, (10, 7))

    def test_no_inplace_raw_input_mutation(self):
        def infer(obs):
            obs["past_actions"][:] = 999
            return dict(actions=np.ones((10, 7), np.float32))

        self.native.infer.side_effect = infer
        with self.modules:
            Pi05Policy(self.config).predict(self.obs)
        self.assertTrue(np.all(self.obs["past_actions"] == 8.25))

    def test_baseline_uses_ten_steps_and_rejects_history(self):
        c = dict(self.config, method="gaussian", config_name="pi05_libero")
        p = Pi05Policy(c)
        with self.assertRaises(ValueError):
            p.predict(self.obs)
        obs = {
            k: v for k, v in self.obs.items() if k not in {"past_actions", "past_action_valid_mask"}
        }
        self.training.model.a2a_enabled = False
        with self.modules:
            p.predict(obs)
        self.assertEqual(self.factory.call_args.kwargs["sample_kwargs"], {"num_steps": 10})

    def test_delta_rejects_mixed_torch_and_jax_release_before_native_import(self):
        marker = self.root / "model.safetensors"
        marker.write_bytes(b"not-loaded")
        (self.root / "src/openpi/training").mkdir(parents=True)
        (self.root / "src/openpi/training/config.py").touch()
        with self.modules as importer:
            with self.assertRaisesRegex(ValueError, "JAX-only"):
                Pi05Policy(self.config)
            with self.assertRaisesRegex(ValueError, "JAX-only"):
                build_serve_command(dict(self.config, source_dir=str(self.root)))
        importer.assert_not_called()
        self.factory.assert_not_called()
        self.assertEqual(marker.read_bytes(), b"not-loaded")

    def test_delta_params_must_be_a_directory(self):
        (self.root / "params").rmdir()
        (self.root / "params").write_bytes(b"not-a-params-directory")
        with self.assertRaisesRegex(ValueError, "JAX params directories"):
            Pi05Policy(self.config)

    def test_gaussian_native_checkpoint_format_selection_remains_available(self):
        (self.root / "params").rmdir()
        (self.root / "model.safetensors").write_bytes(b"native-format-marker")
        self.training.model.a2a_enabled = False
        observation = {key: value for key, value in self.obs.items()
                       if key not in {"past_actions", "past_action_valid_mask"}}
        with self.modules:
            result = Pi05Policy(dict(self.config, method="gaussian", config_name="pi05_libero")).predict(observation)
        self.assertEqual(result["actions"].shape, (10, 7))
        self.factory.assert_called_once()

    def test_reject_wrong_method_and_sonic_dimensions(self):
        for changes in [
            dict(schema="sonic76"),
            dict(action_horizon=16),
            dict(action_dim=76),
            dict(method="gaussian"),
            dict(num_steps=10),
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                Pi05Policy(dict(self.config, **changes))

    def test_reject_future_labels_invalid_shapes_and_no_history(self):
        cases = [
            dict(actions=np.zeros((10, 7))),
            dict(past_actions=np.zeros((16, 76))),
            dict(past_action_valid_mask=np.zeros(10, dtype=bool)),
            {"observation/state": np.full(8, np.nan)},
            {"observation/image": np.zeros((224, 224, 3), np.float32)},
        ]
        p = Pi05Policy(self.config)
        for change in cases:
            with self.subTest(change=list(change)), self.assertRaises(ValueError):
                p.predict(dict(self.obs, **change))
        self.factory.assert_not_called()

    def test_installed_config_method_mismatch_is_rejected(self):
        self.training.model.a2a_enabled = False
        with self.modules, self.assertRaises(ValueError):
            Pi05Policy(self.config).predict(self.obs)

    def test_native_invalid_output_rejected(self):
        self.native.infer.return_value = {"actions": np.ones((10, 76))}
        with self.modules, self.assertRaises(RuntimeError):
            Pi05Policy(self.config).predict(self.obs)

    def test_reset_delegates_after_first_load(self):
        p = Pi05Policy(self.config)
        p.reset()
        with self.modules:
            p.predict(self.obs)
        p.reset()
        self.native.reset.assert_called_once()

    def test_history_padding_matches_native_client(self):
        past, valid = pad_executed_history(np.arange(21).reshape(3, 7))
        self.assertEqual(past.shape, (10, 7))
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(past[:8], np.repeat(past[:1], 8, axis=0))
        with self.assertRaises(ValueError):
            pad_executed_history(np.zeros((0, 7)))

    def test_native_command_builders_preserve_paths_and_explicit_budgets(self):
        (self.root / "src/openpi/training").mkdir(parents=True)
        (self.root / "src/openpi/training/config.py").touch()
        c = dict(
            self.config,
            source_dir=str(self.root),
            exp_name="run with spaces",
            batch_size=384,
            num_train_steps=3560,
            python="/backend/env/python",
        )
        train = build_train_command(c, python="/env/python")
        self.assertEqual(
            train["argv"][:4], ["/env/python", "scripts/train.py", c["config_name"], "--exp-name"]
        )
        self.assertIn("run with spaces", train["argv"])
        serve = build_serve_command(dict(c, port=8041))
        self.assertEqual(serve["argv"][0], "/backend/env/python")
        self.assertIn("--policy.num-steps", serve["argv"])
        self.assertEqual(serve["argv"][-1], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
