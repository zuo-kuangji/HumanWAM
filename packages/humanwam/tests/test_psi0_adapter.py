import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from humanwam.backends.psi0 import Psi0Policy, validate_config
from humanwam.backends.psi0_commands import build_serve_command, build_train_command


class FakeMessage:
    def __init__(self, **fields):
        self.fields = fields

    def serialize(self):
        return self.fields


class FakeServer:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)
        self.Da, self.Tp, self.device = 80, 30, "test-device"
        self.launch_cfg = SimpleNamespace(
            data=SimpleNamespace(
                transform=SimpleNamespace(
                    repack=SimpleNamespace(image_keys=["observation.images.egocentric"])
                )
            )
        )
        self._pending_init_prev = np.ones(80)
        self.parsed = []
        self.postprocessed = []

    def _parse_obs_payload(self, message):
        self.parsed.append(message)
        return dict(
            imgs=[[message["image"]]],
            obs=np.zeros((1, 1, 45), dtype=np.float32),
            text_instructions=[message["instruction"].lower()],
            pooled_projections="cached",
        )

    def _postprocess_action(self, action):
        self.postprocessed.append(action)
        return action + 2.0

    def run(self, *args, **kwargs):
        raise AssertionError(
            "Offline adapter must never bind a native server or start a robot controller"
        )


class Psi0Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.run = self.root / "run"
        (self.run / "checkpoints/ckpt_40000").mkdir(parents=True)
        (self.run / "argv.txt").write_text("finetune_sonic_psi0_config")
        (self.run / "run_config.json").write_text(
            json.dumps(dict(model=dict(action_dim=80, action_chunk_size=30, odim=45)))
        )
        self.config = dict(run_dir=str(self.run), ckpt_step=40000, raw_state_dim=43)
        self.observation = dict(
            image_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
            state_raw=np.arange(43, dtype=np.float32),
            instruction="Put cup on table",
        )
        FakeServer.calls = []

    def policy(self, inference=None, server=FakeServer):
        return Psi0Policy(
            self.config,
            server_factory=server,
            message_factory=FakeMessage,
            inference=inference or (lambda server, prepared, steps: np.ones((1, 30, 80))),
        )

    def source(self):
        root = self.root / "source"
        for path in [
            "src/psi/models/psi0.py",
            "src/psi/deploy/serve_psi0_sonic.py",
            "scripts/train.py",
            "src/psi/config/train/finetune_sonic_psi0_config.py",
            ".env",
        ]:
            file = root / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text("# native entry fixture\n")
        return root

    def test_lazy_load_full_chunk_and_native_normalization_once(self):
        policy = self.policy()
        self.assertEqual(FakeServer.calls, [])
        result = policy.predict(self.observation)
        self.assertEqual(len(FakeServer.calls), 1)
        self.assertFalse(FakeServer.calls[0]["enable_rtc"])
        self.assertFalse(FakeServer.calls[0]["dashboard"])
        self.assertEqual(result["actions"].shape, (30, 80))
        self.assertTrue(np.all(result["actions"] == 3.0))
        self.assertFalse(result["normalized"])
        self.assertEqual(result["action_space"], "psi0_sonic_native")
        np.testing.assert_array_equal(
            policy._server.parsed[0]["state"]["states"], self.observation["state_raw"]
        )
        policy.predict(self.observation)
        self.assertEqual(len(FakeServer.calls), 1)
        self.assertEqual(len(policy._server.postprocessed), 2)

    def test_raw_shape_future_and_wrong_units_rejected_before_model_load(self):
        bad = dict(self.observation, state_raw=np.zeros(44))
        for observation in [
            bad,
            dict(self.observation, past_actions=np.zeros((30, 80))),
            dict(self.observation, future_state=np.zeros(43)),
        ]:
            with self.subTest(keys=list(observation)):
                with self.assertRaises(ValueError):
                    self.policy().predict(observation)
        self.assertEqual(FakeServer.calls, [])

    def test_nonfinite_state_bad_rgb_and_empty_instruction_rejected(self):
        for observation in [
            dict(self.observation, state_raw=np.full(43, np.nan)),
            dict(self.observation, image_rgb=np.zeros((3, 24, 32))),
            dict(self.observation, instruction=" "),
        ]:
            with self.assertRaises(ValueError):
                self.policy().predict(observation)

    def test_mismatched_checkpoint_horizon_and_feedback76_rejected(self):
        for delta in [
            dict(horizon=16),
            dict(action_dim=76),
            dict(action_schema="sonic_feedback76"),
            dict(ckpt_step="latest"),
            dict(ckpt_step=True),
        ]:
            with self.subTest(delta=delta):
                with self.assertRaises(ValueError):
                    validate_config(dict(self.config, **delta))

    def test_missing_checkpoint_companions_rejected(self):
        (self.run / "argv.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            self.policy()

    def test_real_release_metadata_keeps_neck_state_channels(self):
        fixture = Path(__file__).parent / "fixtures/psi0_sonic80_metadata.json"
        released = json.loads(fixture.read_text())
        (self.run / "run_config.json").write_text(json.dumps(released))
        actual = validate_config(dict(self.config, raw_state_dim=45))
        self.assertEqual(
            (actual["action_dim"], actual["horizon"], actual["model_state_dim"]), (80, 30, 45)
        )
        with self.assertRaisesRegex(ValueError, "measured inputs"):
            validate_config(dict(self.config, raw_state_dim=43))

    def test_native_output_shape_and_finite_checks(self):
        for invalid in [np.ones((1, 16, 76)), np.full((1, 30, 80), np.nan)]:
            with self.assertRaises(ValueError):
                self.policy(inference=lambda *args: invalid).predict(self.observation)

    def test_multi_camera_native_runtime_rejected(self):
        class Multi(FakeServer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.launch_cfg.data.transform.repack.image_keys.append("wrist")

        with self.assertRaises(ValueError):
            self.policy(server=Multi).predict(self.observation)

    def test_reset_keeps_loaded_model_clears_pending_rtc_seed(self):
        policy = self.policy()
        policy.predict(self.observation)
        native = policy._server
        policy.reset()
        self.assertIs(policy._server, native)
        self.assertIsNone(native._pending_init_prev)

    def test_native_serve_argv_is_rtc_websocket_not_robot_process(self):
        plan = build_serve_command(
            dict(self.config, python="/env/bin/python", port=8014), self.source()
        )
        self.assertEqual(plan["argv"][0], "/env/bin/python")
        self.assertIn("psi.deploy.serve_psi0_sonic", plan["argv"])
        self.assertIn("--rtc", plan["argv"])
        self.assertIn("40000", plan["argv"])
        self.assertNotIn("shell", plan)
        self.assertNotIn("robot", " ".join(plan["argv"]))

    def test_native_training_explicit_json_flags_and_no_auto_commit(self):
        source = self.source()
        args = self.root / "args.json"
        args.write_text(
            json.dumps(["--exp", "cpu-contract", "--train.max_training_steps", "40000"])
        )
        plan = build_train_command(
            dict(native_args_file=str(args), python="/env/bin/python", num_processes=8),
            source,
        )
        self.assertEqual(plan["argv"][0], "/env/bin/python")
        self.assertIn("torch.distributed.run", plan["argv"])
        self.assertIn("finetune_sonic_psi0_config", plan["argv"])
        args.write_text(json.dumps(["--auto-tag-run"]))
        with self.assertRaises(ValueError):
            build_train_command(dict(native_args_file=str(args)), source)
        args.write_text(json.dumps(["--exp", "cpu-contract"]))
        (source / ".env").unlink()
        with self.assertRaises(FileNotFoundError):
            build_train_command(dict(native_args_file=str(args)), source)


if __name__ == "__main__":
    unittest.main()
