import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from humanwam.backends.psi0 import Psi0Policy, validate_config
from humanwam.backends.psi0_commands import build_serve_command, build_train_command


def managed_values(argv, name):
    """Read both native --key=value and --key values without executing a command."""
    matches = []
    for index, token in enumerate(argv):
        flag, separator, inline = token.partition("=")
        if flag.replace("_", "-") != name:
            continue
        values = [inline] if separator else []
        for value in argv[index + 1 :]:
            if value.startswith("--"):
                break
            values.append(value)
        matches.append(values)
    return matches


class FakeMessage:
    def __init__(self, **fields):
        self.fields = fields

    def serialize(self):
        return self.fields


class FakeServer:
    calls = []
    action_width = 78
    state_width = 43

    def __init__(self, **kwargs):
        self.calls.append(kwargs)
        self.Da, self.Tp, self.device = self.action_width, 30, "test-device"
        self.launch_cfg = SimpleNamespace(
            model=SimpleNamespace(odim=self.state_width, action_dim=self.action_width),
            data=SimpleNamespace(
                transform=SimpleNamespace(
                    repack=SimpleNamespace(image_keys=["observation.images.egocentric"])
                )
            ),
        )
        self._pending_init_prev = np.ones(self.action_width)
        self.parsed = []
        self.postprocessed = []

    def _parse_obs_payload(self, message):
        self.parsed.append(message)
        return dict(
            imgs=[[message["image"]]],
            obs=np.zeros((1, 1, self.state_width), dtype=np.float32),
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
            json.dumps(dict(model=dict(action_dim=78, action_chunk_size=30, odim=43)))
        )
        self.config = dict(run_dir=str(self.run), ckpt_step=40000, raw_state_dim=43, action_dim=78)
        self.observation = dict(
            image_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
            state_raw=np.arange(43, dtype=np.float32),
            instruction="Put cup on table",
        )
        FakeServer.calls = []

    def policy(self, inference=None, server=FakeServer, config=None):
        return Psi0Policy(
            self.config if config is None else config,
            server_factory=server,
            message_factory=FakeMessage,
            inference=inference or (lambda server, prepared, steps: np.ones((1, 30, server.Da))),
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
        self.assertEqual(result["actions"].shape, (30, 78))
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
            dict(self.observation, past_actions=np.zeros((30, 78))),
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

    def test_real_release_metadata_needs_explicit_checkpoint_native_profile(self):
        fixture = Path(__file__).parent / "fixtures/psi0_sonic80_metadata.json"
        released = json.loads(fixture.read_text())
        (self.run / "run_config.json").write_text(json.dumps(released))
        with self.assertRaises(ValueError):
            validate_config(dict(self.config, raw_state_dim=45, action_dim=80))
        actual = validate_config(
            dict(
                self.config, raw_state_dim=45, action_dim=80, embodiment_profile="checkpoint_native"
            )
        )
        self.assertEqual(
            (actual["action_dim"], actual["horizon"], actual["model_state_dim"]), (80, 30, 45)
        )
        with self.assertRaisesRegex(ValueError, "measured inputs"):
            validate_config(
                dict(
                    self.config,
                    raw_state_dim=43,
                    action_dim=80,
                    embodiment_profile="checkpoint_native",
                )
            )

    def test_real_official_neckless_checkpoint_metadata_accepts_default_profile(self):
        fixture = Path(__file__).parent / "fixtures/psi0_neckless43_action78_metadata.json"
        released = json.loads(fixture.read_text())
        self.assertEqual(
            released["_source"]["run_config_sha256"],
            "075149ab6417a5a3bfab20d7b7d481a2c43eef7a4d908c7632bf52eac245e02e",
        )
        (self.run / "run_config.json").write_text(json.dumps(released))
        actual = validate_config(dict(run_dir=str(self.run), ckpt_step=40000))
        self.assertEqual(actual["embodiment_profile"], "g1_neckless43_action78")
        self.assertEqual(
            (
                actual["raw_state_dim"],
                actual["model_state_dim"],
                actual["action_dim"],
                actual["horizon"],
            ),
            (43, 43, 78, 30),
        )
        policy = self.policy()
        self.assertEqual(FakeServer.calls, [])
        self.assertEqual(policy.predict(self.observation)["actions"].shape, (30, 78))

    def test_real_neckless_metadata_drift_is_rejected_before_native_allocation(self):
        fixture = Path(__file__).parent / "fixtures/psi0_neckless43_action78_metadata.json"
        released = json.loads(fixture.read_text())
        cases = [
            ("model", "odim", 45),
            ("model", "action_dim", 80),
            ("repack", "action_keys", ["action.body_token", "action[:14]", "action.neck"]),
            ("repack", "state_keys", ["observation.state", "observation.neck"]),
            ("field", "stat_action_keys", ["action.neck"]),
            ("field", "stat_state_keys", ["observation.neck"]),
            ("field", "state_min", [-1.0] * 45),
            ("field", "state_max", [1.0] * 45),
            ("field", "action_min", [-1.0] * 80),
            ("field", "action_max", [1.0] * 80),
        ]
        for section, name, value in cases:
            with self.subTest(section=section, name=name):
                changed = deepcopy(released)
                target = (
                    changed["model"]
                    if section == "model"
                    else changed["data"]["transform"][section]
                )
                target[name] = value
                (self.run / "run_config.json").write_text(json.dumps(changed))
                with self.assertRaises(ValueError):
                    self.policy()
                self.assertEqual(FakeServer.calls, [])

    def test_native_output_shape_and_finite_checks(self):
        for invalid in [np.ones((1, 16, 76)), np.ones((1, 30, 80)), np.full((1, 30, 78), np.nan)]:
            with self.assertRaises(ValueError):
                self.policy(inference=lambda *args: invalid).predict(self.observation)

    def test_default_and_explicit_neckless_profiles_bind_checkpoint_dimensions(self):
        for profile in ({}, {"embodiment_profile": "g1_neckless43_action78"}):
            config = dict(self.config, **profile)
            validated = validate_config(config)
            self.assertEqual(
                (validated["raw_state_dim"], validated["model_state_dim"], validated["action_dim"]),
                (43, 43, 78),
            )
            for state_width, action_width in ((45, 80), (45, 78), (43, 80)):
                with self.subTest(profile=profile, state=state_width, action=action_width):
                    (self.run / "run_config.json").write_text(
                        json.dumps(
                            {
                                "model": {
                                    "odim": state_width,
                                    "action_dim": action_width,
                                    "action_chunk_size": 30,
                                }
                            }
                        )
                    )
                    with self.assertRaises(ValueError):
                        validate_config(config)
            (self.run / "run_config.json").write_text(
                json.dumps({"model": {"odim": 43, "action_dim": 78, "action_chunk_size": 30}})
            )
        with self.assertRaises(ValueError):
            validate_config(dict(self.config, embodiment_profile="guess_neckless"))

    def test_neckless_parse_may_not_reintroduce_hidden_45d_padding(self):
        class IncorrectPadding(FakeServer):
            def _parse_obs_payload(self, message):
                value = super()._parse_obs_payload(message)
                value["obs"] = np.zeros((1, 1, 45), dtype=np.float32)
                return value

        def never_infer(*args):
            raise AssertionError("Reject hidden neck padding before model inference")

        with self.assertRaises(ValueError):
            self.policy(server=IncorrectPadding, inference=never_infer).predict(self.observation)

    def test_explicit_native_profile_keeps_historical_full_width_runtime(self):
        class Native80(FakeServer):
            action_width, state_width = 80, 45

        (self.run / "run_config.json").write_text(
            json.dumps({"model": {"odim": 45, "action_dim": 80, "action_chunk_size": 30}})
        )
        config = dict(
            self.config, embodiment_profile="checkpoint_native", raw_state_dim=45, action_dim=80
        )
        observation = dict(self.observation, state_raw=np.arange(45, dtype=np.float32))
        result = self.policy(server=Native80, config=config).predict(observation)
        self.assertEqual(result["actions"].shape, (30, 80))
        self.assertEqual(result["native"]["raw_state_dim"], 45)

    def test_shipped_runtime_and_training_configs_explicitly_select_no_neck(self):
        root = Path(__file__).resolve().parents[3]
        for name in ("psi0-sonic.json", "psi0-train.json"):
            options = json.loads((root / "configs/humanwam" / name).read_text())["options"]
            self.assertEqual(options["embodiment_profile"], "g1_neckless43_action78")
            self.assertEqual((options["raw_state_dim"], options["action_dim"]), (43, 78))
            self.assertEqual(options["horizon"], 30)
            if name == "psi0-train.json":
                self.assertEqual(options["action_keys"], ["action.body_token_v1_1", "action[:14]"])
                self.assertEqual(options["state_keys"], ["observation.state"])

    def training_args(self, flags=None):
        path = self.root / "args.json"
        path.write_text(
            json.dumps(flags if flags is not None else ["--exp", "neckless-cpu-contract"])
        )
        return path

    def test_default_training_fills_all_neckless_dimensions_keys_and_horizon(self):
        args = self.training_args()
        before = args.read_text()
        plan = build_train_command(dict(native_args_file=str(args)), self.source())
        expected = {
            "--model.action-dim": ["78"],
            "--model.odim": ["43"],
            "--model.action-chunk-size": ["30"],
            "--model.action-exec-horizon": ["30"],
            "--data.transform.repack.action-chunk-size": ["30"],
            "--data.transform.repack.pad-action-dim": ["78"],
            "--data.transform.repack.pad-state-dim": ["43"],
            "--data.transform.field.pad-action-dim": ["78"],
            "--data.transform.field.pad-state-dim": ["43"],
            "--data.transform.repack.action-keys": ["action.body_token_v1_1", "action[:14]"],
            "--data.transform.field.stat-action-keys": ["action.body_token_v1_1", "action[:14]"],
            "--data.transform.repack.state-keys": ["observation.state"],
            "--data.transform.field.stat-state-keys": ["observation.state"],
        }
        for key, value in expected.items():
            with self.subTest(flag=key):
                self.assertEqual(managed_values(plan["argv"], key), [value])
        self.assertEqual(args.read_text(), before)

    def test_training_can_use_assembled78_action_and_an_explicit_chunk_length(self):
        args = self.training_args()
        plan = build_train_command(
            dict(
                native_args_file=str(args),
                action_keys=["action"],
                state_keys=["observation.state"],
                horizon=16,
            ),
            self.source(),
        )
        for flag in (
            "--data.transform.repack.action-keys",
            "--data.transform.field.stat-action-keys",
        ):
            self.assertEqual(managed_values(plan["argv"], flag), [["action"]])
        for flag in ("--model.action-chunk-size", "--data.transform.repack.action-chunk-size"):
            self.assertEqual(managed_values(plan["argv"], flag), [["16"]])
        self.assertEqual(managed_values(plan["argv"], "--model.action-exec-horizon"), [["16"]])
        args = self.training_args(["--exp", "short", "--model.action-exec-horizon", "30"])
        with self.assertRaises(ValueError):
            build_train_command(dict(native_args_file=str(args), horizon=16), self.source())
        with self.assertRaises(ValueError):
            build_train_command(dict(native_args_file=str(self.training_args()), horizon=16,
                                     action_exec_horizon=30), self.source())
        shorter = build_train_command(dict(native_args_file=str(self.training_args()), horizon=16,
                                           action_exec_horizon=8), self.source())
        self.assertEqual(managed_values(shorter["argv"], "--model.action-exec-horizon"), [["8"]])

    def test_training_matching_managed_aliases_are_kept_once(self):
        args = self.training_args(
            [
                "--exp",
                "contract",
                "--model.action_dim=78",
                "--model.odim",
                "43",
                "--data.transform.repack.action_keys",
                "action.body_token_v1_1",
                "action[:14]",
            ]
        )
        plan = build_train_command(dict(native_args_file=str(args)), self.source())
        self.assertEqual(managed_values(plan["argv"], "--model.action-dim"), [["78"]])
        self.assertEqual(managed_values(plan["argv"], "--model.odim"), [["43"]])
        self.assertEqual(
            managed_values(plan["argv"], "--data.transform.repack.action-keys"),
            [["action.body_token_v1_1", "action[:14]"]],
        )

    def test_training_conflicts_duplicate_flags_and_neck_keys_are_rejected(self):
        source = self.source()
        conflicts = [
            ["--model.action-dim", "80"],
            ["--model.odim=45"],
            ["--model.action-chunk-size", "16"],
            ["--data.transform.repack.action-chunk-size=16"],
            ["--data.transform.repack.pad-action-dim", "80"],
            ["--data.transform.repack.pad-state-dim", "45"],
            ["--data.transform.field.pad-action-dim", "80"],
            ["--data.transform.field.pad-state-dim", "45"],
            ["--model.odim=43", "--model.odim", "43"],
            ["--data.transform.repack.action-keys", "action.neck"],
            ["--data.transform.field.stat-action-keys", "action", "action.neck"],
        ]
        for flags in conflicts:
            with self.subTest(flags=flags):
                args = self.training_args(["--exp", "contract", *flags])
                with self.assertRaises(ValueError):
                    build_train_command(dict(native_args_file=str(args)), source)
        for changes in (
            dict(action_keys=["action.neck"]),
            dict(state_keys=["observation.neck"]),
            dict(raw_state_dim=45),
            dict(action_dim=80),
        ):
            with self.subTest(changes=changes):
                args = self.training_args()
                with self.assertRaises(ValueError):
                    build_train_command(dict(native_args_file=str(args), **changes), source)

    def test_explicit_checkpoint_native_training_preserves_original_flags_without_injection(self):
        flags = ["--exp", "historical-native", "--model.action-dim", "80", "--model.odim", "45"]
        args = self.training_args(flags)
        plan = build_train_command(
            dict(
                native_args_file=str(args),
                embodiment_profile="checkpoint_native",
                raw_state_dim=45,
                action_dim=80,
            ),
            self.source(),
        )
        start = plan["argv"].index("finetune_sonic_psi0_config") + 1
        self.assertEqual(plan["argv"][start:], flags)

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
