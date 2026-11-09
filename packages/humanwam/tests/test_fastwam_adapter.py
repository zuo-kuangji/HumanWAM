"""Offline contract tests; injected non-model runtime is not GPU validation."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from humanwam.backends import fastwam as module
from humanwam.backends import fastwam_commands as commands

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "third_party/FastWAM"


def stats_fixture(dim, state_dim):
    result = {}
    for group, size in (("action", dim), ("state", state_dim)):
        result[group] = {
            "default": {
                "global_min": [-1.0] * size,
                "global_max": [1.0] * size,
                "global_mean": [0.0] * size,
                "global_std": [1.0] * size,
            }
        }
    result["action"]["default"]["stepwise_min"] = [[-1.0] * dim for _ in range(32)]
    return result


class Runtime:
    def __init__(self, config, profile):
        self.profile = profile
        self.calls = []
        self.output = np.full((1, 32, profile["action_dim"]), 0.25, dtype=np.float32)

    def infer(self, image, state, prompt, **kwargs):
        self.calls.append((image, state, prompt, kwargs))
        return self.output


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ckpt = self.root / "task.pt"
        self.ckpt.write_bytes(b"fixture-not-a-model")
        self.stats = self.root / "stats.json"
        self.stats.write_text(json.dumps(stats_fixture(7, 8)))
        self.config = {
            "profile": "libero_uncond_2cam224",
            "checkpoint": str(self.ckpt),
            "stats_path": str(self.stats),
            "binarize_gripper": False,
        }
        self.policy = module.FastWAMPolicy(self.config, runtime_factory=Runtime)
        self.obs = {
            "images": [
                np.full((80, 90, 3), 20, dtype=np.uint8),
                np.full((80, 90, 3), 70, dtype=np.uint8),
            ],
            "state": np.arange(8, dtype=np.float32),
            "prompt": "put bowl down",
        }

    def tearDown(self):
        self.tmp.cleanup()

    def test_lazy_single_load_and_native_current_image_state_contract(self):
        self.assertIsNone(self.policy._runtime)
        output = self.policy.predict(self.obs)
        runtime = self.policy._runtime
        self.policy.predict(dict(self.obs, images=np.stack(self.obs["images"])))
        self.assertIs(self.policy._runtime, runtime)
        self.assertEqual(len(runtime.calls), 2)
        image, state, prompt, kwargs = runtime.calls[0]
        self.assertEqual(image.shape, (224, 448, 3))
        np.testing.assert_array_equal(image[:, :224], 20)
        np.testing.assert_array_equal(image[:, 224:], 70)
        np.testing.assert_array_equal(state, self.obs["state"])
        self.assertTrue(prompt.endswith("instruction: put bowl down"))
        self.assertEqual((kwargs["action_horizon"], kwargs["sigma_shift"]), (32, 5.0))
        self.assertEqual(np.asarray(output["actions"]).shape, (32, 7))
        np.testing.assert_allclose(np.asarray(output["actions"])[:, -1], 0.5)
        self.assertFalse(output["normalized"])
        self.assertEqual(output["action_space"], "libero_eef_delta7_gripper")
        self.assertFalse(output["native"]["robot_execution_verified"])

    def test_robotwin_three_camera_native_layout_and_joint_units(self):
        self.stats.write_text(json.dumps(stats_fixture(14, 14)))
        policy = module.FastWAMPolicy(
            dict(self.config, profile="robotwin_uncond_3cam_384"), runtime_factory=Runtime
        )
        obs = {
            "images": [np.full((60, 70, 3), c, dtype=np.uint8) for c in [10, 30, 90]],
            "state": np.zeros(14),
            "prompt": "move bottle",
        }
        output = policy.predict(obs)
        image = policy._runtime.calls[0][0]
        self.assertEqual(image.shape, (384, 320, 3))
        np.testing.assert_array_equal(image[:256], 10)
        np.testing.assert_array_equal(image[256:, :160], 30)
        np.testing.assert_array_equal(image[256:, 160:], 90)
        self.assertEqual(output["action_space"], "robotwin_joint14")
        np.testing.assert_allclose(output["actions"], 0.25)

    def test_reject_semantic_and_dimension_relabeling_before_loading(self):
        for options in [
            {"action_schema": "sonic76"},
            {"action_dim": 76},
            {"horizon": 16},
            {"method": "delta_flow"},
            {"profile": "g1"},
        ]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                module.FastWAMPolicy(dict(self.config, **options), runtime_factory=Runtime)

    def test_bad_inputs_do_not_load_runtime(self):
        for options in [
            {"past_actions": np.zeros((16, 7))},
            {"state_normalized": np.zeros(8)},
            {"state": np.zeros(76)},
            {"state": np.full(8, np.nan)},
            {"images": self.obs["images"][:1]},
            {"prompt": ""},
        ]:
            with self.subTest(options=list(options)), self.assertRaises(ValueError):
                self.policy.predict(dict(self.obs, **options))
            self.assertIsNone(self.policy._runtime)

    def test_stats_horizon_dimensions_and_hash_are_checked_on_cpu(self):
        values = stats_fixture(7, 44)
        self.stats.write_text(json.dumps(values))
        with self.assertRaisesRegex(ValueError, "state.default"):
            module.FastWAMPolicy(self.config)
        self.stats.write_text(json.dumps(stats_fixture(7, 8)))
        with self.assertRaisesRegex(ValueError, "SHA256"):
            module.FastWAMPolicy(dict(self.config, stats_sha256="0" * 64))

    def test_wrong_output_shape_and_nan_fail(self):
        self.policy.predict(self.obs)
        for output in [np.zeros((1, 16, 7)), np.full((1, 32, 7), np.nan)]:
            self.policy._runtime.output = output
            with self.assertRaisesRegex(ValueError, "Native FastWAM output"):
                self.policy.predict(self.obs)

    def test_native_permissive_checkpoint_does_not_allow_partial_weights(self):
        tensor = SimpleNamespace(shape=(2, 3))
        model = SimpleNamespace(
            mot=SimpleNamespace(state_dict=lambda: {"w": tensor}),
            proprio_encoder=SimpleNamespace(state_dict=lambda: {"p": tensor}),
        )
        module._validate_model_payload(
            model, {"mot": {"w": tensor}, "proprio_encoder": {"p": tensor}}
        )
        for payload in [
            {"dit": {}},
            {"mot": {"w": tensor}},
            {"mot": {}, "proprio_encoder": {"p": tensor}},
            {"mot": {"w": SimpleNamespace(shape=(2, 7))}, "proprio_encoder": {"p": tensor}},
        ]:
            with self.subTest(payload=list(payload)), self.assertRaises(ValueError):
                module._validate_model_payload(model, payload)

    def test_local_assets_preflight_and_environment_restoration(self):
        with self.assertRaises(FileNotFoundError):
            module.validate_local_assets(self.root)
        for name in [
            "Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth",
            "Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth",
            "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer_config.json",
            "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/tokenizer.json",
        ]:
            p = self.root / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()
        self.assertEqual(module.validate_local_assets(self.root), self.root.resolve())
        previous = os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD")
        with self.assertRaisesRegex(RuntimeError, "fixture"):
            with module._local_asset_environment(self.root):
                self.assertEqual(os.environ["DIFFSYNTH_SKIP_DOWNLOAD"], "true")
                raise RuntimeError("fixture")
        self.assertEqual(os.environ.get("DIFFSYNTH_SKIP_DOWNLOAD"), previous)

    def test_import_and_construction_do_not_import_torch_or_fastwam(self):
        script = "import sys; from humanwam.backends.fastwam import FastWAMPolicy; "
        script += "FastWAMPolicy(" + repr(self.config) + "); "
        script += "assert 'torch' not in sys.modules and 'fastwam' not in sys.modules"
        subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


class LaunchGuardTests(unittest.TestCase):
    def test_lazy_native_load_rechecks_exact_statistics_bytes_before_model_import(self):
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stats.json"
            original = json.dumps(stats_fixture(7, 8)).encode()
            path.write_bytes(original)
            config = {"stats_path": str(path), "stats_sha256": hashlib.sha256(original).hexdigest()}
            profile = module.PROFILES["libero_uncond_2cam224"]
            self.assertEqual(module._verified_stats_bytes(config, profile), original)
            path.write_text("{}")
            # No source_dir or model assets are needed: reject before imports/allocation.
            with self.assertRaisesRegex(ValueError, "statistics SHA256"):
                module._NativeRuntime(config, profile)

    def test_hydra_add_prefix_cannot_override_launch_guards(self):
        for prefix in ("", "+", "++"):
            for key in ("task", "output_dir", "resume", "output_dir.child"):
                with self.subTest(key=prefix + key), self.assertRaises(ValueError):
                    commands._overrides({prefix + key: "/existing/run"})


@unittest.skipUnless((SOURCE / "scripts/train.py").is_file(), "pinned upstream not checked out")
class NativeSourceTests(unittest.TestCase):
    def test_used_native_inference_keyword_signature_matches_pinned_source(self):
        source = SOURCE / "src/fastwam/models/wan22/fastwam.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FastWAM")
        infer = next(
            n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "infer_action"
        )
        names = {a.arg for a in infer.args.args + infer.args.kwonlyargs}
        self.assertTrue(
            {
                "prompt",
                "input_image",
                "proprio",
                "action_horizon",
                "sigma_shift",
                "compile_action_infer",
                "num_inference_steps",
            }
            <= names
        )

    @unittest.skipUnless(importlib.util.find_spec("hydra"), "Hydra absent in CPU QA environment")
    def test_native_hydra_training_command_composes_without_torch(self):
        from hydra import compose, initialize_config_dir

        with tempfile.TemporaryDirectory() as tmp:
            command = commands.build_train_command(
                {
                    "task": "libero_uncond_2cam224_1e-4",
                    "model_cache_root": tmp,
                    "output_dir": str(Path(tmp) / "run with spaces"),
                    "overrides": {
                        "model.redirect_common_files": False,
                        "data.train.dataset_dirs": ["/data/a space"],
                        "batch_size": 8,
                    },
                },
                SOURCE,
            )
            start = command["argv"].index(str(SOURCE / "scripts/train.py")) + 1
            with initialize_config_dir(config_dir=str(SOURCE / "configs"), version_base="1.3"):
                cfg = compose(config_name="train", overrides=command["argv"][start:])
            self.assertEqual(cfg.batch_size, 8)
            self.assertEqual(cfg.model.proprio_dim, 8)
            self.assertEqual(cfg.model.action_dit_config.action_dim, 7)
            self.assertEqual(cfg.data.train.dataset_dirs, ["/data/a space"])
            self.assertTrue(cfg.output_dir.endswith("run with spaces"))

    def test_commands_and_existing_output_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            cfg = {"output_dir": str(output), "model_cache_root": tmp}
            cmd = commands.build_train_command(cfg, SOURCE)
            self.assertEqual(cmd["cwd"], str(SOURCE))
            self.assertIn("accelerate.commands.launch", cmd["argv"])
            self.assertEqual(cmd["env"]["DIFFSYNTH_SKIP_DOWNLOAD"], "true")
            output.mkdir()
            (output / "log").touch()
            with self.assertRaisesRegex(ValueError, "fresh"):
                commands.build_train_command(cfg, SOURCE)
            with self.assertRaises(ValueError):
                commands.build_train_command(dict(cfg, task="../../outside"), SOURCE)
            launcher = Path(tmp) / "native.json"
            launcher.write_text("{}")
            serve = commands.build_serve_command({"launcher_config": str(launcher)}, SOURCE)
            self.assertIn("humanwam.http_server", serve["argv"])
            self.assertEqual(serve["argv"][-1], "8094")


if __name__ == "__main__":
    unittest.main(verbosity=2)
