"""CPU boundary tests. These do not claim GPU inference or robot validation."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from humanwam.backends import groot as adapter


class Native:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_action(self, observation, options=None):
        self.last_observation = observation
        return {
            k: np.zeros((1, 16, d), np.float32)
            for k, d in [("motion_token", 64), ("left_hand_joints", 6), ("right_hand_joints", 6)]
        }, {"native": True}

    def get_modality_config(self):
        return {
            "action": SimpleNamespace(
                modality_keys=["motion_token", "left_hand_joints", "right_hand_joints"],
                delta_indices=list(range(16)),
            )
        }

    def reset(self, options=None):
        return {"reset": options}


class GrootTests(unittest.TestCase):
    def setUp(self):
        self.module = ModuleType("gr00t.policy.gr00t_policy")
        self.module.Gr00tPolicy = Native
        self.context = patch.dict(sys.modules, {"gr00t.policy.gr00t_policy": self.module})
        self.context.start()
        self.addCleanup(self.context.stop)
        self.config = dict(
            model_path="/tmp/fixture-model", device="cpu", embodiment_tag="NEW_EMBODIMENT"
        )
        self.obs = dict(
            video={"ego_view": np.zeros((1, 1, 8, 8, 3), np.uint8)}, state={}, language={}
        )

    def test_native_observation_and_info_roundtrip(self):
        p = adapter.build_policy(self.config)
        r = p.predict(self.obs)
        self.assertIs(p.native.last_observation, self.obs)
        self.assertEqual(r["info"], {"native": True})
        self.assertEqual(p.native.kwargs["device"], "cpu")
        self.assertTrue(p.native.kwargs["strict"])

    def test_feedback76_dense_output_only_explicit_matching_schema(self):
        p = adapter.build_policy(dict(self.config, output_format="feedback76"))
        self.assertEqual(p.predict(self.obs)["action"].shape, (16, 76))
        with self.assertRaises(ValueError):
            adapter.build_policy(
                dict(self.config, output_format="feedback76", embodiment_tag="UNITREE_G1_SONIC")
            )

    def test_reject_wrong_horizon(self):
        class Wrong(Native):
            def get_modality_config(self):
                c = super().get_modality_config()
                c["action"].delta_indices = list(range(40))
                return c

        self.module.Gr00tPolicy = Wrong
        with self.assertRaises(ValueError):
            adapter.build_policy(dict(self.config, output_format="feedback76"))

    def test_reject_stock78_output_in_feedback76_recipe(self):
        class Wrong(Native):
            def get_action(self, *args, **kwargs):
                a, i = super().get_action(*args, **kwargs)
                a["left_hand_joints"] = np.zeros((1, 16, 7), np.float32)
                return a, i

        self.module.Gr00tPolicy = Wrong
        p = adapter.build_policy(dict(self.config, output_format="feedback76"))
        with self.assertRaises(ValueError):
            p.predict(self.obs)

    def test_native_wire_methods(self):
        p = adapter.build_policy(self.config)
        self.assertEqual(p.reset({"episode": 3}), {"reset": {"episode": 3}})
        self.assertEqual(p.get_action(self.obs)[1], {"native": True})
        self.assertEqual(p.get_modality_config()["action"].delta_indices, list(range(16)))

    def test_dont_guess_flat_observation_schema(self):
        p = adapter.build_policy(self.config)
        with self.assertRaises(ValueError):
            p.predict({"image": np.zeros((8, 8, 3), np.uint8)})

    def test_real_native_command_paths_and_flags(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        for name in ["examples/finetune.sh", "gr00t/eval/run_gr00t_server.py"]:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        train = adapter.build_train_command(
            dict(
                base_model_path="/weights/base",
                dataset_path="/data/a b",
                output_dir="/runs/x",
                num_gpus=8,
            ),
            root,
        )
        self.assertIn("/data/a b", train["argv"])
        self.assertEqual(train["env"]["NUM_GPUS"], "8")
        serve = adapter.build_serve_command(dict(self.config, python="/groot/env/python"), root)
        self.assertEqual(serve["argv"][0], "/groot/env/python")
        self.assertIn("--model-path", serve["argv"])
        self.assertIn("127.0.0.1", serve["argv"])
        self.assertNotIn("http", str(serve))


if __name__ == "__main__":
    unittest.main()
