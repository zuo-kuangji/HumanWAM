"""Observation and rollout contracts without native model weights or CUDA."""
import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from humanwam.backends.imagewam_joint_reference import (
    build_policy,
    build_train_command,
    normalized_history,
    validate_observation,
)


def observation(dimension=41, count=16):
    return {"current_state": np.zeros(44, np.float32),
            "past_actions": np.zeros((count, dimension), np.float32),
            "past_action_frame_offsets": np.arange(-2 * count, 0, 2),
            "ego_rgb": np.zeros((224, 224, 3), np.uint8),
            "chest_rgb": np.zeros((224, 224, 3), np.uint8)}


class JointAdapterTests(unittest.TestCase):
    def test_lazy_construction_does_not_import_torch_or_imagewam(self):
        code = """import sys
from humanwam.backends.imagewam_joint_reference import build_policy
for schema in ('joint41_external_orientation', 'joint47_predicted_rotation6d'):
    assert build_policy({'schema':schema,'horizon':16})._model is None
assert 'torch' not in sys.modules
assert 'imagewam' not in sys.modules
"""
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    def test_explicit_schema_and_horizon(self):
        for schema, dimension in (("joint41_external_orientation", 41),
                                   ("joint47_predicted_rotation6d", 47)):
            self.assertEqual(build_policy({"schema": schema}).dimension, dimension)
            with self.assertRaises(ValueError):
                build_policy({"schema": schema, "horizon": 40})
        with self.assertRaises(ValueError):
            build_policy({"schema": "motion76"})

    def test_explicit_strided_history_accepts_both_representations(self):
        for dimension in (41, 47):
            raw = validate_observation(observation(dimension), dimension)
            self.assertEqual(raw["past_actions"].shape, (16, dimension))

    def test_contiguous_or_current_or_long_history_rejected(self):
        for offsets in (np.arange(-16, 0), np.arange(-30, 2, 2), np.arange(-32, 0, 2) + 1):
            raw = observation()
            raw["past_action_frame_offsets"] = offsets
            with self.assertRaises(ValueError):
                validate_observation(raw, 41)
        with self.assertRaises(ValueError):
            validate_observation(observation(count=17), 41)

    def test_empty_json_history_needs_explicit_episode_start(self):
        raw = observation(count=0)
        raw.update(past_actions=[], past_action_frame_offsets=[])
        with self.assertRaises(ValueError):
            validate_observation(raw, 41)
        raw["at_episode_start"] = True
        self.assertEqual(validate_observation(raw, 41)["past_actions"].shape, (0, 41))

    def test_future_or_malformed_or_command_hands_rejected(self):
        changes = [{"future_state": np.zeros(44)}, {"future_image": np.zeros(3)},
                   {"normalized_past_actions": np.zeros((16, 41))},
                   {"current_state": np.zeros((16, 44))},
                   {"ego_rgb": np.zeros((3, 224, 224), np.uint8)},
                   {"chest_rgb": np.zeros((224, 224, 3), np.float32)},
                   {"past_actions": np.zeros((16, 76), np.float32)}]
        for change in changes:
            raw = observation()
            raw.update(change)
            with self.subTest(keys=list(change)), self.assertRaises(ValueError):
                validate_observation(raw, 41)
        for field, index in (("current_state", 30), ("past_actions", (0, 30))):
            raw = observation()
            raw[field][index] = 2.0
            with self.assertRaises(ValueError):
                validate_observation(raw, 41)

    def test_missing_history_is_normalized_zero_matching_training(self):
        class Normalizer:
            def forward(self, values, field):
                return values + 7
        source = np.zeros((2, 47), np.float32)
        before = copy.deepcopy(source)
        result = normalized_history(source, 47, Normalizer())
        np.testing.assert_array_equal(result[:14], 0)
        np.testing.assert_array_equal(result[14:], 7)
        np.testing.assert_array_equal(source, before)
        np.testing.assert_array_equal(normalized_history(np.empty((0, 47)), 47, Normalizer()), 0)

    def test_published_examples_have_distinct_schema_and_launcher(self):
        repo = Path(__file__).resolve().parents[3]
        for dimension in (41, 47):
            cfg = json.loads((repo / f"configs/humanwam/imagewam-joint{dimension}.json").read_text())
            self.assertEqual(cfg["backend"], "imagewam_joint_reference")
            self.assertEqual(build_policy(cfg["options"]).dimension, dimension)
        command = build_train_command({"native_config": __file__, "num_processes": 8}, repo)
        self.assertIn("8", command["argv"])
        self.assertIn(str(repo / "scripts/wholebody/train.py"), command["argv"])


if __name__ == "__main__":
    unittest.main()
