import tempfile
import unittest
from pathlib import Path

import numpy as np

from humanwam.backends.imagewam_delta import (
    LegacyClothesPolicy,
    WholeBodyDeltaPolicy,
    build_train_command,
    validate_observation,
)


class ImageWAMBoundaryTests(unittest.TestCase):
    def observation(self):
        state = np.zeros(44, dtype=np.float32)
        state[29:41] = 0.5
        return dict(
            ego_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
            chest_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
            current_state=state,
            past_actions=np.zeros((0, 76), dtype=np.float32),
        )

    def test_explicit_empty_history_and_two_current_rgb(self):
        result = validate_observation(self.observation())
        self.assertEqual(result["past_actions"].shape, (0, 76))
        self.assertEqual(result["current_state"].shape, (44,))

    def test_reject_future_state_and_command_hand_history(self):
        bad = self.observation()
        bad["future_state"] = np.zeros(44)
        with self.assertRaises(ValueError):
            validate_observation(bad)
        bad = self.observation()
        bad["past_actions"] = np.zeros((2, 78))
        with self.assertRaises(ValueError):
            validate_observation(bad)
        bad = self.observation()
        bad["current_state"][29] = 500
        with self.assertRaises(ValueError):
            validate_observation(bad)

    def test_reject_nonfinite_and_encoded_images(self):
        bad = self.observation()
        bad["current_state"][0] = np.nan
        with self.assertRaises(ValueError):
            validate_observation(bad)
        bad = self.observation()
        bad["ego_rgb"] = np.zeros((3, 24, 32), dtype=np.float32)
        with self.assertRaises(ValueError):
            validate_observation(bad)

    def test_h16_default_legacy40_explicit_without_loading_model(self):
        self.assertEqual(WholeBodyDeltaPolicy({}).horizon, 16)
        self.assertEqual(LegacyClothesPolicy(dict(horizon=40)).horizon, 40)
        with self.assertRaises(ValueError):
            LegacyClothesPolicy({})
        with self.assertRaises(ValueError):
            WholeBodyDeltaPolicy(dict(schema="sonic78_command_hands"))

    def test_real_training_entry_and_no_shell_command_interpolation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            entry = root / "scripts/wholebody/train.py"
            entry.parent.mkdir(parents=True)
            entry.write_text("# native entry fixture")
            config = root / "native config.yaml"
            config.touch()
            plan = build_train_command(
                dict(native_config=str(config), python="env/bin/python", num_processes=8), root
            )
            self.assertIsInstance(plan["argv"], list)
            self.assertIn(str(config.resolve()), plan["argv"])
            self.assertEqual(plan["argv"][0], "env/bin/python")
            self.assertNotIn("shell", plan)


if __name__ == "__main__":
    unittest.main()
