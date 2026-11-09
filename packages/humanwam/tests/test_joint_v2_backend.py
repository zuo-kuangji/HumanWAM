"""Check heading-aware inference inputs and prediction-to-SONIC handoff without actuation."""

import importlib.util
import json
import time
import unittest
from pathlib import Path

import numpy as np

from humanwam.backends.imagewam_joint_reference import (
    build_policy,
    normalized_history,
    validate_observation,
)


def observation(count=16):
    history = np.zeros((count, 47), np.float32)
    history[:, 29:41] = 0.5
    history[:, 41:47] = [0, 1, 0, -1, 0, 0]  # world yaw +90deg
    return dict(
        current_state=np.zeros(44, np.float32),
        past_actions=history,
        past_action_frame_offsets=np.arange(-2 * count, 0, 2),
        current_root_quaternion_wxyz=np.array([2**-0.5, 0, 0, 2**-0.5]),
        observation_time_s=time.monotonic() - 0.2,
        timestamp_clock="monotonic",
        ego_rgb=np.zeros((224, 224, 3), np.uint8),
        chest_rgb=np.zeros((224, 224, 3), np.uint8),
    )


class IdentityNormalizer:
    def forward(self, value, field):
        return value.copy()

    def backward(self, value, field):
        return value.copy()


class JointV2BackendTests(unittest.TestCase):
    def test_v2_requires_current_heading_and_real_observation_time(self):
        raw = observation()
        self.assertEqual(
            validate_observation(raw, 47, recipe_version="v2")["past_actions"].shape, (16, 47)
        )
        for changes in (
            {"current_root_quaternion_wxyz": [0, 0, 0, 0]},
            {"timestamp_clock": "wall"},
            {"observation_time_s": time.monotonic() + 100},
            {"observation_time_s": True},
            {"past_root_orientation_valid": [1] * 16},
        ):
            altered = dict(raw, **changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_observation(altered, 47, recipe_version="v2")

    def test_history_uses_current_heading_and_preserves_unknown_history_mask(self):
        raw = observation(count=2)
        result = normalized_history(
            raw["past_actions"],
            47,
            IdentityNormalizer(),
            root_anchor=raw["current_root_quaternion_wxyz"],
            orientation_valid=np.array([True, False]),
        )
        np.testing.assert_array_equal(result[:14], 0)
        np.testing.assert_allclose(result[14, 41:47], [1, 0, 0, 0, 1, 0], atol=1e-6)
        np.testing.assert_array_equal(result[15, 41:47], 0)
        np.testing.assert_array_equal(raw["past_actions"][:, 41:47], [[0, 1, 0, -1, 0, 0]] * 2)

    def test_episode_start_accepts_empty_json_history_without_inventing_a_pose(self):
        for optional_mask in ({}, {"past_root_orientation_valid": []}):
            raw = observation(count=0)
            raw.update(
                past_actions=[],
                past_action_frame_offsets=[],
                at_episode_start=True,
                **optional_mask,
            )
            checked = validate_observation(raw, 47, recipe_version="v2")
            self.assertEqual(checked["past_root_orientation_valid"].dtype.kind, "b")
            np.testing.assert_array_equal(
                normalized_history(
                    checked["past_actions"],
                    47,
                    IdentityNormalizer(),
                    root_anchor=checked["current_root_quaternion_wxyz"],
                    orientation_valid=checked["past_root_orientation_valid"],
                ),
                np.zeros((16, 47)),
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch"), "Native inference handoff needs PyTorch, no GPU"
    )
    def test_prediction_restores_world_and_builds_time_checked_native_packet(self):
        import torch
        from imagewam.joint_reference_contract import make_joint_recipe
        from imagewam.joint_reference_rotation import (
            heading_relative_rotation6d_to_world_quaternion,
            world_quaternion_to_heading_relative_rotation6d,
        )

        from humanwam.sonic_joint_reference import deserialize_pose_v1
        from humanwam.sonic_joint_runtime import JointReferenceBuffer, TimedJointPlan

        class Model:
            def infer_action(inner, **kwargs):
                self.assertEqual(tuple(kwargs["action_init"].shape), (1, 16, 47))
                self.assertEqual(kwargs["action_horizon"], 40)
                values = torch.zeros(40, 47)
                values[:, :29] = torch.arange(40)[:, None] * (2 / 30) * 0.1
                values[:, 29:41] = 0.5
                values[:, 41:47] = torch.tensor([1.0, 0, 0, 0, 1, 0])
                return {"action": values}

        repo = Path(__file__).resolve().parents[3]
        contract = json.loads(
            (repo / "configs/wholebody/source_contracts/g1_existing_measured_v1.json").read_text()
        )
        policy = build_policy(
            dict(schema="joint47_heading_relative_rotation6d_v2", horizon=40, execution_horizon=16)
        )
        policy._model, policy.normalizer = Model(), IdentityNormalizer()
        policy.body_target = "measured"
        policy.source_contract, policy.joint_recipe = contract, make_joint_recipe(contract)
        policy.reference_source_verified = False
        policy.to_world_quaternion = heading_relative_rotation6d_to_world_quaternion
        policy.to_rotation6d = world_quaternion_to_heading_relative_rotation6d
        policy.hidden, policy.mask = torch.zeros(1), torch.ones(1)
        policy.resize = lambda image: torch.from_numpy(image.copy()).permute(2, 0, 1).float() / 255
        raw = observation()
        output = policy.predict(raw)
        self.assertEqual(output["actions"].shape, (40, 47))
        np.testing.assert_allclose(
            output["root_quaternions_wxyz"],
            np.tile(raw["current_root_quaternion_wxyz"], (40, 1)),
            atol=1e-6,
        )
        self.assertEqual(output["execution_horizon"], 16)
        with self.assertRaises(ValueError):
            TimedJointPlan.from_policy_prediction(output)
        plan = TimedJointPlan.from_policy_prediction(output, allow_experimental_source=True)
        now = time.monotonic()
        buffer = JointReferenceBuffer(control_epoch_time_s=raw["observation_time_s"] - 1)
        buffer.submit(plan, now_s=now)
        window = buffer.window(now_s=now)
        decoded = deserialize_pose_v1(window.body_packet)
        np.testing.assert_allclose(decoded["joint_vel"], 0.1, atol=1e-5)
        self.assertEqual(window.inspire_hand_states.shape[1], 12)
        self.assertNotIn("inspire_hand_states", decoded)


if __name__ == "__main__":
    unittest.main()
