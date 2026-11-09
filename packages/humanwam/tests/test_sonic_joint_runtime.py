"""CPU simulated-clock planning tests; no encoder, GPU, socket or robot starts."""

import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from humanwam.sonic_joint_reference import G1_ISAACLAB_JOINT_NAMES, deserialize_pose_v1
from humanwam.sonic_joint_runtime import (
    ClockRegressionError,
    JointPlanError,
    JointPlanOverrunError,
    JointReferenceBuffer,
    StaleJointPlanError,
    TimedJointPlan,
)


def fields(*, observation=1000.0, ready=1000.1, plan_id="p0", horizon=40, interval=2 / 30):
    times = np.arange(horizon, dtype=np.float64) * interval
    return dict(
        plan_id=plan_id,
        observation_time_s=observation,
        prediction_ready_time_s=ready,
        joint_positions=np.tile((times * 0.4)[:, None], (1, 29)),
        root_quaternions_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (horizon, 1)),
        inspire_hand_states=np.tile(np.linspace(0.1, 0.9, 12), (horizon, 1)),
        relative_timestamps=times,
        joint_names=G1_ISAACLAB_JOINT_NAMES,
        joint_order="isaaclab",
        body_target_source="wbc_motor_qtarget",
        prediction_horizon=horizon,
        execution_horizon=16,
        action_interval_s=interval,
    )


def buffer(**kwargs):
    return JointReferenceBuffer(control_epoch_time_s=1000.0, **kwargs)


def test_main16_tail24_units_are_distinct_from_control_ticks_and_wire_hands_are_separate():
    plan = TimedJointPlan.from_prediction(**fields())
    b = buffer()
    d = b.submit(plan, now_s=1000.1)
    window = b.window(now_s=1000.101)
    assert plan.prediction_horizon == 40
    assert window.diagnostics["future_tail_samples"] == 24
    assert window.diagnostics["execution_horizon_source_intervals"] == 16
    assert math.isclose(window.diagnostics["execution_duration_s"], 16 * 2 / 30)
    assert math.isclose(window.execution_deadline_s, 1000.12 + 16 * 2 / 30)
    assert d["future_margin_after_reserved_jitter_s"] > 0.4
    assert window.reference.frame_index[0] == 6
    np.testing.assert_allclose(np.diff(window.absolute_timestamps_s), 1 / 50, atol=1e-12)
    decoded = deserialize_pose_v1(window.body_packet)
    assert set(decoded) == {"joint_pos", "joint_vel", "body_quat", "frame_index"}
    assert window.inspire_hand_states.shape[1] == 12
    assert not window.diagnostics["transport_started"]


def test_latency_and_jitter_are_budgeted_and_no_32_sample_lookahead_claim():
    plan32 = TimedJointPlan.from_prediction(**fields(horizon=32, ready=1000.11))
    with pytest.raises(StaleJointPlanError) as caught:
        buffer().submit(plan32, now_s=1000.11)
    assert caught.value.diagnostics["future_margin_after_reserved_jitter_s"] < 0
    plan40 = TimedJointPlan.from_prediction(**fields(ready=1000.59))
    with pytest.raises(StaleJointPlanError):
        buffer().submit(plan40, now_s=1000.59)
    b = buffer()
    b.submit(TimedJointPlan.from_prediction(**fields()), now_s=1000.1)
    # A fresh prediction cannot hide a long scheduling stall between submit and first use.
    with pytest.raises(StaleJointPlanError):
        b.window(now_s=1000.8)
    assert b.status(now_s=1000.8)["state"] == "stale"


def test_absolute_grid_ceil_and_positive_future_start_are_never_backfilled():
    f = fields()
    f["relative_timestamps"] += 0.2
    plan = TimedJointPlan.from_prediction(**f)
    b = buffer()
    with pytest.raises(StaleJointPlanError, match="future"):
        b.submit(plan, now_s=1000.1)
    b.submit(plan, now_s=1000.201)
    w = b.window(now_s=1000.201)
    assert w.reference.frame_index[0] == 11
    assert math.isclose(w.absolute_timestamps_s[0], 1000.22)
    assert w.reference.relative_timestamps[0] >= 0.2


def test_repeated_reads_do_not_extend_execution_deadline_and_overrun_is_explicit():
    b = buffer()
    b.submit(TimedJointPlan.from_prediction(**fields()), now_s=1000.1)
    first = b.window(now_s=1000.1)
    later = b.window(now_s=1000.8)
    assert first.execution_deadline_s == later.execution_deadline_s
    overlap = later.reference.frame_index[0] - first.reference.frame_index[0]
    np.testing.assert_array_equal(first.reference.joint_pos[overlap:], later.reference.joint_pos)
    np.testing.assert_array_equal(first.reference.joint_vel[overlap:], later.reference.joint_vel)
    assert b.status(now_s=first.execution_deadline_s + 0.001)["state"] == "overrun"
    with pytest.raises(JointPlanOverrunError):
        b.window(now_s=first.execution_deadline_s + 0.001)


def test_qvel_is_same_plan_global_grid_derivative_not_timestamp_delta_confusion():
    f = fields()
    times = f["relative_timestamps"]
    f["joint_positions"] = np.tile((times * times)[:, None], (1, 29))
    b = buffer()
    b.submit(TimedJointPlan.from_prediction(**f), now_s=1000.1)
    a = b.window(now_s=1000.1)
    c = b.window(now_s=1000.24)
    offset = c.reference.frame_index[0] - a.reference.frame_index[0]
    np.testing.assert_array_equal(a.reference.joint_vel[offset:], c.reference.joint_vel)
    assert np.max(c.reference.joint_vel) > 1
    supplied = np.full((40, 29), 0.7)
    with pytest.raises(ValueError, match="same reference"):
        TimedJointPlan.from_prediction(**dict(fields(), joint_velocities=supplied))
    b2 = buffer()
    b2.submit(
        TimedJointPlan.from_prediction(
            **dict(
                fields(),
                joint_velocities=supplied,
                joint_velocity_source="same_reference_trajectory_rad_per_second",
            )
        ),
        now_s=1000.1,
    )
    np.testing.assert_allclose(b2.window(now_s=1000.1).reference.joint_vel, 0.7, atol=1e-7)


def test_clock_order_ready_time_clock_id_and_plan_reuse_fail_explicitly():
    b = buffer()
    p = TimedJointPlan.from_prediction(**fields())
    with pytest.raises(JointPlanError, match="ready timestamp"):
        b.submit(p, now_s=1000.05)
    b.submit(p, now_s=1000.1)
    b.window(now_s=1000.2)
    with pytest.raises(ClockRegressionError):
        b.window(now_s=1000.19)
    with pytest.raises(JointPlanError, match="plan_id"):
        b.submit(p, now_s=1000.21)
    with pytest.raises(JointPlanError, match="clock"):
        buffer(clock_id="controller2").submit(p, now_s=1000.1)
    with pytest.raises(JointPlanError, match="Out-of-order"):
        b.submit(TimedJointPlan.from_prediction(**fields(plan_id="new-id")), now_s=1000.22)
    with pytest.raises(ValueError, match="before observation"):
        TimedJointPlan.from_prediction(**fields(ready=999.99))


def test_fresh_plan_switches_only_future_grid_and_reports_discontinuity_without_blending():
    b = buffer()
    b.submit(TimedJointPlan.from_prediction(**fields()), now_s=1000.1)
    old = b.window(now_s=1000.1)
    saved = old.reference.joint_pos.copy()
    future = fields(observation=1000.8, ready=1000.9, plan_id="next")
    future["joint_positions"] += 0.7
    handoff = b.submit(TimedJointPlan.from_prediction(**future), now_s=1000.901)
    nxt = b.window(now_s=1000.901)
    assert nxt.reference.frame_index[0] == 46
    assert nxt.execution_deadline_s > old.execution_deadline_s
    assert handoff["handoff"]["previous_plan_id"] == "p0"
    assert not handoff["handoff"]["trajectory_blended"]
    assert handoff["handoff"]["joint_position_jump_max_abs_rad"] > 0.3
    np.testing.assert_array_equal(old.reference.joint_pos, saved)
    assert not nxt.reference.joint_pos.flags.writeable


def test_other_explicit_source_dt_is_supported_without_hardcoding30fps():
    f = fields(interval=0.05)
    p = TimedJointPlan.from_prediction(**f)
    b = buffer(action_interval_s=0.05)
    b.submit(p, now_s=1000.1)
    w = b.window(now_s=1000.1)
    assert math.isclose(w.diagnostics["execution_duration_s"], 0.8)
    assert w.diagnostics["control_dt_s"] == 0.02
    with pytest.raises(JointPlanError, match="timing"):
        buffer().submit(p, now_s=1000.1)
    wrong = fields()
    wrong["relative_timestamps"] *= 2
    with pytest.raises(ValueError, match="timestamps"):
        TimedJointPlan.from_prediction(**wrong)


@pytest.mark.parametrize("uptime", [1e7, 1e8])
@pytest.mark.parametrize("epoch_from_observation", [False, True])
def test_long_monotonic_uptime_uses_precise_relative_resampling_grid(
    uptime, epoch_from_observation
):
    # Neither the capture time nor the requested use time lies on the global
    # 50Hz grid. Large absolute-time subtraction must not jitter the relative
    # resampling grid or contaminate finite-difference velocities.
    observation = uptime + 0.0134567
    epoch = observation - 1 if epoch_from_observation else uptime
    plan = TimedJointPlan.from_prediction(
        **fields(observation=observation, ready=observation + 0.1)
    )
    b = JointReferenceBuffer(control_epoch_time_s=epoch)
    b.submit(plan, now_s=observation + 0.101)
    first = b.window(now_s=observation + 0.101)
    later = b.window(now_s=observation + 0.241)
    for window in (first, later):
        np.testing.assert_allclose(
            np.diff(window.reference.relative_timestamps), 0.02, rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(window.reference.joint_vel, 0.4, rtol=0, atol=1e-6)
        assert window.absolute_timestamps_s[0] >= window.diagnostics["now_s"]
        assert not window.diagnostics["future_padding"]
        deserialize_pose_v1(window.body_packet)
    offset = later.reference.frame_index[0] - first.reference.frame_index[0]
    np.testing.assert_array_equal(first.reference.joint_vel[offset:], later.reference.joint_vel)


@pytest.mark.parametrize(
    "changes",
    [
        {"root_orientation_frame": "robot_heading_relative"},
        {"root_orientation_source": "recorded_future"},
        {"timestamp_clock": "wall"},
        {"normalized": True},
        {"joint_units": "degree"},
        {"prediction_horizon": 16},
        {"action_interval_s": 0},
        {"observation_time_s": float("nan")},
    ],
)
def test_ambiguous_trajectory_schema_or_time_is_rejected(changes):
    with pytest.raises(ValueError):
        TimedJointPlan.from_prediction(**dict(fields(), **changes))


def policy_result(body_target="direct_reference", *, synchronized=True, coordinates=True):
    """Synthetic source receipt and world-restored backend output, never a robot recording."""
    f = fields()
    source_key = {
        "direct_reference": "reference.joint_position",
        "wbc": "action.wbc",
        "measured": "observation.state",
    }[body_target]
    label = {
        "direct_reference": "direct_joint_reference",
        "wbc": "wbc_motor_qtarget_experiment",
        "measured": "reference_from_demonstration",
    }[body_target]
    contract = {
        "schema": "g1_joint_reference_source_v1",
        "body_target": body_target,
        "joint_position_key": source_key,
        "joint_position_indices": list(range(29))
        if body_target == "direct_reference"
        else list(range(22)) + list(range(29, 36)),
        "joint_names": list(f["joint_names"]),
        "joint_units": "rad",
        "root_quaternion_key": "observation.root_orientation",
        "root_quaternion_order": "wxyz",
        "root_frame": "world",
        "root_validity_policy": "teleop_recorded_flags",
        "same_stage_pairing": True,
        "label_semantics": label,
        "provenance": {
            "description": "Synthetic CPU source contract",
            "source": "unit-test",
            "revision": "fixture-v1",
            "coordinate_frame_verified": coordinates,
            "synchronization_verified": synchronized,
        },
    }
    digest = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    recipe = {
        "recipe_version": "v2",
        "prediction_horizon": 40,
        "past_action_horizon": 16,
        "execution_horizon": 16,
        "lookahead_tail_horizon": 24,
        "action_stride": 2,
        "reference_fps": 30,
        "future_image_offset": 32,
        "source_history_rule": "history16_then_hold_last24",
        "orientation_target": "predicted_rotation6d",
        "orientation_prediction_frame": "current_heading_relative",
        "orientation_anchor_source_key": "observation.root_orientation",
        "orientation_anchor_convention": "world_wxyz_heading_about_z",
        "normalization_strategy": "fit_joint_reference_body29_q01q99_fixed_inspire_hands_heading_relative_rotation6d",
        "reference_source_contract": copy.deepcopy(contract),
        "reference_source_contract_sha256": digest,
    }
    actions = np.concatenate(
        [
            f["joint_positions"],
            f["inspire_hand_states"],
            np.tile([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], (40, 1)),
        ],
        axis=1,
    )
    return {
        "actions": actions,
        "body_joint_position29": actions[:, :29],
        "inspire_hand_state12": actions[:, 29:41],
        "root_quaternions_wxyz": f["root_quaternions_wxyz"],
        "relative_timestamps": f["relative_timestamps"],
        "joint_names": list(f["joint_names"]),
        "joint_order": f["joint_order"],
        "body_target_source": {
            "direct_reference": "direct_joint_reference",
            "wbc": "wbc_motor_qtarget",
            "measured": "measured_joint_position",
        }[body_target],
        "root_orientation_source": "predicted_future",
        "root_orientation_frame": "world",
        "observation_time_s": f["observation_time_s"],
        "prediction_ready_time_s": f["prediction_ready_time_s"],
        "timestamp_clock": "monotonic",
        "plan_id": f["plan_id"],
        "joint_recipe": recipe,
        "reference_source_contract": contract,
        "source_contract_sha256": digest,
        "normalized": False,
        "prediction_horizon": 40,
        "past_action_horizon": 16,
        "execution_horizon": 16,
        "lookahead_tail_horizon": 24,
        "action_interval_s": 2 / 30,
    }


def test_complete_policy_prediction_preserves_receipt_but_never_claims_controller_validation():
    prediction = policy_result()
    # Quaternion signs are equivalent orientations.
    prediction["root_quaternions_wxyz"][5::2] *= -1
    plan = TimedJointPlan.from_policy_prediction(prediction)
    b = buffer()
    b.submit(plan, now_s=1000.1)
    provenance = b.window(now_s=1000.1).diagnostics["source_provenance"]
    assert not provenance["experimental_control_chain"]
    assert not provenance["robot_execution_validated"]
    assert not provenance["controller_validated"]
    assert provenance["source_synchronization_verified"]
    prediction["reference_source_contract"]["provenance"]["description"] = "changed"
    assert (
        plan.source_provenance["reference_source_contract"]["provenance"]["description"]
        == "Synthetic CPU source contract"
    )


@pytest.mark.parametrize("source", ["wbc", "measured"])
def test_experimental_measured_and_motor_target_sources_require_explicit_opt_in(source):
    prediction = policy_result(source, synchronized=False)
    with pytest.raises(ValueError, match="allow_experimental_source"):
        TimedJointPlan.from_policy_prediction(prediction)
    plan = TimedJointPlan.from_policy_prediction(prediction, allow_experimental_source=True)
    assert plan.source_provenance["experimental_control_chain"]
    assert not plan.source_provenance["source_synchronization_verified"]
    assert not plan.source_provenance["controller_validated"]


@pytest.mark.parametrize("source", ["direct_reference", "wbc", "measured"])
def test_unverified_coordinates_cannot_be_sent_to_world_frame_runtime_even_with_opt_in(source):
    with pytest.raises(ValueError, match="Coordinate frame"):
        TimedJointPlan.from_policy_prediction(
            policy_result(source, coordinates=False), allow_experimental_source=True
        )


def test_policy_source_hash_recipe_and_raw_action_views_cannot_disagree():
    for mutate, pattern in [
        (
            lambda p: p["reference_source_contract"]["provenance"].update(revision="tampered"),
            "SHA256",
        ),
        (lambda p: p["joint_recipe"].update(prediction_horizon=16), "recipe"),
        (lambda p: p.update(lookahead_tail_horizon=0), "timing"),
        (lambda p: p.update(normalized=True), "denormalized"),
        (lambda p: p.update(root_orientation_frame="current_heading_relative"), "world"),
        (lambda p: p.update(joint_names=list(reversed(p["joint_names"]))), "joint names"),
        (lambda p: p.update(body_joint_position29=p["body_joint_position29"] + 1), "views"),
        (
            lambda p: p.update(root_quaternions_wxyz=np.tile([0.0, 1.0, 0.0, 0.0], (40, 1))),
            "rotation6D",
        ),
    ]:
        prediction = policy_result()
        mutate(prediction)
        with pytest.raises(ValueError, match=pattern):
            TimedJointPlan.from_policy_prediction(prediction)


def _json_arrays(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_arrays(v) for k, v in value.items()}
    return value


def run_export(tmp_path, prediction, *, now="1000.1", experimental=False):
    # Simulated clock values are intentionally explicit, with no hardware activity.
    source = tmp_path / "prediction.json"
    source.write_text(json.dumps(_json_arrays(prediction)))
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "window"
    command = [
        sys.executable,
        str(root / "scripts/humanwam/build_sonic_joint_window.py"),
        "--input",
        str(source),
        "--output-dir",
        str(output),
        "--now",
        now,
        "--control-epoch",
        "1000.0",
    ]
    if experimental:
        command += ["--allow-experimental-source"]
    env = dict(os.environ, PYTHONPATH=str(root / "packages/humanwam/src"))
    return subprocess.run(command, env=env, text=True, capture_output=True), output


def test_export_cli_outputs_existing_body_wire_and_separate_hands_without_actuation(tmp_path):
    result, output = run_export(tmp_path, policy_result())
    assert result.returncode == 0, result.stderr
    assert set(p.name for p in output.iterdir()) == {
        "body_pose_v1.bin",
        "inspire_reference50hz.npz",
        "window_manifest.json",
    }
    body = deserialize_pose_v1((output / "body_pose_v1.bin").read_bytes())
    with np.load(output / "inspire_reference50hz.npz") as hands:
        assert hands["inspire_hand_states"].shape[1] == 12
        np.testing.assert_array_equal(hands["frame_index"], body["frame_index"])
        assert hands["absolute_timestamps_s"][0] == 1000.1
    manifest = json.loads((output / "window_manifest.json").read_text())
    assert not manifest["transport_started"] and not manifest["robot_execution_validated"]
    again, _ = run_export(tmp_path, policy_result())
    assert again.returncode != 0 and "already exists" in again.stderr


@pytest.mark.parametrize(
    "source,now,experimental",
    [
        ("wbc", "1000.1", False),
        ("direct_reference", "1001.0", False),
        ("measured", "nan", True),
    ],
)
def test_export_cli_invalid_source_or_expired_clock_creates_no_output(
    tmp_path, source, now, experimental
):
    result, output = run_export(tmp_path, policy_result(source), now=now, experimental=experimental)
    assert result.returncode != 0
    assert not output.exists()
