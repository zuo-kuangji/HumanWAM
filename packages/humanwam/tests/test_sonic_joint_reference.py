"""Offline references and byte compatibility; never instantiate a ZMQ publisher."""

import ast
import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from humanwam.sonic_joint_reference import (
    G1_ISAACLAB_JOINT_NAMES,
    G1_MUJOCO_JOINT_NAMES,
    G1_MUJOCO_TO_ISAACLAB,
    HEADER_SIZE,
    InsufficientLookaheadError,
    build_joint_reference,
    deserialize_pose_v1,
    rotation6d_to_quaternions_wxyz,
    serialize_pose_v1,
    split_model_prediction,
)

ROOT = Path(__file__).resolve().parents[3]
WBC = ROOT / "third_party/GR00T-WholeBodyControl"


def example(n=16, span=1.0):
    times = np.linspace(0, span, n)
    positions = times[:, None] * np.arange(1, 30)[None, :] / 50
    yaw = times * np.pi / 2
    return dict(
        joint_positions=positions,
        joint_names=G1_ISAACLAB_JOINT_NAMES,
        joint_order="isaaclab",
        root_quaternions_wxyz=np.column_stack((np.cos(yaw / 2), np.zeros((n, 2)), np.sin(yaw / 2))),
        relative_timestamps=times,
        inspire_hand_states=np.tile(np.linspace(0.1, 0.9, 12), (n, 1)),
        normalized=False,
        joint_units="rad",
        action_semantics="absolute_joint_position_reference",
        body_target_source="wbc_motor_qtarget",
        root_orientation_source="predicted_future",
        frame_index_start=123,
    )


def test_h16_contiguous30hz_refused_but_stride2_covered_without_padding():
    inputs = example(span=15 / 30)
    with pytest.raises(InsufficientLookaheadError) as caught:
        build_joint_reference(**inputs)
    assert caught.value.coverage["available_control_frames"] == 26
    reference = build_joint_reference(**example(span=15 * 2 / 30))
    assert reference.joint_pos.shape == (51, 29)
    np.testing.assert_array_equal(reference.frame_index, np.arange(123, 174))
    assert reference.relative_timestamps[-1] == 1
    assert not reference.metadata["future_padding"]
    assert reference.metadata["body_target_source"] == "wbc_motor_qtarget"
    assert reference.metadata["experimental_control_chain"]
    assert not reference.metadata["original_sonic_reference_equivalence_verified"]
    assert reference.inspire_hand_states.shape == (51, 12)
    assert set(reference.protocol_fields()) == {"joint_pos", "joint_vel", "body_quat", "frame_index"}


def test_linear_reference_derivative_and_shortest_arc_slerp():
    inputs = example(n=2)
    inputs["root_quaternions_wxyz"][1] *= -1  # q and -q are the same orientation.
    reference = build_joint_reference(**inputs)
    np.testing.assert_allclose(reference.joint_vel, np.tile(np.arange(1, 30) / 50, (51, 1)), atol=1e-6)
    np.testing.assert_allclose(reference.body_quat[25, 0], [np.cos(np.pi / 8), 0, 0, np.sin(np.pi / 8)], atol=1e-7)
    assert reference.metadata["velocity_source"].startswith("finite_difference")
    assert reference.metadata["root_orientation_source"] == "predicted_future"


def test_no_extrapolation_beyond_last_fractional_control_tick():
    reference = build_joint_reference(**example(span=0.917))
    assert len(reference.frame_index) == 46
    assert reference.relative_timestamps[-1] == 0.9
    assert reference.relative_timestamps[-1] <= 0.917


def test_positive_future_offset_is_preserved_and_never_filled_back_to_present():
    inputs = example()
    inputs["relative_timestamps"] += 2 / 30
    reference = build_joint_reference(**inputs)
    assert len(reference.frame_index) == 51
    assert reference.relative_timestamps[0] == 2 / 30
    assert reference.metadata["first_reference_relative_time"] == 2 / 30
    np.testing.assert_allclose(reference.joint_pos[0], inputs["joint_positions"][0], atol=1e-7)


def test_order_conversion_and_supplied_velocity_are_not_guessed():
    original = example()
    expected = build_joint_reference(**original)
    take = [G1_ISAACLAB_JOINT_NAMES.index(name) for name in G1_MUJOCO_JOINT_NAMES]
    original.update(joint_names=G1_MUJOCO_JOINT_NAMES, joint_order="mujoco",
                    joint_positions=original["joint_positions"][:, take],
                    joint_velocities=np.tile(np.arange(29), (16, 1))[:, take])
    actual = build_joint_reference(**original)
    np.testing.assert_array_equal(actual.joint_pos, expected.joint_pos)
    np.testing.assert_array_equal(actual.joint_vel[0], np.arange(29))
    original["joint_order"] = "isaaclab"
    with pytest.raises(ValueError, match="disagree"):
        build_joint_reference(**original)


@pytest.mark.parametrize("changes", [
    {"normalized": True}, {"joint_units": "degree"},
    {"body_target_source": "guess"},
    {"action_semantics": "motion64"}, {"action_semantics": "pd_residual"},
    {"root_quaternions_wxyz": None}, {"root_orientation_source": "current_pose"},
    {"joint_positions": np.zeros((16, 41))}, {"joint_positions": np.zeros((16, 64))},
    {"joint_positions": np.full((16, 29), np.nan)},
    {"inspire_hand_states": np.zeros((16, 14))},
    {"inspire_hand_states": np.full((16, 12), 1.2)},
    {"root_quaternions_wxyz": np.zeros((16, 4))},
    {"relative_timestamps": np.ones(16)},
    {"relative_timestamps": np.linspace(-0.1, 1.1, 16)},
    {"joint_names": ("waist_yaw_joint",) * 29},
    {"frame_index_start": -1}, {"frame_index_start": True},
])
def test_invalid_or_ambiguous_inputs_fail(changes):
    with pytest.raises(ValueError):
        build_joint_reference(**(example() | changes))


def test_isaaclab_names_and_permutation_match_pinned_native_sources():
    source = WBC / "gear_sonic/envs/manager_env/robots/g1.py"
    xml = WBC / "gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml"
    if not source.exists():
        pytest.skip("Pinned SONIC submodule is not initialized")
    module = ast.parse(source.read_text())
    assignments = {node.targets[0].id: ast.literal_eval(node.value) for node in module.body
                   if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                   and node.targets[0].id in {"G1_MUJOCO_TO_ISAACLAB_DOF", "G1_ISAACLAB_TO_MUJOCO_DOF"}}
    native_names = [j.attrib["name"] for j in ET.parse(xml).getroot().findall(".//worldbody//joint")
                    if j.attrib.get("type") != "free"]
    assert tuple(native_names) == G1_MUJOCO_JOINT_NAMES
    assert tuple(assignments["G1_MUJOCO_TO_ISAACLAB_DOF"]) == G1_MUJOCO_TO_ISAACLAB
    native_isaac_names = tuple(native_names[i] for i in assignments["G1_MUJOCO_TO_ISAACLAB_DOF"])
    assert native_isaac_names == G1_ISAACLAB_JOINT_NAMES
    assert [native_isaac_names[i] for i in assignments["G1_ISAACLAB_TO_MUJOCO_DOF"]] == native_names


def test_native_official_packer_and_unpacked_arrays_match_without_sockets():
    source = WBC / "gear_sonic/utils/teleop/zmq/zmq_planner_sender.py"
    if not source.exists():
        pytest.skip("Pinned SONIC submodule is not initialized")
    spec = importlib.util.spec_from_file_location("sonic_offline_packer_test", source)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)  # This module has no socket imports or instances.
    parser_source = WBC / "gear_sonic/scripts/run_data_exporter.py"
    function = next(node for node in ast.parse(parser_source.read_text()).body
                    if isinstance(node, ast.FunctionDef) and node.name == "unpack_pose_message")
    namespace = {"np": np, "json": json}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(parser_source), "exec"), namespace)
    reference = build_joint_reference(**example(span=0.9))
    packet = serialize_pose_v1(reference)
    assert len(packet) == 13060
    assert packet == native.pack_pose_message(reference.protocol_fields(), version=1)
    decoded = namespace["unpack_pose_message"](packet)
    assert decoded["version"] == 1
    for key, value in reference.protocol_fields().items():
        np.testing.assert_array_equal(decoded[key], value)
        np.testing.assert_array_equal(deserialize_pose_v1(packet)[key], value)


def test_strict_wire_decoder_rejects_corruption_and_non_dense_indices():
    packet = serialize_pose_v1(build_joint_reference(**example()))
    for invalid in (packet[:-1], packet + b"trailing", b"command" + packet[4:], b"pose\x00"):
        with pytest.raises(ValueError):
            deserialize_pose_v1(invalid)
    header = json.loads(packet[4:4 + HEADER_SIZE].split(b"\x00", 1)[0])
    header["v"] = 4
    changed = b"pose" + json.dumps(header).encode().ljust(HEADER_SIZE, b"\x00") + packet[4 + HEADER_SIZE:]
    with pytest.raises(ValueError, match="v1"):
        deserialize_pose_v1(changed)
    bad = bytearray(packet)
    # Last field is frame_index, ensure a gap is rejected rather than reindexed.
    bad[-8:] = np.asarray([900], dtype="<i8").tobytes()
    with pytest.raises(ValueError, match="dense"):
        deserialize_pose_v1(bytes(bad))


def test_rotation6d_schema_handles_identity_half_turn_and_rejects_degenerate():
    rotations = np.asarray([[1, 0, 0, 0, 1, 0], [1, 0, 0, 0, -1, 0],
                            [-1, 0, 0, 0, 1, 0], [-1, 0, 0, 0, -1, 0]], dtype=float)
    expected = np.eye(4)
    actual = rotation6d_to_quaternions_wxyz(rotations)
    np.testing.assert_allclose(np.abs(actual), expected, atol=1e-7)
    for invalid in (np.zeros((2, 6)), np.tile([1, 0, 0, 2, 0, 0], (2, 1))):
        with pytest.raises(ValueError, match="Degenerate"):
            rotation6d_to_quaternions_wxyz(invalid)


def test_split_raw47_rotation_columns_and_external41_root_requirement():
    inputs = example()
    raw41 = np.column_stack((inputs["joint_positions"], inputs["inspire_hand_states"]))
    raw47 = np.column_stack((raw41, np.tile([0, 1, 0, -1, 0, 0], (16, 1))))
    split = split_model_prediction(raw47, prediction_schema="predicted_rotation6d47", normalized=False,
                                   body_target_source="wbc_motor_qtarget")
    np.testing.assert_allclose(split["root_quaternions_wxyz"], np.tile([2**-0.5, 0, 0, 2**-0.5], (16, 1)))
    with pytest.raises(ValueError, match="separately"):
        split_model_prediction(raw41, prediction_schema="external41", normalized=False,
                               body_target_source="wbc_motor_qtarget")
    with pytest.raises(ValueError, match="Denormalize"):
        split_model_prediction(raw47, prediction_schema="predicted_rotation6d47", normalized=True,
                               body_target_source="wbc_motor_qtarget")
    for schema, values in (("external41", raw41), ("predicted_rotation6d47", raw47)):
        kwargs = {"external_root_quaternions_wxyz": inputs["root_quaternions_wxyz"],
                  "external_root_orientation_source": "recorded_future"} if schema == "external41" else {}
        fields = split_model_prediction(values, prediction_schema=schema, normalized=False,
                                        body_target_source="wbc_motor_qtarget", **kwargs)
        reference = build_joint_reference(**(inputs | fields))
        assert reference.joint_pos.shape == (51, 29)


def test_cli_exports_only_files_and_fails_closed_for_short_chunk(tmp_path):
    inputs = example()
    source, output = tmp_path / "input.npz", tmp_path / "output.npz"
    wire = tmp_path / "pose.bin"
    np.savez(source, **inputs)
    command = [sys.executable, str(ROOT / "scripts/humanwam/export_sonic_joint_reference.py"),
               "--input", str(source), "--output", str(output), "--wire-output", str(wire)]
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["transport_started"] is False
    with np.load(output, allow_pickle=False) as saved:
        assert saved["inspire_hand_states"].shape == (51, 12)
        assert saved["joint_names"].tolist() == list(G1_ISAACLAB_JOINT_NAMES)
    assert deserialize_pose_v1(wire.read_bytes())["joint_pos"].shape == (51, 29)
    repeated = subprocess.run(command, capture_output=True, text=True, timeout=20)
    assert repeated.returncode == 2
    short = tmp_path / "short.npz"
    np.savez(short, **example(span=0.5))
    failed = subprocess.run(command[:3] + [str(short), "--output", str(tmp_path / "rejected.npz")],
                            capture_output=True, text=True, timeout=20)
    assert failed.returncode == 2
    assert json.loads(failed.stderr)["coverage"]["available_control_frames"] == 26
    assert not (tmp_path / "rejected.npz").exists()
