"""Lightweight provenance, timing and heading checks; no torch/GPU dependency."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from imagewam.joint_reference_contract import (
    validate_source_contract,
    source_contract_sha256,
    deployment_ready,
    make_joint_recipe,
    expected_schema,
    timing_v2,
)
from imagewam.joint_reference_rotation import (
    quaternion_matrix,
    heading_from_quaternion_wxyz,
    world_quaternion_to_heading_relative_rotation6d,
    heading_relative_rotation6d_to_world_quaternion,
)


def contract():
    return json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "configs/wholebody/source_contracts/g1_existing_measured_v1.json"
        ).read_text()
    )


def test_none_and_implicit_sources_are_errors_before_native_model_loading():
    for value in (None, [], "path"):
        with pytest.raises(ValueError, match="source_contract"):
            make_joint_recipe(value)
    bad = contract()
    bad.pop("provenance")
    with pytest.raises(ValueError):
        make_joint_recipe(bad)


def test_canonical_digest_is_order_independent_but_binds_provenance():
    a = contract()
    b = dict(reversed(list(a.items())))
    assert source_contract_sha256(a) == source_contract_sha256(b)
    b = copy.deepcopy(a)
    b["provenance"]["revision"] += " extra"
    assert source_contract_sha256(a) != source_contract_sha256(b)


def test_hydra_nested_config_is_canonicalized_only_for_reference_contracts():
    omegaconf = pytest.importorskip("omegaconf")
    value = contract()
    nested = omegaconf.OmegaConf.create(value)
    assert make_joint_recipe(nested) == make_joint_recipe(value)
    assert source_contract_sha256(nested) == source_contract_sha256(value)
    assert validate_source_contract(nested, "measured", "predicted_rotation6d") == value


@pytest.mark.parametrize(
    "field,value",
    [
        ("prediction_horizon", 32),
        ("execution_horizon", 40),
        ("past_action_horizon", 40),
        ("action_stride", 1),
        ("reference_fps", 60),
        ("future_image_offset", 80),
    ],
)
def test_real_prediction_clock_cannot_be_faked_by_config_labels(field, value):
    with pytest.raises(ValueError):
        make_joint_recipe(contract(), **{field: value})


def test_same_stage_demonstration_is_explicit_experiment_not_validated_sonic_reference():
    a = contract()
    a["same_stage_pairing"] = False
    result = validate_source_contract(a, "measured", "predicted_rotation6d")
    assert result["label_semantics"] == "reference_from_demonstration"
    assert not deployment_ready(result)
    assert expected_schema("measured", "predicted_rotation6d", "v2").endswith("-v2")


def test_direct_reference_must_be_a_separate_source_with_audited_pairing():
    a = contract()
    a.update(body_target="direct_reference", label_semantics="direct_joint_reference")
    with pytest.raises(ValueError, match="not direct reference"):
        make_joint_recipe(a)
    a.update(
        joint_position_key="teleop.original_qref",
        joint_position_indices=list(range(29)),
    )
    with pytest.raises(ValueError, match="verified same-stage"):
        make_joint_recipe(a)
    a["provenance"]["synchronization_verified"] = True
    result = make_joint_recipe(a)
    assert (
        result["reference_source_contract"]["joint_position_key"]
        == "teleop.original_qref"
    )
    assert deployment_ready(a)  # Source audit only; no robot execution assertion.


def test_current_heading_rotation_roundtrip_with_nontrivial_pitch_roll_and_sign():
    rng = np.random.default_rng(19)
    q = rng.normal(size=(40, 4))
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    current = q[0]
    relative = world_quaternion_to_heading_relative_rotation6d(q, current)
    world = heading_relative_rotation6d_to_world_quaternion(relative, current)
    np.testing.assert_allclose(
        quaternion_matrix(world), quaternion_matrix(q), atol=2e-6
    )
    np.testing.assert_array_equal(
        relative, world_quaternion_to_heading_relative_rotation6d(-q, -current)
    )


def test_common_world_heading_shift_does_not_change_relative_targets():
    theta = np.asarray([0.4, 0.7, 1.3])
    q = np.stack(
        (np.cos(theta / 2), np.zeros(3), np.zeros(3), np.sin(theta / 2)), axis=-1
    )
    before = world_quaternion_to_heading_relative_rotation6d(q, q[0])
    theta += 1.2
    q = np.stack(
        (np.cos(theta / 2), np.zeros(3), np.zeros(3), np.sin(theta / 2)), axis=-1
    )
    after = world_quaternion_to_heading_relative_rotation6d(q, q[0])
    np.testing.assert_allclose(before, after, atol=1e-6)
    assert np.isclose(timing_v2()["future_action_span_seconds"], 2.6)


def test_heading_anchor_or_rotation_degeneracy_cannot_be_silently_padded():
    with pytest.raises(ValueError):
        heading_from_quaternion_wxyz([0, 0, 0, 0])
    with pytest.raises(ValueError):
        heading_from_quaternion_wxyz([2**-0.5, 0, 2**-0.5, 0])
    with pytest.raises(ValueError):
        heading_relative_rotation6d_to_world_quaternion(np.zeros((3, 6)), [1, 0, 0, 0])
