"""Versioned joint-reference provenance and timing; no model/runtime imports."""

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence

SOURCE_SCHEMA = "g1_joint_reference_source_v1"
V2_SOURCE_HISTORY_RULE = "history16_then_hold_last24"
NORMALIZATION_V2 = (
    "fit_joint_reference_body29_q01q99_fixed_inspire_hands_heading_relative_rotation6d"
)


def plain_value(value):
    """Materialize nested Hydra DictConfig/ListConfig without changing other config trees."""
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("Reference contract keys must be strings")
        return {k: plain_value(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [plain_value(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("Reference contracts contain only JSON-compatible values")


def expected_schema(body_target, orientation_target, recipe_version="v2"):
    if body_target not in (
        "wbc",
        "measured",
        "direct_reference",
    ) or orientation_target not in ("external", "predicted_rotation6d"):
        raise ValueError("Explicit body/orientation targets required")
    if recipe_version not in ("v1", "v2") or (
        body_target == "direct_reference" and recipe_version == "v1"
    ):
        raise ValueError("Unsupported joint recipe/source combination")
    dim = 41 if orientation_target == "external" else 47
    label = {
        "wbc": "wbc29",
        "measured": "measured29",
        "direct_reference": "direct-reference29",
    }[body_target]
    suffix = "external-orientation" if dim == 41 else "quatwxyz-rotation6d-columns"
    return f"wholebody-state44-joint{dim}-{label}-measured-inspire12-{suffix}-{recipe_version}"


def make_joint_recipe(
    source_contract,
    *,
    prediction_horizon=40,
    past_action_horizon=16,
    execution_horizon=16,
    action_stride=2,
    reference_fps=30,
    future_image_offset=32,
    orientation_target="predicted_rotation6d",
):
    source_contract = plain_value(source_contract)
    if not isinstance(source_contract, dict):
        raise ValueError("Joint v2 requires an explicit reference source_contract")
    contract = validate_source_contract(
        source_contract, source_contract.get("body_target"), orientation_target
    )
    timing_v2(
        action_horizon=prediction_horizon,
        past_action_size=past_action_horizon,
        execution_horizon=execution_horizon,
        action_stride=action_stride,
        reference_fps=reference_fps,
        future_image_offset=future_image_offset,
    )
    return {
        "recipe_version": "v2",
        "prediction_horizon": prediction_horizon,
        "past_action_horizon": past_action_horizon,
        "execution_horizon": execution_horizon,
        "lookahead_tail_horizon": prediction_horizon - execution_horizon,
        "action_stride": action_stride,
        "reference_fps": reference_fps,
        "future_image_offset": future_image_offset,
        "source_history_rule": V2_SOURCE_HISTORY_RULE,
        "orientation_target": orientation_target,
        "orientation_prediction_frame": "current_heading_relative"
        if orientation_target == "predicted_rotation6d"
        else "external",
        "orientation_anchor_source_key": "observation.root_orientation",
        "orientation_anchor_convention": "world_wxyz_heading_about_z",
        "normalization_strategy": NORMALIZATION_V2,
        "reference_source_contract": contract,
        "reference_source_contract_sha256": source_contract_sha256(contract),
    }


def source_contract_sha256(contract):
    return hashlib.sha256(
        json.dumps(
            plain_value(contract),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def validate_source_contract(contract, body_target, orientation_target):
    contract = plain_value(contract)
    if not isinstance(contract, dict):
        raise ValueError(
            "Joint v2 requires an explicit reference source_contract; no source fallback"
        )
    required = {
        "schema",
        "body_target",
        "joint_position_key",
        "joint_position_indices",
        "joint_names",
        "joint_units",
        "root_quaternion_key",
        "root_quaternion_order",
        "root_frame",
        "root_validity_policy",
        "same_stage_pairing",
        "label_semantics",
        "provenance",
    }
    if (
        set(contract) != required
        or contract["schema"] != SOURCE_SCHEMA
        or contract["body_target"] != body_target
    ):
        raise ValueError("Reference source contract schema/fields/body target differ")
    semantics = {
        "direct_reference": "direct_joint_reference",
        "measured": "reference_from_demonstration",
        "wbc": "wbc_motor_qtarget_experiment",
    }
    if (
        body_target not in semantics
        or contract["label_semantics"] != semantics[body_target]
    ):
        raise ValueError(
            "Body label semantics must explicitly distinguish demonstration, motor target and direct reference"
        )
    indices, names = contract["joint_position_indices"], contract["joint_names"]
    if (
        not isinstance(indices, list)
        or len(indices) != 29
        or any(type(i) is not int or i < 0 for i in indices)
        or len(set(indices)) != 29
        or not isinstance(names, list)
        or len(names) != 29
        or any(not isinstance(s, str) or not s for s in names)
        or len(set(names)) != 29
    ):
        raise ValueError(
            "Explicit 29 unique source indices and joint names are required"
        )
    if (
        contract["joint_units"] != "rad"
        or contract["root_quaternion_order"] != "wxyz"
        or contract["root_frame"] != "world"
    ):
        raise ValueError(
            "Declare radians and world-frame wxyz root quaternions; no implicit conversion"
        )
    for key in ("joint_position_key", "root_quaternion_key"):
        if not isinstance(contract[key], str) or not contract[key].strip():
            raise ValueError("Reference field keys must be explicit nonempty strings")
    if contract["root_validity_policy"] not in (
        "teleop_recorded_flags",
        "finite_nonzero_quaternion",
    ):
        raise ValueError("Explicit root validity policy is required")
    if type(contract["same_stage_pairing"]) is not bool:
        raise ValueError("same_stage_pairing must be an explicit boolean")
    provenance = contract["provenance"]
    if (
        not isinstance(provenance, dict)
        or set(provenance)
        != {
            "description",
            "source",
            "revision",
            "coordinate_frame_verified",
            "synchronization_verified",
        }
        or any(
            not isinstance(provenance.get(k), str) or not provenance[k].strip()
            for k in ("description", "source", "revision")
        )
        or any(
            type(provenance.get(k)) is not bool
            for k in ("coordinate_frame_verified", "synchronization_verified")
        )
    ):
        raise ValueError(
            "Reference provenance requires source/revision/description and explicit verification flags"
        )
    if body_target in ("wbc", "measured"):
        expected = "action.wbc" if body_target == "wbc" else "observation.state"
        if contract["joint_position_key"] != expected or indices != list(
            range(22)
        ) + list(range(29, 36)):
            raise ValueError(
                "Experimental wbc/measured source indices cannot be silently reinterpreted"
            )
    else:
        if contract["joint_position_key"] in (
            "action.wbc",
            "observation.state",
            "action.motion_token",
        ):
            raise ValueError(
                "Decoder motor targets/measured joints/motion tokens are not direct reference labels"
            )
        if not (
            contract["same_stage_pairing"]
            and provenance["coordinate_frame_verified"]
            and provenance["synchronization_verified"]
        ):
            raise ValueError(
                "Direct references require verified same-stage body/root coordinates and synchronization"
            )
    if orientation_target not in ("external", "predicted_rotation6d"):
        raise ValueError("Unsupported orientation target")
    return copy.deepcopy(contract)


def deployment_ready(contract):
    """Source provenance alone is never a robot/controller validation claim."""
    p = contract["provenance"]
    return bool(
        contract["body_target"] == "direct_reference"
        and contract["same_stage_pairing"]
        and p["coordinate_frame_verified"]
        and p["synchronization_verified"]
    )


def timing_v2(
    *,
    action_horizon=40,
    past_action_size=16,
    execution_horizon=16,
    action_stride=2,
    future_image_offset=32,
    reference_fps=30,
):
    values = (
        action_horizon,
        past_action_size,
        execution_horizon,
        action_stride,
        future_image_offset,
        reference_fps,
    )
    if any(type(x) is not int for x in values) or values != (40, 16, 16, 2, 32, 30):
        raise ValueError(
            "Joint v2 requires prediction40/history16/execution16 at recorded30FPS/stride2 and image offset32"
        )
    dt = action_stride / reference_fps
    return {
        "recipe_version": "v2",
        "fps": reference_fps,
        "action_horizon": action_horizon,
        "prediction_horizon": action_horizon,
        "past_action_size": past_action_size,
        "past_action_horizon": past_action_size,
        "execution_horizon": execution_horizon,
        "lookahead_tail_horizon": action_horizon - execution_horizon,
        "source_history_rule": V2_SOURCE_HISTORY_RULE,
        "action_stride": action_stride,
        "future_image_offset": future_image_offset,
        "visual_prediction_role": "execution_segment_endpoint_not_lookahead_endpoint",
        "future_action_offsets": list(
            range(0, action_horizon * action_stride, action_stride)
        ),
        "past_action_offsets": list(
            range(-past_action_size * action_stride, 0, action_stride)
        ),
        "action_interval_seconds": dt,
        "future_action_relative_timestamps": [i * dt for i in range(action_horizon)],
        "future_action_span_seconds": (action_horizon - 1) * dt,
        "execution_reference_span_seconds": (execution_horizon - 1) * dt,
        "execution_segment_duration_seconds": execution_horizon * dt,
        "state_condition_time": "current anchor only",
        "future_state_in_model_input": False,
    }
