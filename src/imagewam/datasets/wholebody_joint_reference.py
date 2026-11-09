"""Read-only G1 motor-target datasets with measured Inspire hand targets.

The default body label is the 29 motor targets issued by the WBC decoder,
not the original teleoperation reference. Measured body joints are an explicit
alternative. The optional world orientation is a prediction target only.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from imagewam.joint_reference_contract import (
    validate_source_contract,
    source_contract_sha256,
    deployment_ready,
    timing_v2,
    NORMALIZATION_V2,
    make_joint_recipe,
)
from imagewam.joint_reference_rotation import (
    world_quaternion_to_heading_relative_rotation6d,
)

from .wholebody import (
    CAMERAS,
    DEFAULT_PROMPT,
    HAND_FIELD_ORDER,
    STATE_KEY,
    WholeBodyLeRobotDataset,
    _field_stats,
    all_training_episode_ids,
    decode_resized_frames,
    metadata_paths,
    sha256,
)


ACTION_KEY = "wholebody_joint_reference"
NORMALIZATION_STRATEGY = (
    "fit_joint_reference_body29_orientation6d_q01q99_fixed_inspire_hands"
)
BODY_INDICES = tuple(range(22)) + tuple(range(29, 36))
BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
ORIENTATION_SOURCE_KEY = "teleop.body_quat_w"
ORIENTATION_VALIDITY_KEYS = ("teleop.smpl_frame_index", "teleop.stream_mode")
ORIENTATION_VALIDITY_RULE = "smpl_frame_index>0_and_stream_mode_in_1_4"
ROTATION6D_ORDER = ("R00", "R10", "R20", "R01", "R11", "R21")
BODY_TARGETS = ("wbc", "measured", "direct_reference")
ORIENTATION_TARGETS = ("external", "predicted_rotation6d")
INDEX_COLUMNS = ("timestamp", "frame_index", "episode_index", "index", "task_index")
BASE_COLUMNS = (
    "observation.state",
    "observation.inspire_left_hand_state",
    "observation.inspire_right_hand_state",
    "observation.projected_gravity",
)


def action_dimension(orientation_target: str) -> int:
    if orientation_target not in ORIENTATION_TARGETS:
        raise ValueError(f"orientation_target must be one of {ORIENTATION_TARGETS}")
    return 41 if orientation_target == "external" else 47


def schema_for(
    body_target: str = "wbc",
    orientation_target: str = "external",
    recipe_version: str = "v1",
) -> str:
    if body_target not in BODY_TARGETS:
        raise ValueError(
            f"body_target must be one of {BODY_TARGETS}; missing WBC never falls back"
        )
    dim = action_dimension(orientation_target)
    suffix = "external-orientation" if dim == 41 else "quatwxyz-rotation6d-columns"
    if recipe_version not in ("v1", "v2") or (
        recipe_version == "v1" and body_target == "direct_reference"
    ):
        raise ValueError(
            "Direct references require the explicit v2 provenance contract"
        )
    body = {
        "wbc": "wbc29",
        "measured": "measured29",
        "direct_reference": "direct-reference29",
    }[body_target]
    return f"wholebody-state44-joint{dim}-{body}-measured-inspire12-{suffix}-{recipe_version}"


def _contract(body_target, orientation_target, recipe_version, source_contract):
    schema_for(body_target, orientation_target, recipe_version)
    if recipe_version == "v1":
        if source_contract is not None:
            raise ValueError("Legacy v1 cannot silently adopt a new source contract")
        return None
    result = validate_source_contract(source_contract, body_target, orientation_target)
    if tuple(result["joint_names"]) != BODY_JOINT_NAMES:
        raise ValueError(
            "Source contract joint_names must declare the canonical BODY_JOINT_NAMES order"
        )
    return result


def numeric_columns(
    body_target: str = "wbc",
    orientation_target: str = "external",
    *,
    recipe_version="v1",
    source_contract=None,
) -> tuple[str, ...]:
    contract = _contract(
        body_target, orientation_target, recipe_version, source_contract
    )
    if contract is not None:
        selected = (contract["joint_position_key"], "observation.root_orientation")
        orientation = (
            (contract["root_quaternion_key"],)
            if orientation_target == "predicted_rotation6d"
            else ()
        )
        if orientation and contract["root_validity_policy"] == "teleop_recorded_flags":
            orientation += ORIENTATION_VALIDITY_KEYS
        return tuple(
            dict.fromkeys(BASE_COLUMNS + selected + orientation + INDEX_COLUMNS)
        )
    body = ("action.wbc",) if body_target == "wbc" else ()
    orientation = (
        (ORIENTATION_SOURCE_KEY,) + ORIENTATION_VALIDITY_KEYS
        if orientation_target == "predicted_rotation6d"
        else ()
    )
    return BASE_COLUMNS + body + orientation + INDEX_COLUMNS


def orientation_validity(columns: dict, source_contract=None) -> np.ndarray:
    """Conservatively exclude recorder defaults and ambiguous SMPL frame0.

    The logged pose_sample_age_ms is not the receive-message age used by the
    recorder's100ms gate, so it must not be thresholded as that validity flag.
    A genuine identity rotation at a positive frame index remains valid.
    """
    n = len(columns["observation.state"])
    if (
        source_contract is not None
        and source_contract["root_validity_policy"] == "finite_nonzero_quaternion"
    ):
        quat = np.asarray(columns[source_contract["root_quaternion_key"]])
        if quat.shape != (n, 4):
            raise ValueError("Root source must provide a quaternion per recorded row")
        return np.isfinite(quat).all(axis=-1) & (np.linalg.norm(quat, axis=-1) > 1e-8)
    values = []
    for key in ORIENTATION_VALIDITY_KEYS:
        if key not in columns:
            raise ValueError(
                f"47D orientation requires recorded validity source: {key}"
            )
        a = np.asarray(columns[key])
        if a.shape == (n, 1):
            a = a[:, 0]
        if a.shape != (n,) or a.dtype.kind not in "iu":
            raise ValueError(
                f"Orientation validity source must contain integer scalar rows: {key}"
            )
        values.append(a)
    frame_index, stream_mode = values
    return ((frame_index > 0) & np.isin(stream_mode, (1, 4))).astype(bool)


def quaternion_wxyz_to_rotation6d(values: np.ndarray) -> np.ndarray:
    """Unit-normalize wxyz and concatenate the first two rotation columns.

    Output order is [R00,R10,R20,R01,R11,R21]. q and -q give the
    same result. This does not reinterpret the logged orientation6 field.
    """
    quat = np.asarray(values, dtype=np.float64)
    if quat.ndim < 1 or quat.shape[-1] != 4 or not np.isfinite(quat).all():
        raise ValueError("Expected finite world quaternions in wxyz order")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm <= 1e-8):
        raise ValueError("Zero/near-zero body quaternion cannot define a rotation")
    w, x, y, z = np.moveaxis(quat / norm, -1, 0)
    return np.stack((1 - 2 * (y*y + z*z), 2 * (x*y + w*z), 2 * (x*z - w*y),
                     2 * (x*y - w*z), 1 - 2 * (x*x + z*z), 2 * (y*z + w*x)), axis=-1).astype(np.float32)


def timing_configuration(
    action_horizon: int = 16,
    past_action_size: int = 16,
    action_stride: int = 2,
    future_image_offset: int | None = None,
    *,
    recipe_version="v1",
    execution_horizon=16,
    reference_fps=30,
) -> dict:
    if recipe_version == "v2":
        return timing_v2(
            action_horizon=action_horizon,
            past_action_size=past_action_size,
            execution_horizon=execution_horizon,
            action_stride=action_stride,
            future_image_offset=32
            if future_image_offset is None
            else future_image_offset,
            reference_fps=reference_fps,
        )
    if recipe_version != "v1":
        raise ValueError("Unknown joint recipe version")
    if any(
        type(x) is not int or x < 1
        for x in (action_horizon, past_action_size, action_stride)
    ):
        raise ValueError(
            "Horizon, history length and action_stride must be positive integers"
        )
    if future_image_offset is None:
        future_image_offset = action_horizon * action_stride
    if type(future_image_offset) is not int or future_image_offset < 1:
        raise ValueError("future_image_offset must be a positive integer")
    return {
        "fps": 30,
        "action_horizon": action_horizon,
        "past_action_size": past_action_size,
        "action_stride": action_stride,
        "future_image_offset": future_image_offset,
        "future_action_offsets": list(
            range(0, action_horizon * action_stride, action_stride)
        ),
        "past_action_offsets": list(
            range(-past_action_size * action_stride, 0, action_stride)
        ),
        "action_interval_seconds": action_stride / 30.0,
        "future_action_span_seconds": (action_horizon - 1) * action_stride / 30.0,
        "state_condition_time": "current anchor only",
        "future_state_in_model_input": False,
    }


def load_metadata(
    dataset_dir: str | Path,
    body_target: str = "wbc",
    orientation_target: str = "external",
    *,
    recipe_version="v1",
    source_contract=None,
) -> dict:
    """Validate names and selected sources without requiring unused motion tokens."""
    contract = _contract(
        body_target, orientation_target, recipe_version, source_contract
    )
    root = Path(dataset_dir).resolve()
    metadata_paths(root)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [
        json.loads(x)
        for x in (root / "meta/episodes.jsonl").read_text().splitlines()
        if x.strip()
    ]
    tasks = [
        json.loads(x)
        for x in (root / "meta/tasks.jsonl").read_text().splitlines()
        if x.strip()
    ]
    episode_stats = [
        json.loads(x)
        for x in (root / "meta/episodes_stats.jsonl").read_text().splitlines()
        if x.strip()
    ]
    ids = [int(e["episode_index"]) for e in episodes]
    count, total = int(info["total_episodes"]), int(info["total_frames"])
    if (
        info["codebase_version"] != "v2.1"
        or float(info["fps"]) != 30.0
        or count < 1
        or total < 1
        or ids != list(range(count))
        or info.get("splits") != {"train": f"0:{count}"}
        or int(info["chunks_size"]) != 1000
    ):
        raise ValueError(
            "Expected contiguous all-training LeRobot v2.1 / 30 FPS collection"
        )
    if (
        len(tasks) != 1
        or tasks[0].get("task_index") != 0
        or not isinstance(tasks[0].get("task"), str)
        or not tasks[0]["task"].strip()
    ):
        raise ValueError("Expected one recorded nonempty instruction with task_index=0")
    if [int(s["episode_index"]) for s in episode_stats] != ids:
        raise ValueError("Episode statistics IDs differ from the curated episodes")
    if (
        any(int(e["length"]) < 1 for e in episodes)
        or sum(int(e["length"]) for e in episodes) != total
    ):
        raise ValueError("Metadata episode/frame count mismatch")
    features = info["features"]
    dims = {
        "observation.state": 43,
        "observation.inspire_left_hand_state": 6,
        "observation.inspire_right_hand_state": 6,
        "observation.projected_gravity": 3,
    }
    if body_target == "wbc":
        dims["action.wbc"] = 43
    if contract is not None:
        dims["observation.root_orientation"] = 4
        selected = contract["joint_position_key"]
        feature = features.get(selected, {})
        shape = feature.get("shape", [])
        if (
            len(shape) != 1
            or type(shape[0]) is not int
            or shape[0] <= max(contract["joint_position_indices"])
        ):
            raise ValueError(
                "Declared reference joint source is missing or cannot provide all29 indexed joints"
            )
        dims[selected] = shape[0]
    if orientation_target == "predicted_rotation6d":
        dims[
            contract["root_quaternion_key"] if contract else ORIENTATION_SOURCE_KEY
        ] = 4
        if (
            contract is None
            or contract["root_validity_policy"] == "teleop_recorded_flags"
        ):
            dims.update({key: 1 for key in ORIENTATION_VALIDITY_KEYS})
    for key, dim in dims.items():
        if key not in features or features[key].get("shape") != [dim]:
            raise ValueError(
                f"Required feature missing/wrong shape: {key}; no implicit source fallback"
            )
    for key in ("observation.state",) + (
        ("action.wbc",) if body_target == "wbc" else ()
    ):
        names = features[key].get("names")
        if (
            not isinstance(names, list)
            or len(names) != 43
            or tuple(names[i] for i in BODY_INDICES) != BODY_JOINT_NAMES
        ):
            raise ValueError(f"Expected explicit canonical motor names/order in {key}")
    if contract is not None:
        names = features[contract["joint_position_key"]].get("names")
        if (
            not isinstance(names, list)
            or tuple(names[i] for i in contract["joint_position_indices"])
            != BODY_JOINT_NAMES
        ):
            raise ValueError(
                "Declared source joint names/order differ from the reference contract"
            )
    for side in ("left", "right"):
        key = f"observation.inspire_{side}_hand_state"
        names = tuple(features[key].get("names", ()))
        if names not in (
            HAND_FIELD_ORDER,
            tuple(f"{side}_{name}" for name in HAND_FIELD_ORDER),
        ):
            raise ValueError(f"Wrong/missing measured Inspire field order: {key}")
    for camera in CAMERAS:
        shape = features.get(camera, {}).get("shape", [])
        if (
            len(shape) != 3
            or shape[-1] != 3
            or features[camera].get("dtype") != "video"
            or any(type(x) is not int or x < 1 for x in shape)
        ):
            raise ValueError(f"Wrong RGB camera specification: {camera}")
    if info.get("total_videos") != count * len(CAMERAS):
        raise ValueError("Wrong recorded video count")
    for episode, stats in zip(episodes, episode_stats):
        for key, value in stats["stats"].items():
            if "count" in value and np.any(
                np.asarray(value["count"]) != int(episode["length"])
            ):
                raise ValueError(
                    f"Wrong statistics count: episode{episode['episode_index']} {key}"
                )
    return {
        "root": root,
        "info": info,
        "episodes": episodes,
        "tasks": tasks,
        "train": ids,
        "validation": [],
        "split": {"train_episode_ids": ids, "validation_episode_ids": []},
    }


def map_numeric(
    columns: dict,
    body_target: str = "wbc",
    orientation_target: str = "external",
    *,
    recipe_version="v1",
    source_contract=None,
) -> tuple[np.ndarray, np.ndarray]:
    contract = _contract(
        body_target, orientation_target, recipe_version, source_contract
    )
    n = len(columns["observation.state"])
    dims = {
        "observation.state": 43,
        "observation.inspire_left_hand_state": 6,
        "observation.inspire_right_hand_state": 6,
        "observation.projected_gravity": 3,
    }
    if body_target == "wbc":
        dims["action.wbc"] = 43
    if contract is not None:
        key = contract["joint_position_key"]
        if key not in columns:
            raise ValueError(
                f"Declared joint source missing: {key}; no source fallback"
            )
        a = np.asarray(columns[key])
        if (
            a.ndim != 2
            or a.shape[0] != n
            or a.shape[1] <= max(contract["joint_position_indices"])
        ):
            raise ValueError(
                "Declared joint reference source cannot provide all29 joints"
            )
        dims[key] = a.shape[1]
    if orientation_target == "predicted_rotation6d":
        dims[
            contract["root_quaternion_key"] if contract else ORIENTATION_SOURCE_KEY
        ] = 4
    arrays = {}
    for key, dim in dims.items():
        if key not in columns:
            raise ValueError(
                f"Required numeric source missing: {key}; no implicit source fallback"
            )
        a = np.asarray(columns[key], dtype=np.float32)
        orientation_key = (
            contract["root_quaternion_key"] if contract else ORIENTATION_SOURCE_KEY
        )
        is_maskable_root = contract is not None and key == orientation_key
        if a.shape != (n, dim) or (not is_maskable_root and not np.isfinite(a).all()):
            raise ValueError(f"Wrong shape/non-finite values in {key}: {a.shape}")
        if "inspire_" in key and np.any((a < 0) | (a > 1)):
            raise ValueError(f"Measured Inspire feedback must be in [0,1]: {key}")
        arrays[key] = a
    left, right = arrays[BASE_COLUMNS[1]], arrays[BASE_COLUMNS[2]]
    state = np.concatenate(
        (
            arrays["observation.state"][:, BODY_INDICES],
            left,
            right,
            arrays["observation.projected_gravity"],
        ),
        axis=-1,
    )
    source = (
        contract["joint_position_key"]
        if contract
        else ("action.wbc" if body_target == "wbc" else "observation.state")
    )
    indices = contract["joint_position_indices"] if contract else BODY_INDICES
    parts = [arrays[source][:, indices], left, right]
    if orientation_target == "predicted_rotation6d":
        valid = orientation_validity(columns, contract)
        quat = arrays[
            contract["root_quaternion_key"] if contract else ORIENTATION_SOURCE_KEY
        ].copy()
        if contract is not None:
            quat[~valid] = [
                1,
                0,
                0,
                0,
            ]  # Numerically safe placeholder, subsequently zeroed/masked.
        parts.append(quaternion_wxyz_to_rotation6d(quat))
    return state, np.concatenate(parts, axis=-1)


def read_episode(
    root: Path,
    episode_id: int,
    expected_length: int,
    expected_global_start: int | None = None,
    *,
    body_target: str = "wbc",
    orientation_target: str = "external",
    recipe_version="v1",
    source_contract=None,
) -> tuple[np.ndarray, ...]:
    import pyarrow.parquet as pq

    path = (
        root / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
    )
    columns = pq.read_table(
        path,
        columns=list(
            numeric_columns(
                body_target,
                orientation_target,
                recipe_version=recipe_version,
                source_contract=source_contract,
            )
        ),
        use_threads=False,
    ).to_pydict()
    n = len(columns["frame_index"])
    if n != expected_length or not np.array_equal(columns["frame_index"], np.arange(n)):
        raise ValueError(f"Nonsequential/mismatched frame indices: episode{episode_id}")
    if not np.all(np.asarray(columns["episode_index"]) == episode_id):
        raise ValueError(f"Cross-episode numeric data: episode{episode_id}")
    if not np.all(np.asarray(columns["task_index"]) == 0):
        raise ValueError(f"Unexpected instruction ID: episode{episode_id}")
    if not np.allclose(columns["timestamp"], np.arange(n) / 30.0, rtol=0, atol=3e-6):
        raise ValueError(f"Timestamp/frame alignment mismatch: episode{episode_id}")
    if expected_global_start is not None and not np.array_equal(
        columns["index"], np.arange(expected_global_start, expected_global_start + n)
    ):
        raise ValueError(f"Global index mismatch: episode{episode_id}")
    state, action = map_numeric(
        columns,
        body_target,
        orientation_target,
        recipe_version=recipe_version,
        source_contract=source_contract,
    )
    if orientation_target == "predicted_rotation6d":
        valid = orientation_validity(columns, source_contract)
        if recipe_version == "v2":
            anchor = np.asarray(
                columns["observation.root_orientation"], dtype=np.float32
            )
            if (
                anchor.shape != (n, 4)
                or not np.isfinite(anchor).all()
                or np.any(np.linalg.norm(anchor, axis=-1) < 1e-8)
            ):
                raise ValueError(
                    "v2 heading anchors require valid CURRENT measured root orientations"
                )
            root_quat = np.asarray(
                columns[source_contract["root_quaternion_key"]], dtype=np.float32
            )
            root_quat[~valid] = [1, 0, 0, 0]
            return state, action, valid, anchor, root_quat
        return state, action, valid
    return state, action


def window_indices(
    anchor: int,
    length: int,
    horizon: int = 16,
    history: int = 16,
    action_stride: int = 2,
    future_image_offset: int | None = None,
    *,
    recipe_version="v1",
    execution_horizon=16,
) -> dict:
    timing = timing_configuration(
        horizon,
        history,
        action_stride,
        future_image_offset,
        recipe_version=recipe_version,
        execution_horizon=execution_horizon,
    )
    if type(anchor) is not int or type(length) is not int or not 0 <= anchor < length:
        raise ValueError("Invalid episode anchor/length")
    future = anchor + np.asarray(timing["future_action_offsets"], dtype=np.int64)
    past = anchor + np.asarray(timing["past_action_offsets"], dtype=np.int64)
    image = anchor + timing["future_image_offset"]
    return {
        "action": np.minimum(future, length - 1),
        "action_is_pad": future >= length,
        "past_action": np.maximum(past, 0),
        "past_action_is_pad": past < 0,
        "image": np.asarray([anchor, min(image, length - 1)]),
        "image_is_pad": np.asarray([False, image >= length]),
    }


def fit_statistics(
    episode_arrays: dict[int, tuple[np.ndarray, ...]],
    train_ids: list[int],
    *,
    body_target: str = "wbc",
    orientation_target: str = "external",
    action_horizon: int = 16,
    past_action_size: int = 16,
    action_stride: int = 2,
    future_image_offset: int | None = None,
    recipe_version="v1",
    execution_horizon=16,
    source_contract=None,
) -> dict:
    contract = _contract(
        body_target, orientation_target, recipe_version, source_contract
    )
    schema = schema_for(body_target, orientation_target, recipe_version)
    dim = action_dimension(orientation_target)
    timing = timing_configuration(
        action_horizon,
        past_action_size,
        action_stride,
        future_image_offset,
        recipe_version=recipe_version,
        execution_horizon=execution_horizon,
    )
    if (
        train_ids != sorted(episode_arrays)
        or len(set(train_ids)) != len(train_ids)
        or not train_ids
    ):
        raise ValueError(
            "Statistics must use every episode exactly once in sorted order"
        )
    state = np.concatenate([episode_arrays[i][0] for i in train_ids])
    action = np.concatenate([episode_arrays[i][1] for i in train_ids])
    if (
        state.shape != (len(action), 44)
        or action.shape[1:] != (dim,)
        or not np.isfinite(state).all()
        or not np.isfinite(action).all()
    ):
        raise ValueError(f"Expected finite state44/action{dim}")
    if not np.array_equal(state[:, 29:41], action[:, 29:41]) or np.any(
        (action[:, 29:41] < 0) | (action[:, 29:41] > 1)
    ):
        raise ValueError(
            "State/action hands must share same-row physical measured feedback"
        )
    if body_target == "measured" and not np.array_equal(state[:, :29], action[:, :29]):
        raise ValueError(
            "Measured body ablation must use observation.state as its body labels"
        )
    action_stats = _field_stats(action)
    std = np.std(action.astype(np.float64), axis=0)
    valid_orientation = None
    if dim == 47:
        if any(
            len(episode_arrays[i]) != (5 if recipe_version == "v2" else 3)
            for i in train_ids
        ):
            raise ValueError(
                "47D statistics require explicit per-row orientation validity flags"
            )
        valid_orientation = np.concatenate([episode_arrays[i][2] for i in train_ids])
        if (
            valid_orientation.shape != (len(action),)
            or valid_orientation.dtype != bool
            or not valid_orientation.any()
        ):
            raise ValueError(
                "47D statistics require a boolean validity mask and at least one valid orientation"
            )
        valid_stats = _field_stats(action[valid_orientation, 41:47])
        for key in action_stats:
            action_stats[key][41:47] = valid_stats[key]
        std[41:47] = np.std(action[valid_orientation, 41:47].astype(np.float64), axis=0)
    result = {
        "schema": schema,
        "action": {ACTION_KEY: action_stats},
        "state": {STATE_KEY: _field_stats(state)},
        "action_dim": dim,
        "proprio_dim": 44,
        "body_target": body_target,
        "orientation_target": orientation_target,
        "body_target_source": "action.wbc"
        if body_target == "wbc"
        else "observation.state",
        "body_target_semantics": "WBC decoder-issued motor targets, not original teleop reference"
        if body_target == "wbc"
        else "measured body joint positions",
        "body_joint_names": list(BODY_JOINT_NAMES),
        "body_state_indices": list(BODY_INDICES),
        "body_action_indices": list(BODY_INDICES),
        "body_dataset_units": "radians",
        "orientation_source_key": ORIENTATION_SOURCE_KEY if dim == 47 else None,
        "orientation_input_order": "wxyz" if dim == 47 else None,
        "orientation_target_frame": "world" if dim == 47 else None,
        "orientation_representation": "first_two_columns"
        if dim == 47
        else "external_not_predicted",
        "rotation6d_order": list(ROTATION6D_ORDER) if dim == 47 else [],
        "orientation_conversion": "normalize quaternion; standard active rotation matrix; concatenate first two columns"
        if dim == 47
        else None,
        "orientation_condition_in_model_input": False,
        "action_order": "body29,left_measured_hand6_native_order,right_measured_hand6_native_order"
        + (",rotation6d_columns" if dim == 47 else ""),
        "state_order": "body29,left_measured_hand6_native_order,right_measured_hand6_native_order,projected_gravity3",
        "hand_target_time_offset": 0,
        "body_target_time_offset": 0,
        "orientation_target_time_offset": 0 if dim == 47 else None,
        "hand_target_type": "same-row measured feedback, not issued hand command",
        "hand_field_order": list(HAND_FIELD_ORDER),
        "hand_dataset_units": "[0,1]; do not divide feedback by 1000 again",
        "hand_deployment_raw1000_rule": "if runtime reports raw [0,1000], divide by1000 exactly once before normalization",
        "normalization": {
            "mode": "q01/q99",
            "clamp": [-5.0, 5.0],
            "state_fixed_bounds": {"start": 29, "end": 41, "min": 0.0, "max": 1.0},
            "action_fixed_bounds": {"start": 29, "end": 41, "min": 0.0, "max": 1.0},
            "constant_range_tolerance": 1e-4,
        },
        "normalization_strategy": NORMALIZATION_STRATEGY,
        "train_episode_ids": list(train_ids),
        "normalization_reference_episode_ids": list(train_ids),
        "train_frames": int(len(action)),
        "normalization_reference_train_frames": int(len(action)),
        "train_split_mode": "all_episodes",
        "heldout_episode_ids": [],
        "constant_action_dims": np.flatnonzero(std < 1e-8).tolist(),
        "timing": timing,
        **timing,
    }
    if dim == 47:
        result.update(
            orientation_validity_source_keys=list(ORIENTATION_VALIDITY_KEYS),
            orientation_validity_rule=ORIENTATION_VALIDITY_RULE,
            orientation_invalid_policy="mask_only_last6_future_and_zero_last6_normalized_future_history",
            orientation_statistics_valid_frames=int(valid_orientation.sum()),
            orientation_statistics_invalid_frames=int((~valid_orientation).sum()),
            orientation_statistics_use_valid_rows_only=True,
            orientation_validity_rejects_missing_flags=True,
            orientation_age_threshold_applied=False,
            orientation_validity_caveat="Conservatively masks SMPL frame0 even if a genuine first sample; logged sample age differs from receive age",
        )
    if recipe_version == "v2":
        root_key = contract["root_quaternion_key"]
        policy = contract["root_validity_policy"]
        result.update(
            recipe_version="v2",
            reference_source_contract=contract,
            reference_source_contract_sha256=source_contract_sha256(contract),
            joint_recipe=make_joint_recipe(
                contract, orientation_target=orientation_target
            ),
            normalization_strategy=NORMALIZATION_V2,
            body_target_source=contract["joint_position_key"],
            body_target_semantics=contract["label_semantics"],
            source_provenance_deployment_ready=deployment_ready(contract),
            original_sonic_reference_equivalence_verified=False,
            orientation_prediction_frame="current_heading_relative"
            if dim == 47
            else "external",
            orientation_anchor_source_key="observation.root_orientation",
            orientation_anchor_convention="world_wxyz_heading_about_z",
        )
        if dim == 47:
            result.update(
                orientation_source_key=root_key,
                orientation_target_frame="current_heading_relative",
                orientation_validity_rule=policy,
                orientation_validity_source_keys=list(ORIENTATION_VALIDITY_KEYS)
                if policy == "teleop_recorded_flags"
                else [root_key],
                orientation_validity_caveat="Validity is declared by this source contract; current heading anchors must be finite nonzero measured root quaternions",
                orientation_statistics_mode="fixed_minus1_plus1_heading_relative_rotation6d",
                orientation_condition_in_model_input=False,
            )
            for key, values in action_stats.items():
                values[41:47] = (
                    [-1.0] * 6
                    if key in ("global_min", "global_q01")
                    else [1.0] * 6
                    if key in ("global_max", "global_q99", "global_std")
                    else [0.0] * 6
                )
    return result


class JointReferenceNormalizer:
    """Collection-specific q01/q99 body/orientation, fixed [0,1] measured hands."""

    def __init__(self, stats: dict):
        body = stats.get("body_target")
        orientation = stats.get("orientation_target")
        recipe_version = stats.get("recipe_version", "v1")
        contract = _contract(
            body, orientation, recipe_version, stats.get("reference_source_contract")
        )
        dim = action_dimension(orientation)
        normalization = (
            NORMALIZATION_V2 if recipe_version == "v2" else NORMALIZATION_STRATEGY
        )
        if (
            stats.get("schema") != schema_for(body, orientation, recipe_version)
            or stats.get("action_dim") != dim
            or stats.get("proprio_dim") != 44
            or stats.get("train_split_mode") != "all_episodes"
            or stats.get("normalization_strategy") != normalization
            or stats.get("hand_target_time_offset") != 0
            or stats.get("body_target_time_offset") != 0
            or stats.get("orientation_condition_in_model_input") is not False
        ):
            raise ValueError(
                "Unsupported joint-reference normalization schema/source/dimensions"
            )
        root_key = (
            contract["root_quaternion_key"] if contract else ORIENTATION_SOURCE_KEY
        )
        validity_rule = (
            contract["root_validity_policy"] if contract else ORIENTATION_VALIDITY_RULE
        )
        validity_keys = (
            [root_key]
            if contract and validity_rule == "finite_nonzero_quaternion"
            else list(ORIENTATION_VALIDITY_KEYS)
        )
        if dim == 47 and (
            stats.get("orientation_source_key") != root_key
            or stats.get("rotation6d_order") != list(ROTATION6D_ORDER)
            or stats.get("orientation_input_order") != "wxyz"
            or stats.get("orientation_validity_rule") != validity_rule
            or stats.get("orientation_validity_source_keys") != validity_keys
            or stats.get("orientation_statistics_use_valid_rows_only") is not True
        ):
            raise ValueError("Incompatible quaternion/rotation6d convention")
        if contract is not None:
            if (
                stats.get("reference_source_contract_sha256")
                != source_contract_sha256(contract)
                or stats.get("joint_recipe")
                != make_joint_recipe(contract, orientation_target=orientation)
                or stats.get("orientation_prediction_frame")
                != ("current_heading_relative" if dim == 47 else "external")
            ):
                raise ValueError(
                    "Source/heading/prediction recipe contract changed after fitting"
                )
            if dim == 47:
                values = stats["action"][ACTION_KEY]
                if (
                    values["global_q01"][41:47] != [-1.0] * 6
                    or values["global_q99"][41:47] != [1.0] * 6
                    or stats.get("orientation_statistics_mode")
                    != "fixed_minus1_plus1_heading_relative_rotation6d"
                ):
                    raise ValueError(
                        "Heading-relative rotation6D requires explicitly fixed[-1,1] normalization"
                    )
        self.stats, self.action_dim, self.transforms = stats, dim, {}
        for field, key, dimension in (
            ("action", ACTION_KEY, dim),
            ("state", STATE_KEY, 44),
        ):
            values = stats[field][key]
            lower = np.asarray(values["global_q01"], dtype=np.float32).copy()
            upper = np.asarray(values["global_q99"], dtype=np.float32).copy()
            if (
                lower.shape != (dimension,)
                or upper.shape != (dimension,)
                or not np.isfinite(lower).all()
                or not np.isfinite(upper).all()
                or np.any(upper < lower)
            ):
                raise ValueError(f"Invalid normalization quantiles: {field}")
            lower[29:41], upper[29:41] = 0.0, 1.0
            if recipe_version == "v2" and field == "action" and dim == 47:
                lower[41:47], upper[41:47] = -1.0, 1.0
            span = upper - lower
            constant = span < 1e-4
            span[constant] = 2.0
            scale = 2.0 / span
            offset = -1.0 - scale * lower
            offset[constant] = -lower[constant]
            self.transforms[field] = (scale, offset)

    def forward(self, values: np.ndarray, field: str) -> np.ndarray:
        scale, offset = self.transforms[field]
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1:] != scale.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid {field} normalization input")
        return np.clip(values * scale + offset, -5.0, 5.0).astype(np.float32)

    def backward(self, values: np.ndarray, field: str = "action") -> np.ndarray:
        scale, offset = self.transforms[field]
        values = np.asarray(values, dtype=np.float32)
        if values.shape[-1:] != scale.shape or not np.isfinite(values).all():
            raise ValueError(f"Invalid {field} denormalization input")
        return ((values - offset) / scale).astype(np.float32)


def prepare_statistics(
    dataset_dir: str,
    output_dir: str,
    train_split_mode: str = "all_episodes",
    *,
    body_target: str = "wbc",
    orientation_target: str = "external",
    action_horizon: int = 16,
    past_action_size: int = 16,
    action_stride: int = 2,
    future_image_offset: int | None = None,
    recipe_version="v1",
    execution_horizon=16,
    source_contract=None,
) -> dict:
    contract = _contract(
        body_target, orientation_target, recipe_version, source_contract
    )
    meta = load_metadata(
        dataset_dir,
        body_target,
        orientation_target,
        recipe_version=recipe_version,
        source_contract=contract,
    )
    root, output = meta["root"], Path(output_dir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError(
            "Statistics outputs must be independent of the read-only dataset"
        )
    if any(
        (output / name).exists()
        for name in ("dataset_stats.json", "dataset_audit.json")
    ):
        raise FileExistsError("Refusing to overwrite existing statistics/audit")
    before = {name: sha256(root / name) for name in metadata_paths(root)}
    ids = all_training_episode_ids(meta, train_split_mode)
    arrays, hashes, start = {}, {}, 0
    for episode in meta["episodes"]:
        ep, n = int(episode["episode_index"]), int(episode["length"])
        relative = f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        digest = sha256(root / relative)
        arrays[ep] = read_episode(
            root,
            ep,
            n,
            start,
            body_target=body_target,
            orientation_target=orientation_target,
            recipe_version=recipe_version,
            source_contract=contract,
        )
        if sha256(root / relative) != digest:
            raise ValueError(f"Parquet changed during preparation: {relative}")
        hashes[relative] = digest
        start += n
        for camera in CAMERAS:
            video = (
                root / f"videos/chunk-{ep // 1000:03d}/{camera}/episode_{ep:06d}.mp4"
            )
            if not video.is_file() or video.stat().st_size == 0:
                raise FileNotFoundError(video)
    stats = fit_statistics(
        arrays,
        ids,
        body_target=body_target,
        orientation_target=orientation_target,
        action_horizon=action_horizon,
        past_action_size=past_action_size,
        action_stride=action_stride,
        future_image_offset=future_image_offset,
        recipe_version=recipe_version,
        execution_horizon=execution_horizon,
        source_contract=contract,
    )
    stats.update(
        dataset_root=str(root),
        instruction=meta["tasks"][0]["task"],
        metadata_sha256=before,
        parquet_sha256=hashes,
        source_feature_names={
            key: meta["info"]["features"][key].get("names")
            for key in numeric_columns(
                body_target,
                orientation_target,
                recipe_version=recipe_version,
                source_contract=contract,
            )
            if key not in INDEX_COLUMNS
        },
    )
    if before != {
        name: sha256(root / name) for name in metadata_paths(root)
    } or hashes != {name: sha256(root / name) for name in hashes}:
        raise ValueError("Dataset changed during statistics preparation")
    audit = {
        "schema": "wholebody-joint-reference-all-training-audit-v1",
        "dataset_root": str(root),
        "raw_dataset_read_only": True,
        "source_unchanged_during_preparation": True,
        "episodes": len(ids),
        "total_frames": start,
        "train_frames": start,
        "validation_frames": 0,
        "train_episodes": ids,
        "validation_episodes": [],
        "train_split_mode": "all_episodes",
        "metadata_sha256": before,
        "parquet_sha256": hashes,
        "normalization_strategy": stats["normalization_strategy"],
        "normalization_inherited_from_other_task": False,
        "numeric_dimensions_time_and_finite_checks": "PASS",
        "instruction": stats["instruction"],
        "body_target": body_target,
        "body_target_source": stats["body_target_source"],
        "body_target_semantics": stats["body_target_semantics"],
        "orientation_target": orientation_target,
        "orientation_source_key": stats["orientation_source_key"],
        "rotation6d_order": stats["rotation6d_order"],
        "action_dim": stats["action_dim"],
        "proprio_dim": 44,
        "timing": stats["timing"],
        "future_state_in_model_input": False,
        "orientation_condition_in_model_input": False,
        "cameras": list(CAMERAS),
        "output_video_resolution": [224, 448],
        "raw_video_resolutions": {
            cam: meta["info"]["features"][cam]["shape"][:2] for cam in CAMERAS
        },
        "episode_boundary_policy": "same-episode nearest target padding; normalized zero history padding; masks retained",
        "tail_samples_are_complete_deployment_futures": False,
        "per_channel_hardware_synchronization_verified": False,
        "action_source_noise_mask": [True] * stats["action_dim"],
        "video_predecode": False,
    }
    if orientation_target == "predicted_rotation6d":
        audit.update(
            {
                key: stats[key]
                for key in (
                    "orientation_validity_source_keys",
                    "orientation_validity_rule",
                    "orientation_invalid_policy",
                    "orientation_statistics_valid_frames",
                    "orientation_statistics_invalid_frames",
                    "orientation_statistics_use_valid_rows_only",
                    "orientation_validity_rejects_missing_flags",
                    "orientation_age_threshold_applied",
                    "orientation_validity_caveat",
                )
            }
        )
    if recipe_version == "v2":
        audit.update(
            recipe_version="v2",
            reference_source_contract=contract,
            reference_source_contract_sha256=source_contract_sha256(contract),
            joint_recipe=stats["joint_recipe"],
            source_provenance_deployment_ready=stats[
                "source_provenance_deployment_ready"
            ],
            original_sonic_reference_equivalence_verified=False,
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset_stats.json").open("x") as stream:
        json.dump(stats, stream, indent=2, allow_nan=False)
    audit["statistics_sha256"] = sha256(output / "dataset_stats.json")
    with (output / "dataset_audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    return audit


class JointReferenceLeRobotDataset(WholeBodyLeRobotDataset):
    """Same RGB/LRU as action76, with separately validated target/timing schema."""

    def __init__(
        self,
        dataset_dir: str,
        stats_path: str,
        qwen_text_cache_dir: str,
        split: str = "train",
        action_horizon: int = 16,
        past_action_size: int = 16,
        qwen_context_len: int = 128,
        qwen_text_dim: int = 7680,
        video_cache_frames: int = 64,
        decoder: Callable | None = None,
        train_split_mode: str = "all_episodes",
        body_target: str = "wbc",
        orientation_target: str = "external",
        action_stride: int = 2,
        future_image_offset: int | None = None,
        recipe_version="v1",
        execution_horizon=16,
        source_contract=None,
        augmentation=None,
    ):
        if split != "train":
            raise ValueError("This all-episode version has no validation dataset")
        self._configure_augmentation(augmentation, split=split)
        self.recipe_version = recipe_version
        self.source_contract = _contract(
            body_target, orientation_target, recipe_version, source_contract
        )
        self.timing = timing_configuration(
            action_horizon,
            past_action_size,
            action_stride,
            future_image_offset,
            recipe_version=recipe_version,
            execution_horizon=execution_horizon,
        )
        self.meta = load_metadata(
            dataset_dir,
            body_target,
            orientation_target,
            recipe_version=recipe_version,
            source_contract=self.source_contract,
        )
        self.root, self.body_target, self.orientation_target = (
            self.meta["root"],
            body_target,
            orientation_target,
        )
        self.stats = json.loads(Path(stats_path).read_text())
        self.normalizer = JointReferenceNormalizer(self.stats)
        if (
            self.stats.get("schema")
            != schema_for(body_target, orientation_target, recipe_version)
            or self.stats.get("timing") != self.timing
            or self.stats.get("reference_source_contract") != self.source_contract
            or any(self.stats.get(k) != v for k, v in self.timing.items())
        ):
            raise ValueError(
                "Statistics target sources/timing do not match requested training configuration"
            )
        expected = all_training_episode_ids(self.meta, train_split_mode)
        frames = sum(int(e["length"]) for e in self.meta["episodes"])
        if (
            self.stats.get("train_episode_ids") != expected
            or self.stats.get("heldout_episode_ids") != []
            or self.stats.get("normalization_reference_episode_ids") != expected
            or self.stats.get("train_frames") != frames
            or self.stats.get("normalization_reference_train_frames") != frames
            or self.stats.get("instruction") != self.meta["tasks"][0]["task"]
        ):
            raise ValueError(
                "Statistics must be independently fit from this complete collection"
            )
        if set(self.stats.get("metadata_sha256", {})) != set(metadata_paths(self.root)):
            raise ValueError("Dataset metadata file set changed after preparation")
        for relative, digest in self.stats["metadata_sha256"].items():
            if sha256(self.root / relative) != digest:
                raise ValueError(
                    f"Dataset metadata changed after preparation: {relative}"
                )
        self.episode_ids, self.episodes, lengths = expected, {}, []
        expected_files = {
            f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet" for ep in expected
        }
        if set(self.stats.get("parquet_sha256", {})) != expected_files:
            raise ValueError(
                "Statistics must retain the complete collection parquet SHA manifest"
            )
        start = 0
        for entry in self.meta["episodes"]:
            ep, n = int(entry["episode_index"]), int(entry["length"])
            relative = f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
            digest = self.stats["parquet_sha256"][relative]
            if sha256(self.root / relative) != digest:
                raise ValueError(
                    f"Parquet changed after statistics preparation: {relative}"
                )
            self.episodes[ep] = read_episode(
                self.root,
                ep,
                n,
                start,
                body_target=body_target,
                orientation_target=orientation_target,
                recipe_version=recipe_version,
                source_contract=self.source_contract,
            )
            if sha256(self.root / relative) != digest:
                raise ValueError(f"Parquet changed during dataset loading: {relative}")
            start += n
            lengths.append(n)
        self.ends = np.cumsum(lengths)
        self.action_horizon, self.past_action_size = action_horizon, past_action_size
        self.action_dim = action_dimension(orientation_target)
        self.prompt = DEFAULT_PROMPT.format(task=self.meta["tasks"][0]["task"])
        suffix = (
            hashlib.sha256(self.prompt.encode()).hexdigest()
            + f".qwen3_flux2_len{qwen_context_len}.pt"
        )
        payload = torch.load(
            Path(qwen_text_cache_dir) / suffix, map_location="cpu", weights_only=True
        )
        self.text_hidden_states = payload["text_hidden_states"]
        self.text_attention_mask = payload["text_attention_mask"].bool()
        if (
            self.text_hidden_states.shape != (qwen_context_len, qwen_text_dim)
            or self.text_attention_mask.shape != (qwen_context_len,)
            or not torch.isfinite(self.text_hidden_states).all()
            or not self.text_attention_mask.any()
        ):
            raise ValueError("Invalid instruction-only text cache")
        self.video_cache_frames = int(video_cache_frames)
        if self.video_cache_frames < 0:
            raise ValueError("video_cache_frames must be nonnegative")
        self.frame_cache = OrderedDict()
        self.decoder = decoder or decode_resized_frames

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        position = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episode_ids[position]
        anchor = int(index - (self.ends[position - 1] if position else 0))
        arrays = self.episodes[episode]
        states, actions = arrays[:2]
        window = window_indices(
            anchor,
            len(actions),
            self.action_horizon,
            self.past_action_size,
            self.timing["action_stride"],
            self.timing["future_image_offset"],
            recipe_version=self.recipe_version,
            execution_horizon=self.timing.get("execution_horizon", 16),
        )
        generator = self._augmentation_generator()
        visual_parameters = self._training_visual_parameters(generator=generator)
        video = torch.cat(
            [self._training_camera_frames(episode, cam, window["image"].tolist(),
                 generator=generator, parameters=visual_parameters) for cam in CAMERAS],
            dim=-1,
        )
        action = self.normalizer.forward(actions[window["action"]], "action")
        past = self.normalizer.forward(actions[window["past_action"]], "action")
        if self.recipe_version == "v2" and self.action_dim == 47:
            root_anchor = arrays[3][anchor]
            future_raw = actions[window["action"]].copy()
            past_raw = actions[window["past_action"]].copy()
            future_raw[:, 41:47] = world_quaternion_to_heading_relative_rotation6d(
                arrays[4][window["action"]], root_anchor
            )
            past_raw[:, 41:47] = world_quaternion_to_heading_relative_rotation6d(
                arrays[4][window["past_action"]], root_anchor
            )
            action = self.normalizer.forward(future_raw, "action")
            past = self.normalizer.forward(past_raw, "action")
        past[window["past_action_is_pad"]] = 0.0
        current = self.normalizer.forward(states[anchor], "state")
        current = self._training_current_state(current, states[anchor], generator=generator)
        state_condition_keep = self._training_state_keep(generator=generator)
        orientation_masks = {}
        if self.action_dim == 47:
            validity = arrays[2]
            future_pad = ~validity[window["action"]] | window["action_is_pad"]
            past_pad = ~validity[window["past_action"]] | window["past_action_is_pad"]
            action[future_pad, 41:47] = 0.0
            past[past_pad, 41:47] = 0.0
            orientation_masks = {
                "joint_orientation_is_pad": torch.from_numpy(future_pad),
                "past_joint_orientation_is_pad": torch.from_numpy(past_pad),
            }
        return {
            "video": (video * 2 - 1).permute(1, 0, 2, 3).contiguous(),
            "action": torch.from_numpy(action),
            "past_action": torch.from_numpy(past),
            "state_condition_keep": state_condition_keep,
            "proprio": torch.from_numpy(
                np.repeat(current[None], self.action_horizon, axis=0)
            ),
            "action_is_pad": torch.from_numpy(window["action_is_pad"]),
            "past_action_is_pad": torch.from_numpy(window["past_action_is_pad"]),
            "image_is_pad": torch.from_numpy(window["image_is_pad"]),
            "proprio_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
            "action_dim_is_pad": torch.zeros(self.action_dim, dtype=torch.bool),
            "proprio_dim_is_pad": torch.zeros(44, dtype=torch.bool),
            "text_hidden_states": self.text_hidden_states,
            "text_attention_mask": self.text_attention_mask,
            "prompt": self.prompt,
            "instruction": self.prompt,
            "idx": int(index),
            "episode_index": int(episode),
            "frame_index": anchor,
            "task_index": 0,
            "dataset_index": 0,
            **orientation_masks,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--train-split-mode", choices=["all_episodes"], default="all_episodes"
    )
    parser.add_argument("--body-target", choices=BODY_TARGETS, required=True)
    parser.add_argument(
        "--orientation-target", choices=ORIENTATION_TARGETS, default="external"
    )
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--past-action-size", type=int, default=16)
    parser.add_argument("--action-stride", type=int, default=2)
    parser.add_argument("--future-image-offset", type=int)
    parser.add_argument("--recipe-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--execution-horizon", type=int, default=16)
    parser.add_argument("--source-contract", type=Path)
    args = parser.parse_args()
    if args.action_horizon is None:
        args.action_horizon = 40 if args.recipe_version == "v2" else 16
    args.source_contract = (
        json.loads(args.source_contract.read_text()) if args.source_contract else None
    )
    audit = prepare_statistics(**vars(args))
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "episodes",
                    "total_frames",
                    "action_dim",
                    "timing",
                    "statistics_sha256",
                )
            },
            indent=2,
        )
    )
