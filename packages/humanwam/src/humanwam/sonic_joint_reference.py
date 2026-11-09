"""Offline G1 joint references for SONIC v1.1 mode 0. No socket or robot APIs.

The 29 reference angles are *absolute radians*, not policy-normalized actions,
motor residuals, or SONIC64 tokens. Inspire feedback is exported separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

SONIC_SOURCE_COMMIT = "1e1ec4f5f27e6c3e03048f693bd7c60cce04afda"
SCHEMA = "humanwam.g1_joint29_inspire12.sonic_v1_1_mode0.v1"
CONTROL_HZ = 50
MODE0_OFFSETS = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
REQUIRED_FRAMES = 46
HEADER_SIZE = 1280
TOPIC = b"pose"
BODY_TARGET_SOURCES = frozenset({
    "wbc_motor_qtarget", "measured_joint_position", "direct_joint_reference",
})

# Native G1 MJCF kinematic traversal order. Checked against g1.py, the MJCF,
# and deployment policy_parameters.hpp at SONIC_SOURCE_COMMIT in CPU tests.
G1_MUJOCO_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
G1_MUJOCO_TO_ISAACLAB = (
    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
    16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28,
)
G1_ISAACLAB_JOINT_NAMES = tuple(G1_MUJOCO_JOINT_NAMES[i] for i in G1_MUJOCO_TO_ISAACLAB)


class InsufficientLookaheadError(ValueError):
    """A reference must cover the full native future window without padding."""

    def __init__(self, coverage):
        self.coverage = coverage
        super().__init__(
            "SONIC v1.1 mode0 needs offsets0..45: at least46 dense50Hz frames "
            f"over0.9s; input covers{coverage['span_seconds']:.6f}s / "
            f"{coverage['available_control_frames']} frames. No last-frame padding "
            "or future/root substitution is allowed. Use genuinely predicted/recorded "
            "references spanning the window (e.g. trained H16 stride2 at30Hz)."
        )


def _array(value, shape, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu" or raw.shape != shape:
        raise ValueError(f"{name} must be a finite real array of shape{shape}")
    result = raw.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _times(value):
    raw = np.asarray(value)
    if raw.ndim != 1 or len(raw) < 2:
        raise ValueError("relative_timestamps needs at least2 samples")
    result = _array(raw, raw.shape, "relative_timestamps")
    if result[0] < 0 or np.any(np.diff(result) <= 0):
        raise ValueError("relative_timestamps must be nonnegative and strictly increase in seconds")
    return result


def lookahead_coverage(relative_timestamps):
    """Report reference coverage only; never generate or pad missing future data."""
    times = _times(relative_timestamps)
    span = times[-1] - times[0]
    if span > (100_000 - 1) / CONTROL_HZ:
        raise ValueError("Offline export is limited to100000 dense reference frames")
    count = int(np.floor(span * CONTROL_HZ + 1e-8)) + 1
    return {
        "input_samples": len(times),
        "span_seconds": float(span),
        "first_reference_relative_time": float(times[0]),
        "last_reference_relative_time": float(times[-1]),
        "max_input_gap_seconds": float(np.max(np.diff(times))),
        "control_hz": CONTROL_HZ,
        "required_control_frames": REQUIRED_FRAMES,
        "required_span_seconds": MODE0_OFFSETS[-1] / CONTROL_HZ,
        "available_control_frames": count,
        "complete_mode0_window": count >= REQUIRED_FRAMES,
    }


def _readonly_float32(value):
    with np.errstate(over="ignore"):
        result = np.array(value, dtype="<f4", order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Values overflow float32 reference representation")
    result.setflags(write=False)
    return result


def _interpolate(times, values, target):
    return np.column_stack([np.interp(target, times, values[:, i]) for i in range(values.shape[1])])


def _slerp(times, quaternions, target):
    """Shortest-arc quaternion SLERP, with sign continuity and no extrapolation."""
    quat = quaternions.copy()
    norm = np.linalg.norm(quat, axis=1)
    if np.any(np.abs(norm - 1.0) > 1e-3):
        raise ValueError("root_quaternions_wxyz must be unit world-root quaternions")
    quat /= norm[:, None]
    for i in range(1, len(quat)):
        if np.dot(quat[i - 1], quat[i]) < 0:
            quat[i] *= -1
    left = np.clip(np.searchsorted(times, target, side="right") - 1, 0, len(times) - 2)
    fraction = (target - times[left]) / (times[left + 1] - times[left])
    a, b = quat[left], quat[left + 1]
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    linear = dot > 0.9995
    result = (1 - fraction[:, None]) * a + fraction[:, None] * b
    curved = ~linear
    angle = np.arccos(dot[curved])
    weight = fraction[curved]
    result[curved] = (
        np.sin((1 - weight) * angle)[:, None] * a[curved]
        + np.sin(weight * angle)[:, None] * b[curved]
    ) / np.sin(angle)[:, None]
    return result / np.linalg.norm(result, axis=1)[:, None]


def rotation6d_to_quaternions_wxyz(rotation6d):
    """Decode column-concatenated R[:,0],R[:,1] using Gram-Schmidt.

    This matches [R00,R10,R20,R01,R11,R21]. The two input columns need not
    already be orthonormal, but zero/parallel columns are rejected.
    """
    raw = np.asarray(rotation6d)
    if raw.ndim != 2 or raw.shape[1] != 6:
        raise ValueError("rotation6d must have shape[N,6] with concatenated matrix columns")
    values = _array(raw, raw.shape, "rotation6d")
    def unit(vector, label):
        scale = np.max(np.abs(vector), axis=1)
        if np.any(scale < 1e-8):
            raise ValueError(f"Degenerate rotation6d {label} column")
        scaled = vector / scale[:, None]
        return scaled / np.linalg.norm(scaled, axis=1)[:, None]

    first = unit(values[:, :3], "first")
    second = unit(values[:, 3:], "second")
    second = second - np.sum(first * second, axis=1)[:, None] * first
    norm = np.linalg.norm(second, axis=1)
    if np.any(norm < 1e-8):
        raise ValueError("Degenerate parallel rotation6d columns")
    second = second / norm[:, None]
    rotations = np.stack((first, second, np.cross(first, second)), axis=2)
    result = np.empty((len(rotations), 4), dtype=np.float64)
    for i, r in enumerate(rotations):
        trace = np.trace(r)
        if trace > 0:
            scale = 2 * np.sqrt(trace + 1)
            q = (scale / 4, (r[2, 1] - r[1, 2]) / scale,
                 (r[0, 2] - r[2, 0]) / scale, (r[1, 0] - r[0, 1]) / scale)
        else:
            j = int(np.argmax(np.diag(r)))
            k, m = (j + 1) % 3, (j + 2) % 3
            scale = 2 * np.sqrt(1 + r[j, j] - r[k, k] - r[m, m])
            xyz = np.empty(3)
            xyz[j] = scale / 4
            xyz[k] = (r[j, k] + r[k, j]) / scale
            xyz[m] = (r[j, m] + r[m, j]) / scale
            q = ((r[m, k] - r[k, m]) / scale, *xyz)
        result[i] = q / np.linalg.norm(q)
        if result[i, 0] < 0:
            result[i] *= -1
    return result


def split_model_prediction(
    prediction, *, prediction_schema, normalized, body_target_source,
    external_root_quaternions_wxyz=None, external_root_orientation_source=None,
):
    """Split explicitly raw41/external-root or raw47/predicted-root model output.

    This is schema conversion only, not an action-history or SONIC64 decoder.
    The caller must retain target provenance and validate it in the builder.
    """
    if normalized is not False:
        raise ValueError("Denormalize model predictions with their own checkpoint statistics first")
    if body_target_source not in BODY_TARGET_SOURCES:
        raise ValueError("Declare body_target_source: wbc_motor_qtarget, measured_joint_position, or direct_joint_reference")
    dimensions = {"external41": 41, "predicted_rotation6d47": 47}
    raw = np.asarray(prediction)
    if (prediction_schema not in dimensions or raw.ndim != 2
            or raw.shape[1] != dimensions[prediction_schema] or len(raw) < 2):
        raise ValueError("Expected explicit external41 or predicted_rotation6d47 [N,D] schema")
    values = _array(raw, raw.shape, "raw model prediction")
    if prediction_schema == "external41":
        if external_root_quaternions_wxyz is None:
            raise ValueError("external41 needs separately supplied future world-root quaternions")
        if external_root_orientation_source not in {"predicted_future", "recorded_future"}:
            raise ValueError("external41 must declare external_root_orientation_source")
        quat = _array(external_root_quaternions_wxyz, (len(values), 4), "external future root quaternion")
        root_source = external_root_orientation_source
    else:
        if external_root_quaternions_wxyz is not None or external_root_orientation_source is not None:
            raise ValueError("predicted_rotation6d47 already predicts orientation; do not override it")
        quat = rotation6d_to_quaternions_wxyz(values[:, 41:47])
        root_source = "predicted_future"
    return {
        "joint_positions": values[:, :29].copy(),
        "inspire_hand_states": values[:, 29:41].copy(),
        "root_quaternions_wxyz": quat,
        "body_target_source": body_target_source,
        "root_orientation_source": root_source,
    }


@dataclass(frozen=True)
class JointReference:
    """Dense 50Hz body references and separately named Inspire feedback samples."""

    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_quat: np.ndarray
    frame_index: np.ndarray
    relative_timestamps: np.ndarray
    inspire_hand_states: np.ndarray
    metadata: dict

    def protocol_fields(self):
        """Only native body fields. Never label Inspire6+6 as Dex3 7+7."""
        return {
            "joint_pos": self.joint_pos,
            "joint_vel": self.joint_vel,
            "body_quat": self.body_quat,
            "frame_index": self.frame_index,
        }

    def npz_fields(self):
        return {
            **self.protocol_fields(),
            "relative_timestamps": self.relative_timestamps,
            "inspire_hand_states": self.inspire_hand_states,
            "joint_names": np.asarray(G1_ISAACLAB_JOINT_NAMES),
            "metadata_json": np.asarray(json.dumps(self.metadata, ensure_ascii=False)),
        }


def build_joint_reference(
    *, joint_positions, joint_names, joint_order, root_quaternions_wxyz,
    relative_timestamps, inspire_hand_states, normalized, joint_units,
    action_semantics, root_orientation_source, frame_index_start, body_target_source,
    joint_velocities=None,
):
    """Validate and resample an explicit future-reference sequence, without I/O.

    Timestamps retain their declared observation-relative offset. The first
    supplied reference becomes ``frame_index_start``; nothing is filled back
    to time0. A live integration must align that anchor itself.
    Metadata declarations cannot prove that a caller's values are truly raw.
    """
    if normalized is not False or joint_units != "rad":
        raise ValueError("Only explicitly unnormalized absolute joint radians are supported")
    if action_semantics != "absolute_joint_position_reference":
        raise ValueError("Expected absolute_joint_position_reference, not motor residuals or motion64")
    if body_target_source not in BODY_TARGET_SOURCES:
        raise ValueError("Declare body_target_source: wbc_motor_qtarget, measured_joint_position, or direct_joint_reference")
    if root_orientation_source not in {"predicted_future", "recorded_future"}:
        raise ValueError("Provide predicted_future or recorded_future world-root orientations; no current-pose fallback")
    times = _times(relative_timestamps)
    n = len(times)
    joints = _array(joint_positions, (n, 29), "joint_positions (not action41 or motion64)")
    hands = _array(inspire_hand_states, (n, 12), "inspire_hand_states")
    if np.any(hands < 0) or np.any(hands > 1):
        raise ValueError("Raw Inspire measured state6+6 must be in[0,1]; no Dex3 conversion")
    if root_quaternions_wxyz is None:
        raise ValueError("Future world-root quaternions are required; identity/current-pose substitution is forbidden")
    quat = _array(root_quaternions_wxyz, (n, 4), "root_quaternions_wxyz")
    names = tuple(joint_names)
    if (len(names) != 29 or len(set(names)) != 29
            or set(names) != set(G1_ISAACLAB_JOINT_NAMES)):
        raise ValueError("Explicit joint_names must name exactly the29 canonical G1 joints once")
    orders = {"isaaclab": G1_ISAACLAB_JOINT_NAMES, "mujoco": G1_MUJOCO_JOINT_NAMES}
    if joint_order not in {*orders, "named"}:
        raise ValueError("joint_order must be isaaclab, mujoco, or explicit named order")
    if joint_order in orders and names != orders[joint_order]:
        raise ValueError("joint_names disagree with the declared joint_order")
    take = [names.index(name) for name in G1_ISAACLAB_JOINT_NAMES]
    joints = joints[:, take]
    coverage = lookahead_coverage(times)
    if not coverage["complete_mode0_window"]:
        raise InsufficientLookaheadError(coverage)
    count = coverage["available_control_frames"]
    if count > 100_000:
        raise ValueError("Offline export is limited to100000 dense reference frames")
    if (type(frame_index_start) is not int or frame_index_start < 0
            or frame_index_start > np.iinfo(np.int64).max - count):
        raise ValueError("frame_index_start must be a nonnegative int64-safe integer")
    target = times[0] + np.arange(count, dtype=np.float64) / CONTROL_HZ
    # Handle only numerical boundary noise; no semantic extrapolation occurs.
    if target[-1] > times[-1] + 1e-10:
        raise ValueError("Resampling would extrapolate beyond the supplied future reference")
    target = np.minimum(target, times[-1])
    positions = _interpolate(times, joints, target)
    if joint_velocities is None:
        velocities = np.gradient(positions, 1.0 / CONTROL_HZ, axis=0, edge_order=1)
        velocity_source = "finite_difference_resampled50hz_numpy_gradient_edge_order1"
    else:
        velocity = _array(joint_velocities, (n, 29), "joint_velocities rad/s")[:, take]
        velocities = _interpolate(times, velocity, target)
        velocity_source = "provided_rad_per_second_linear_interpolation"
    frames = frame_index_start + np.arange(count, dtype=np.int64)
    frames.setflags(write=False)
    target.setflags(write=False)
    metadata = {
        "schema": SCHEMA, "sonic_source_commit": SONIC_SOURCE_COMMIT,
        "protocol_version": 1, "encode_mode": 0, "control_hz": CONTROL_HZ,
        "action_semantics": action_semantics, "normalized": False, "joint_units": "rad",
        "body_target_source": body_target_source,
        "experimental_control_chain": body_target_source != "direct_joint_reference",
        "original_sonic_reference_equivalence_verified": False,
        "input_joint_order": joint_order, "input_joint_names": list(names),
        "output_joint_order": "isaaclab", "output_joint_names": list(G1_ISAACLAB_JOINT_NAMES),
        "quaternion_convention": "world_root_wxyz", "root_orientation_source": root_orientation_source,
        "velocity_source": velocity_source, "quaternion_interpolation": "shortest_arc_slerp",
        "position_interpolation": "linear", "hand_interpolation": "linear",
        "hand_semantics": "measured_inspire_left6_right6_state_0_1_separate_from_body_wire",
        "native_mode0_offsets": list(MODE0_OFFSETS), "coverage": coverage,
        "frame_index_start": frame_index_start, "first_reference_relative_time": float(times[0]),
        "future_padding": False, "root_substitution": False,
        "transport_started": False, "robot_execution_validated": False,
    }
    return JointReference(
        _readonly_float32(positions), _readonly_float32(velocities),
        _readonly_float32(_slerp(times, quat, target)[:, None, :]), frames,
        target, _readonly_float32(_interpolate(times, hands, target)), metadata,
    )


def serialize_pose_v1(reference: JointReference) -> bytes:
    """Create native-compatible body pose bytes only; never open a ZMQ socket."""
    if not isinstance(reference, JointReference) or reference.metadata.get("schema") != SCHEMA:
        raise ValueError("Serialize only a validated joint-reference bundle")
    fields, payload = [], []
    for name, value in reference.protocol_fields().items():
        dtype = "<i8" if name == "frame_index" else "<f4"
        arr = np.ascontiguousarray(value, dtype=dtype)
        fields.append({"name": name, "dtype": "i64" if name == "frame_index" else "f32", "shape": list(arr.shape)})
        payload.append(arr.tobytes(order="C"))
    header = json.dumps({"v": 1, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":")).encode()
    if len(header) > HEADER_SIZE:
        raise ValueError("Native SONIC header exceeds1280 bytes")
    result = TOPIC + header.ljust(HEADER_SIZE, b"\x00") + b"".join(payload)
    deserialize_pose_v1(result)  # Verify finite shapes, quaternion norms and dense indices.
    return result


def deserialize_pose_v1(data: bytes) -> dict:
    """Strict offline decoder for this v1 body subset; no command/hand fields."""
    if not isinstance(data, bytes) or not data.startswith(TOPIC) or len(data) < len(TOPIC) + HEADER_SIZE:
        raise ValueError("Expected a SONIC pose packet with1280-byte header")
    raw = data[len(TOPIC):len(TOPIC) + HEADER_SIZE]
    split = raw.find(b"\x00")
    if split < 0 or any(raw[split:]):
        raise ValueError("Expected a null-padded SONIC header")
    try:
        header = json.loads(raw[:split].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid SONIC JSON header") from exc
    if (not isinstance(header, dict) or header.get("v") != 1
            or header.get("endian") != "le" or header.get("count") != 1):
        raise ValueError("Only little-endian SONIC v1 count1 body packets are supported")
    descriptors = header.get("fields")
    expected = {"joint_pos", "joint_vel", "body_quat", "frame_index"}
    if (not isinstance(descriptors, list) or len(descriptors) != 4
            or any(not isinstance(f, dict) for f in descriptors)
            or {f.get("name") for f in descriptors} != expected):
        raise ValueError("Required unique joint_pos/joint_vel/body_quat/frame_index fields only")
    result, offset = {}, len(TOPIC) + HEADER_SIZE
    dtype_map = {"f32": "<f4", "f64": "<f8", "i32": "<i4", "i64": "<i8"}
    for field in descriptors:
        name, shape, dtype = field["name"], field.get("shape"), field.get("dtype")
        allowed = {"i32", "i64"} if name == "frame_index" else {"f32", "f64"}
        if (dtype not in allowed or not isinstance(shape, list) or not shape
                or any(type(d) is not int or d <= 0 for d in shape) or shape[0] > 100_000):
            raise ValueError("Invalid native body field dtype/shape")
        expected_shape = [shape[0]] + ({"joint_pos": [29], "joint_vel": [29], "body_quat": [1, 4], "frame_index": []}[name])
        if shape != expected_shape or shape[0] < REQUIRED_FRAMES:
            raise ValueError("Body shape or46-frame minimum does not match SONIC v1.1 mode0")
        count = int(np.prod(shape))
        end = offset + count * np.dtype(dtype_map[dtype]).itemsize
        if end > len(data):
            raise ValueError("Truncated SONIC field payload")
        result[name] = np.frombuffer(data, dtype=dtype_map[dtype], count=count, offset=offset).reshape(shape).copy()
        offset = end
    lengths = {len(v) for v in result.values()}
    if len(lengths) != 1 or offset != len(data):
        raise ValueError("Mismatched frame counts or trailing SONIC payload bytes")
    frames = result["frame_index"]
    if frames[0] < 0 or np.any(frames[1:] <= frames[:-1]) or np.any(np.diff(frames) != 1):
        raise ValueError("Expected nonnegative dense increasing50Hz frame indices")
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError("Nonfinite native body values")
    if np.any(np.abs(np.linalg.norm(result["body_quat"], axis=-1) - 1) > 1e-3):
        raise ValueError("Invalid unit world-root quaternion")
    return result
