"""Clocked SONIC v1.1 joint-reference planning, without sockets or actuation.

Prediction samples and controller ticks use distinct clocks/rates. By default,
40 predicted references at2/30s contain16 execution intervals plus24 tail
samples; the body controller still consumes a dense50Hz reference stream.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .sonic_joint_reference import (
    CONTROL_HZ,
    G1_ISAACLAB_JOINT_NAMES,
    MODE0_OFFSETS,
    REQUIRED_FRAMES,
    JointReference,
    build_joint_reference,
    rotation6d_to_quaternions_wxyz,
    serialize_pose_v1,
)

LOOKAHEAD_S = MODE0_OFFSETS[-1] / CONTROL_HZ
TIMING_SCHEMA = "humanwam.sonic_joint_plan.monotonic_world_root.v1"
_EPS = 1e-9


class JointPlanError(ValueError):
    def __init__(self, message: str, diagnostics: dict | None = None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class ClockRegressionError(JointPlanError):
    pass


class StaleJointPlanError(JointPlanError):
    pass


class JointPlanOverrunError(JointPlanError):
    pass


def _scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be a finite monotonic-clock scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _immutable(value: Any) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class TimedJointPlan:
    plan_id: str
    observation_time_s: float
    prediction_ready_time_s: float
    clock_id: str
    joint_positions: np.ndarray
    root_quaternions_wxyz: np.ndarray
    inspire_hand_states: np.ndarray
    relative_timestamps: np.ndarray
    joint_names: tuple[str, ...]
    joint_order: str
    body_target_source: str
    root_orientation_source: str
    execution_horizon: int
    action_interval_s: float
    prediction_horizon: int
    joint_velocities: np.ndarray | None
    joint_velocity_source: str | None
    source_provenance: dict | None = None

    @classmethod
    def from_policy_prediction(cls, prediction, *, allow_experimental_source=False):
        """Load the complete v2 backend result without importing ImageWAM.

        Source hashes bind declared provenance; they do not establish that a
        controller, synchronization or robot execution has been verified here.
        """
        if not isinstance(prediction, dict) or type(allow_experimental_source) is not bool:
            raise ValueError(
                "Use a complete policy result and an explicit experimental-source boolean"
            )
        recipe = prediction.get("joint_recipe")
        contract = prediction.get("reference_source_contract")
        if not isinstance(recipe, dict) or not isinstance(contract, dict):
            raise ValueError("Policy result requires joint_recipe and reference_source_contract")
        expected = {
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
        }
        if any(recipe.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Policy temporal/orientation recipe must be v2 prediction40/history16/execution16 with its future tail"
            )
        for key, value in {
            "prediction_horizon": 40,
            "past_action_horizon": 16,
            "execution_horizon": 16,
            "lookahead_tail_horizon": 24,
            "action_interval_s": 2 / 30,
        }.items():
            if prediction.get(key) != value:
                raise ValueError("Policy top-level prediction timing differs from its v2 recipe")
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
        body = contract.get("body_target")
        sources = {
            "direct_reference": "direct_joint_reference",
            "wbc": "wbc_motor_qtarget",
            "measured": "measured_joint_position",
        }
        semantics = {
            "direct_reference": "direct_joint_reference",
            "wbc": "wbc_motor_qtarget_experiment",
            "measured": "reference_from_demonstration",
        }
        if (
            set(contract) != required
            or contract["schema"] != "g1_joint_reference_source_v1"
            or body not in sources
            or contract["label_semantics"] != semantics[body]
            or prediction.get("body_target_source") != sources[body]
        ):
            raise ValueError("Explicit body-label source semantics differ from the source contract")
        names, indices = contract["joint_names"], contract["joint_position_indices"]
        if (
            not isinstance(names, list)
            or len(names) != 29
            or set(names) != set(G1_ISAACLAB_JOINT_NAMES)
            or not isinstance(indices, list)
            or len(indices) != 29
            or len(set(indices)) != 29
            or any(type(i) is not int or i < 0 for i in indices)
        ):
            raise ValueError(
                "Source contract requires explicit canonical29 joint names and unique source indices"
            )
        if (
            contract["joint_units"] != "rad"
            or contract["root_frame"] != "world"
            or contract["root_quaternion_order"] != "wxyz"
            or prediction.get("root_orientation_frame") != "world"
            or prediction.get("root_orientation_source") != "predicted_future"
            or prediction.get("normalized") is not False
        ):
            raise ValueError(
                "Only explicitly denormalized predictions restored to world-wxyz root coordinates may be planned"
            )
        if (
            type(contract["same_stage_pairing"]) is not bool
            or contract["root_validity_policy"]
            not in {"teleop_recorded_flags", "finite_nonzero_quaternion"}
            or any(
                not isinstance(contract[k], str) or not contract[k].strip()
                for k in ("joint_position_key", "root_quaternion_key")
            )
        ):
            raise ValueError(
                "Reference stage pairing, field keys and root validity must be explicit"
            )
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
            raise ValueError("Source contract provenance is incomplete")
        if not provenance["coordinate_frame_verified"]:
            raise ValueError(
                "Coordinate frame must be verified before restoring world-root references, even for an experiment"
            )
        experimental = body != "direct_reference"
        if experimental:
            expected_key = "action.wbc" if body == "wbc" else "observation.state"
            if contract["joint_position_key"] != expected_key or indices != list(range(22)) + list(
                range(29, 36)
            ):
                raise ValueError(
                    "Measured/motor-target experimental source fields cannot masquerade as original references"
                )
            if not allow_experimental_source:
                raise ValueError(
                    "Measured or WBC motor targets require allow_experimental_source=True"
                )
        elif (
            contract["joint_position_key"]
            in {"action.wbc", "observation.state", "action.motion_token"}
            or not contract["same_stage_pairing"]
            or not provenance["synchronization_verified"]
        ):
            raise ValueError(
                "Direct reference labels require same-stage coordinates and synchronization"
            )
        digest = hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if (
            prediction.get("source_contract_sha256") != digest
            or recipe.get("reference_source_contract_sha256") != digest
            or recipe.get("reference_source_contract") != contract
        ):
            raise ValueError("Source contract canonical SHA256 or recipe provenance differs")
        actions = np.asarray(prediction.get("actions"))
        body_values = np.asarray(prediction.get("body_joint_position29"))
        hands = np.asarray(prediction.get("inspire_hand_state12"))
        if (
            actions.shape != (40, 47)
            or actions.dtype.kind not in "fiu"
            or not np.isfinite(actions).all()
            or body_values.shape != (40, 29)
            or hands.shape != (40, 12)
            or not np.array_equal(actions[:, :29], body_values)
            or not np.array_equal(actions[:, 29:41], hands)
        ):
            raise ValueError("Policy body/hand views must match the complete raw40x47 prediction")
        if tuple(prediction.get("joint_names", ())) != tuple(names):
            raise ValueError(
                "Prediction joint names must retain their source contract's action order"
            )
        roots = np.asarray(prediction.get("root_quaternions_wxyz"), dtype=np.float64)
        roots_from_actions = rotation6d_to_quaternions_wxyz(actions[:, 41:47])
        if (
            roots.shape != (40, 4)
            or not np.isfinite(roots).all()
            or not np.allclose(np.abs(np.sum(roots_from_actions * roots, axis=1)), 1.0, atol=1e-5)
        ):
            raise ValueError(
                "Policy world-root quaternions must match its restored world rotation6D action view"
            )
        plan = cls.from_prediction(
            plan_id=prediction["plan_id"],
            observation_time_s=prediction["observation_time_s"],
            prediction_ready_time_s=prediction["prediction_ready_time_s"],
            joint_positions=body_values,
            root_quaternions_wxyz=roots,
            inspire_hand_states=hands,
            relative_timestamps=prediction["relative_timestamps"],
            joint_names=prediction["joint_names"],
            joint_order=prediction["joint_order"],
            body_target_source=sources[body],
            root_orientation_source=prediction["root_orientation_source"],
            root_orientation_frame=prediction["root_orientation_frame"],
            timestamp_clock=prediction["timestamp_clock"],
            clock_id=prediction.get("clock_id", "local_monotonic"),
            normalized=prediction["normalized"],
            joint_units=prediction.get("joint_units", "rad"),
            prediction_horizon=40,
            execution_horizon=16,
            action_interval_s=2 / 30,
            joint_velocities=prediction.get("joint_velocities"),
            joint_velocity_source=prediction.get("joint_velocity_source"),
        )
        return replace(
            plan,
            source_provenance={
                "joint_recipe": copy.deepcopy(recipe),
                "reference_source_contract": copy.deepcopy(contract),
                "source_contract_sha256": digest,
                "experimental_source_allowed": allow_experimental_source,
                "experimental_control_chain": experimental,
                "source_coordinate_frame_verified": True,
                "source_synchronization_verified": provenance["synchronization_verified"],
                "controller_validated": False,
                "robot_execution_validated": False,
            },
        )

    @classmethod
    def from_prediction(
        cls,
        *,
        plan_id,
        observation_time_s,
        prediction_ready_time_s,
        joint_positions,
        root_quaternions_wxyz,
        inspire_hand_states,
        relative_timestamps,
        joint_names,
        joint_order,
        body_target_source,
        root_orientation_source="predicted_future",
        root_orientation_frame="world",
        timestamp_clock="monotonic",
        clock_id="local_monotonic",
        normalized=False,
        joint_units="rad",
        action_semantics="absolute_joint_position_reference",
        execution_horizon=16,
        action_interval_s=2 / 30,
        prediction_horizon=40,
        joint_velocities=None,
        joint_velocity_source=None,
    ):
        if not isinstance(plan_id, str) or not plan_id.strip():
            raise ValueError("plan_id must identify the actual model prediction")
        if timestamp_clock != "monotonic" or not isinstance(clock_id, str) or not clock_id:
            raise ValueError("Use one explicitly shared monotonic clock domain")
        if root_orientation_frame != "world" or root_orientation_source != "predicted_future":
            raise ValueError(
                "Online plans require predicted future world-root quaternions, never recorded/current substitutes"
            )
        observation = _scalar(observation_time_s, "observation_time_s")
        ready = _scalar(prediction_ready_time_s, "prediction_ready_time_s")
        if ready < observation:
            raise ValueError("Prediction cannot be ready before observation capture")
        if (
            type(execution_horizon) is not int
            or type(prediction_horizon) is not int
            or execution_horizon < 1
            or prediction_horizon <= execution_horizon
        ):
            raise ValueError(
                "Prediction must contain an explicit positive execution horizon and a future tail"
            )
        interval = _scalar(action_interval_s, "action_interval_s")
        if interval <= 0:
            raise ValueError(
                "action_interval_s must be positive and distinct from the50Hz control dt"
            )
        times = np.asarray(relative_timestamps, dtype=np.float64)
        if (
            times.shape != (prediction_horizon,)
            or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), interval, rtol=0, atol=1e-8)
        ):
            raise ValueError(
                "Explicit prediction timestamps must match prediction_horizon and source action_interval_s"
            )
        if joint_velocities is not None:
            if joint_velocity_source != "same_reference_trajectory_rad_per_second":
                raise ValueError(
                    "Provided qvel must explicitly come from the same reference trajectory in rad/s"
                )
        elif joint_velocity_source is not None:
            raise ValueError("Do not declare supplied velocity provenance without qvel values")
        # Reuse actual body semantics/order/quaternion/bounds checks. This is
        # offline validation only; it neither runs an encoder nor sends a packet.
        build_joint_reference(
            joint_positions=joint_positions,
            joint_names=joint_names,
            joint_order=joint_order,
            root_quaternions_wxyz=root_quaternions_wxyz,
            relative_timestamps=times,
            inspire_hand_states=inspire_hand_states,
            normalized=normalized,
            joint_units=joint_units,
            action_semantics=action_semantics,
            root_orientation_source=root_orientation_source,
            frame_index_start=0,
            body_target_source=body_target_source,
            joint_velocities=joint_velocities,
        )
        return cls(
            plan_id,
            observation,
            ready,
            clock_id,
            _immutable(joint_positions),
            _immutable(root_quaternions_wxyz),
            _immutable(inspire_hand_states),
            _immutable(times),
            tuple(joint_names),
            joint_order,
            body_target_source,
            root_orientation_source,
            execution_horizon,
            interval,
            prediction_horizon,
            None if joint_velocities is None else _immutable(joint_velocities),
            joint_velocity_source,
        )

    @property
    def absolute_start_time_s(self):
        return self.observation_time_s + float(self.relative_timestamps[0])

    @property
    def absolute_end_time_s(self):
        return self.observation_time_s + float(self.relative_timestamps[-1])

    @property
    def execution_duration_s(self):
        # H intervals, not(H-1)sample gaps and notH control ticks.
        return self.execution_horizon * self.action_interval_s

    def reference_at_relative_times(self, target_relative_times, frame_index_start):
        """Resample on a small relative grid, preserving sub-tick precision at long uptimes."""
        relative = np.asarray(target_relative_times, dtype=np.float64)
        return build_joint_reference(
            joint_positions=self.joint_positions,
            joint_names=self.joint_names,
            joint_order=self.joint_order,
            root_quaternions_wxyz=self.root_quaternions_wxyz,
            relative_timestamps=self.relative_timestamps,
            inspire_hand_states=self.inspire_hand_states,
            normalized=False,
            joint_units="rad",
            action_semantics="absolute_joint_position_reference",
            root_orientation_source=self.root_orientation_source,
            frame_index_start=frame_index_start,
            body_target_source=self.body_target_source,
            joint_velocities=self.joint_velocities,
            output_timestamps=relative,
        )


@dataclass(frozen=True)
class TimedJointWindow:
    plan_id: str
    reference: JointReference
    absolute_timestamps_s: np.ndarray
    body_packet: bytes
    inspire_hand_states: np.ndarray
    execution_deadline_s: float
    diagnostics: dict


class JointReferenceBuffer:
    """Maintain an absolute50Hz clock and a bounded source-interval execution chunk.

    Repeated window reads never restart or extend the execution deadline. A new
    model prediction is required before that deadline, even when an old tail
    still exists. Returned references contain no synthetic terminal padding.
    """

    def __init__(
        self,
        *,
        control_epoch_time_s,
        execution_horizon=16,
        action_interval_s=2 / 30,
        jitter_budget_s=0.05,
        clock_id="local_monotonic",
    ):
        self.control_epoch_time_s = _scalar(control_epoch_time_s, "control_epoch_time_s")
        if type(execution_horizon) is not int or execution_horizon < 1:
            raise ValueError("execution_horizon must be a positive source-sample interval count")
        self.execution_horizon = execution_horizon
        self.action_interval_s = _scalar(action_interval_s, "action_interval_s")
        if self.action_interval_s <= 0:
            raise ValueError("action_interval_s must be positive")
        self.jitter_budget_s = _scalar(jitter_budget_s, "jitter_budget_s")
        if not isinstance(clock_id, str) or not clock_id:
            raise ValueError("clock_id must explicitly identify the shared monotonic domain")
        self.clock_id = clock_id
        self._plan = None
        self._last_now_s = None
        self._execution_deadline_s = None
        self._handoff = None
        self._seen_plan_ids = set()
        self._reference_cache = {}

    def _clock(self, now_s):
        now = _scalar(now_s, "now_s")
        if now < self.control_epoch_time_s:
            raise ClockRegressionError("Clock precedes the configured control epoch")
        if self._last_now_s is not None and now < self._last_now_s - _EPS:
            raise ClockRegressionError(
                "Monotonic clock moved backwards; no old frames will be replayed"
            )
        self._last_now_s = now
        return now

    def _index(self, time_s):
        index = int(math.ceil((time_s - self.control_epoch_time_s) * CONTROL_HZ - _EPS))
        if index < 0 or index >= np.iinfo(np.int64).max - 100_000:
            raise JointPlanError(
                "Control clock cannot be represented by nonnegative int64 frame indices"
            )
        return index

    def _time(self, index):
        return self.control_epoch_time_s + index / CONTROL_HZ

    def _budget(self, plan, now, deadline=None):
        index = self._index(now)
        start = self._time(index)
        if start < plan.absolute_start_time_s - _EPS:
            raise StaleJointPlanError(
                "First predicted reference is still in the future; do not fill back to the present"
            )
        deadline = start + plan.execution_duration_s if deadline is None else deadline
        required_end = deadline + LOOKAHEAD_S + self.jitter_budget_s
        diagnostics = {
            "schema": TIMING_SCHEMA,
            "timestamp_clock": "monotonic",
            "clock_id": self.clock_id,
            "observation_time_s": plan.observation_time_s,
            "prediction_ready_time_s": plan.prediction_ready_time_s,
            "now_s": now,
            "first_control_tick_time_s": start,
            "observation_to_ready_latency_s": plan.prediction_ready_time_s
            - plan.observation_time_s,
            "observation_to_control_latency_s": start - plan.observation_time_s,
            "control_grid_roundup_s": start - now,
            "execution_horizon_source_intervals": plan.execution_horizon,
            "source_action_interval_s": plan.action_interval_s,
            "control_dt_s": 1 / CONTROL_HZ,
            "execution_duration_s": plan.execution_duration_s,
            "remaining_execution_duration_s": max(0.0, deadline - start),
            "execution_deadline_s": deadline,
            "encoder_lookahead_s": LOOKAHEAD_S,
            "jitter_budget_s": self.jitter_budget_s,
            "required_trajectory_end_s": required_end,
            "predicted_trajectory_end_s": plan.absolute_end_time_s,
            "future_margin_after_reserved_jitter_s": plan.absolute_end_time_s - required_end,
            "prediction_horizon_samples": plan.prediction_horizon,
            "future_tail_samples": plan.prediction_horizon - plan.execution_horizon,
            "future_padding": False,
            "transport_started": False,
        }
        if required_end > plan.absolute_end_time_s + _EPS:
            raise StaleJointPlanError(
                "Prediction lacks execution plus0.9s lookahead and declared jitter margin",
                diagnostics,
            )
        return index, start, deadline, diagnostics

    def _reference(self, plan, first_index):
        # Derive velocities once on the complete same-plan global50Hz grid.
        # Cropping a later window must not recompute an endpoint derivative and
        # change dq at a timestamp already present in the preceding packet.
        if plan.plan_id not in self._reference_cache:
            initial = self._index(max(plan.absolute_start_time_s, self.control_epoch_time_s))
            # Subtract the two clock origins once, before adding small offsets.
            # Per-point subtraction of large absolute timestamps loses enough
            # bits at long system uptimes to corrupt an otherwise exact50Hz grid.
            observation_offset = plan.observation_time_s - self.control_epoch_time_s
            last = int(
                math.floor((observation_offset + plan.relative_timestamps[-1]) * CONTROL_HZ + _EPS)
            )
            indices = np.arange(initial, last + 1, dtype=np.int64)
            absolute = self.control_epoch_time_s + indices.astype(np.float64) / CONTROL_HZ
            first_relative = -observation_offset + initial / CONTROL_HZ
            relative = first_relative + np.arange(len(indices), dtype=np.float64) / CONTROL_HZ
            self._reference_cache[plan.plan_id] = (
                plan.reference_at_relative_times(relative, initial),
                _immutable(absolute),
            )
        complete, absolute = self._reference_cache[plan.plan_id]
        offset = first_index - int(complete.frame_index[0])
        if offset < 0 or len(complete.frame_index) - offset < REQUIRED_FRAMES:
            raise StaleJointPlanError(
                "No complete unpadded0.9s window exists at the requested control tick"
            )
        reference = JointReference(
            complete.joint_pos[offset:],
            complete.joint_vel[offset:],
            complete.body_quat[offset:],
            complete.frame_index[offset:],
            complete.relative_timestamps[offset:],
            complete.inspire_hand_states[offset:],
            dict(
                complete.metadata,
                frame_index_start=first_index,
                runtime_slice_of_global_reference=True,
                runtime_velocity_fixed_across_windows=True,
            ),
        )
        return reference, absolute[offset:]

    def submit(self, plan: TimedJointPlan, *, now_s):
        now = self._clock(now_s)
        if not isinstance(plan, TimedJointPlan):
            raise TypeError("Submit a validated TimedJointPlan")
        if (
            plan.clock_id != self.clock_id
            or plan.execution_horizon != self.execution_horizon
            or not math.isclose(
                plan.action_interval_s, self.action_interval_s, rel_tol=0, abs_tol=1e-10
            )
        ):
            raise JointPlanError("Plan clock or source execution timing differs from this buffer")
        if plan.prediction_ready_time_s > now + _EPS:
            raise JointPlanError("Prediction ready timestamp is later than submit time")
        if plan.plan_id in self._seen_plan_ids:
            raise JointPlanError("A consumed plan_id cannot reset an old execution deadline")
        if self._plan is not None and plan.observation_time_s <= self._plan.observation_time_s:
            raise JointPlanError(
                "Out-of-order or repeated observation cannot replace the current future"
            )
        index, start, _, diagnostics = self._budget(plan, now)
        handoff = {"previous_plan_id": None, "handoff_time_s": start, "trajectory_blended": False}
        if (
            self._plan is not None
            and self._plan.absolute_start_time_s <= start
            and start + LOOKAHEAD_S <= self._plan.absolute_end_time_s
        ):
            previous, _ = self._reference(self._plan, index)
            current, _ = self._reference(plan, index)
            dot = abs(float(np.dot(previous.body_quat[0, 0], current.body_quat[0, 0])))
            handoff.update(
                previous_plan_id=self._plan.plan_id,
                joint_position_jump_max_abs_rad=float(
                    np.max(np.abs(previous.joint_pos[0] - current.joint_pos[0]))
                ),
                joint_velocity_jump_max_abs_rad_s=float(
                    np.max(np.abs(previous.joint_vel[0] - current.joint_vel[0]))
                ),
                root_orientation_jump_rad=float(2 * np.arccos(np.clip(dot, 0, 1))),
                inspire_state_jump_max_abs=float(
                    np.max(np.abs(previous.inspire_hand_states[0] - current.inspire_hand_states[0]))
                ),
            )
        self._plan = plan
        self._execution_deadline_s = None
        self._handoff = handoff
        self._seen_plan_ids.add(plan.plan_id)
        # Keep only the active prediction; previously returned immutable packet
        # arrays remain valid for callers without retaining an unbounded history.
        if plan.plan_id in self._reference_cache:
            self._reference_cache = {plan.plan_id: self._reference_cache[plan.plan_id]}
        else:
            self._reference_cache.clear()
        return dict(diagnostics, handoff=handoff)

    def window(self, *, now_s) -> TimedJointWindow:
        now = self._clock(now_s)
        if self._plan is None:
            raise JointPlanError("No validated future plan has been submitted")
        if (
            self._execution_deadline_s is not None
            and self._time(self._index(now)) >= self._execution_deadline_s - _EPS
        ):
            raise JointPlanOverrunError(
                "Execution chunk ended without a fresh model plan; old tail will not be replayed",
                {
                    "plan_id": self._plan.plan_id,
                    "now_s": now,
                    "execution_deadline_s": self._execution_deadline_s,
                },
            )
        index, _, deadline, diagnostics = self._budget(self._plan, now, self._execution_deadline_s)
        reference, absolute = self._reference(self._plan, index)
        packet = serialize_pose_v1(reference)
        self._execution_deadline_s = deadline
        diagnostics.update(
            plan_id=self._plan.plan_id,
            handoff=self._handoff,
            joint_velocity_source=self._plan.joint_velocity_source
            or "difference_of_same_resampled_reference_qpos",
            root_orientation_frame="world",
            root_orientation_schema="predicted_future_world_wxyz",
            source_provenance=copy.deepcopy(self._plan.source_provenance),
            controller_validated=False,
            robot_execution_validated=False,
        )
        return TimedJointWindow(
            self._plan.plan_id,
            reference,
            absolute,
            packet,
            reference.inspire_hand_states,
            deadline,
            diagnostics,
        )

    def status(self, *, now_s):
        now = self._clock(now_s)
        if self._plan is None:
            return {"state": "empty", "transport_started": False}
        if (
            self._execution_deadline_s is not None
            and self._time(self._index(now)) >= self._execution_deadline_s - _EPS
        ):
            return {
                "state": "overrun",
                "plan_id": self._plan.plan_id,
                "execution_deadline_s": self._execution_deadline_s,
                "transport_started": False,
            }
        try:
            *_, diagnostics = self._budget(self._plan, now, self._execution_deadline_s)
        except StaleJointPlanError as error:
            return {
                "state": "stale",
                "plan_id": self._plan.plan_id,
                "diagnostics": error.diagnostics,
                "transport_started": False,
            }
        return {
            "state": "ready" if self._execution_deadline_s is None else "executing",
            "plan_id": self._plan.plan_id,
            "diagnostics": diagnostics,
            "transport_started": False,
        }
