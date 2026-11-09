"""Explicit G1 joint41 / joint47 Delta-flow routes, separate from motion76.

These are new architectures with freshly initialized input/output projections.
No legacy task checkpoint or optimizer is inherited. State stays action-only.
"""
from __future__ import annotations

import torch

from imagewam.wholebody_delta import (
    StateOnlyDeltaImageWAM,
    _create_state_only_flux2,
    require,
    validate_expected_sha256,
)

NORMALIZATION_STRATEGY = "fit_joint_reference_body29_orientation6d_q01q99_fixed_inspire_hands"
ROTATION6D_CONVENTION = "matrix_first_two_columns_R00_R10_R20_R01_R11_R21"


def schema_id(action_dim, body_target):
    require(action_dim in (41, 47) and body_target in ("wbc", "measured"),
            "Choose explicit joint41/joint47 and wbc/measured body labels")
    return f"wholebody.joint_reference{action_dim}.{body_target}29.state44.p25.v1"


class JointReferenceDeltaImageWAM(StateOnlyDeltaImageWAM):
    @staticmethod
    def accepts_schema(schema, action_dim):
        return any(schema == schema_id(dim, body) and action_dim == dim
                   for dim in (41, 47) for body in ("wbc", "measured"))

    def reference_settings(self):
        require(self.accepts_schema(self.state_only_schema_id, self.expected_action_dim),
                "Joint checkpoint schema/dimension mismatch")
        body = "measured" if ".measured29." in self.state_only_schema_id else "wbc"
        for attribute in ("joint_statistics_sha256", "joint_instruction_sha256"):
            value = getattr(self, attribute, None)
            require(isinstance(value, str) and len(value) == 64
                    and all(c in "0123456789abcdef" for c in value),
                    "Joint checkpoint must bind its own dataset statistics and instruction identities")
        return {"body_target": body,
                "orientation_target": "external" if self.expected_action_dim == 41 else "predicted_rotation6d",
                "action_stride": 2, "fps": 30, "reference_sample_dt": 2 / 30,
                "rotation6d_convention": ROTATION6D_CONVENTION,
                "statistics_sha256": self.joint_statistics_sha256,
                "instruction_sha256": self.joint_instruction_sha256}

    def validate_state_only_route(self):
        require(self.expected_action_horizon == 16, "Joint reference is an explicit H16 route")
        self.reference_settings()
        return super().validate_state_only_route()

    def build_inputs_flux2(self, sample, tiled=False):
        if self.expected_action_dim == 47:
            for field, tensor_key in (("joint_orientation_is_pad", "action"),
                                      ("past_joint_orientation_is_pad", "past_action")):
                mask, values = sample.get(field), sample[tensor_key]
                require(torch.is_tensor(mask) and mask.dtype == torch.bool
                        and tuple(mask.shape) == tuple(values.shape[:2]),
                        "joint47 requires explicit per-frame recorded-orientation validity masks")
                masked = values[..., 41:47][mask.to(values.device)]
                require(bool((masked == 0).all()),
                        "Missing orientation must be zero in normalized target/history, not recorder identity fallback")
        inputs = super().build_inputs_flux2(sample, tiled=tiled)
        if self.expected_action_dim == 47:
            action = inputs["action"]
            old_mask = inputs.get("action_dim_is_pad")
            if old_mask is None:
                mask = torch.zeros_like(action, dtype=torch.bool)
            else:
                require(old_mask.dtype == torch.bool and tuple(old_mask.shape) == (action.shape[0], 47),
                        "Joint dataset must provide the standard static dimension mask before merging orientation validity")
                mask = old_mask[:, None, :].expand_as(action).clone()
            mask[..., 41:47] |= sample["joint_orientation_is_pad"].to(action.device)[..., None]
            inputs["action_dim_is_pad"] = mask
        return inputs

    @staticmethod
    def _reduce_time_dimension_loss(loss, action_is_pad, action_dim_is_pad):
        require(tuple(action_dim_is_pad.shape) == tuple(loss.shape) and action_dim_is_pad.dtype == torch.bool,
                "Per-frame action dimension mask must match [B,T,D]")
        valid_dim = (~action_dim_is_pad).to(device=loss.device, dtype=loss.dtype)
        counts = valid_dim.sum(dim=-1)
        token_loss = (loss * valid_dim).sum(dim=-1) / counts.clamp(min=1)
        valid_token = counts > 0
        if action_is_pad is not None:
            require(tuple(action_is_pad.shape) == tuple(loss.shape[:2]), "Action time mask must match [B,T]")
            valid_token &= ~action_is_pad.to(device=loss.device, dtype=torch.bool)
        weights = valid_token.to(loss.dtype)
        return (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def _compute_action_loss_per_sample(self, pred_action, target_action, action_is_pad,
                                        action_dim_is_pad=None):
        if action_dim_is_pad is None or action_dim_is_pad.ndim == 2:
            return super()._compute_action_loss_per_sample(pred_action, target_action,
                action_is_pad, action_dim_is_pad)
        loss = (pred_action.float() - target_action.float()).square()
        return self._reduce_time_dimension_loss(loss, action_is_pad, action_dim_is_pad)

    def _compute_action_ic_loss_per_sample(self, pred_action, target_action, action_is_pad,
                                           action_dim_is_pad=None):
        if action_dim_is_pad is None or action_dim_is_pad.ndim == 2:
            return super()._compute_action_ic_loss_per_sample(pred_action, target_action,
                action_is_pad, action_dim_is_pad)
        require(self.action_a2a_ic_loss == "l1", "Joint endpoint consistency uses L1")
        loss = (pred_action.float() - target_action.float()).abs()
        return self._reduce_time_dimension_loss(loss, action_is_pad, action_dim_is_pad)

    def state_only_protocol(self):
        result = super().state_only_protocol()
        settings = self.reference_settings()
        dim = self.expected_action_dim
        result.update(format=f"wholebody_joint_reference{dim}_p25_flux2_v1",
            action_dim=dim, action_source=f"normalized_past_plus_0.5_gaussian_all{dim}",
            action_layout="body_joint29,inspire_left_measured6,inspire_right_measured6" +
                (",reference_root_rotation6d" if dim == 47 else ""),
            normalization_strategy=NORMALIZATION_STRATEGY,
            hand_target_offsets=[0, (self.expected_action_horizon - 1) * 2],
            future_action_frame_offsets=list(range(0, 32, 2)),
            past_action_frame_offsets=list(range(-32, 0, 2)),
            future_image_offset=32,
            future_reference_duration_seconds=1.0,
            root_orientation_source="external_required" if dim == 41 else "predicted_teleop_body_quat_w",
            missing_orientation_policy="not_used" if dim == 41 else "stream_mode_1_or_4_and_positive_smpl_frame_index;invalid_normalized_zero_and_6d_loss_mask",
            sonic_input_protocol="v1_joint_reference_not_v4_motion_token",
            **settings)
        return result


def create_imagewam_flux2_wholebody_joint_reference(*, action_initializer_sha256,
        statistics_sha256, instruction_sha256,
        action_state_dim=44, expected_action_dim=41, expected_action_horizon=16,
        body_target="wbc", orientation_target="external", action_stride=2, reference_fps=30,
        state_only_schema_id=None, state_only_init_seed=42, **kwargs):
    statistics_sha256 = validate_expected_sha256(statistics_sha256)
    instruction_sha256 = validate_expected_sha256(instruction_sha256)
    require((expected_action_dim, orientation_target) in
            ((41, "external"), (47, "predicted_rotation6d")),
            "joint41 requires external orientation; joint47 predicts its own reference rotation")
    require(expected_action_horizon == 16 and action_stride == 2 and reference_fps == 30,
            "This H16 recipe uses 30 FPS / stride2 references, spanning one second")
    expected_schema = schema_id(expected_action_dim, body_target)
    require(state_only_schema_id in (None, expected_schema), "Explicit joint-reference schema mismatch")
    return _create_state_only_flux2(action_initializer_sha256=action_initializer_sha256,
        action_state_dim=action_state_dim, expected_action_dim=expected_action_dim,
        expected_action_horizon=expected_action_horizon, state_only_init_seed=state_only_init_seed,
        state_only_schema_id=expected_schema, route_class=JointReferenceDeltaImageWAM,
        route_metadata={"statistics_sha256": statistics_sha256, "instruction_sha256": instruction_sha256}, **kwargs)
