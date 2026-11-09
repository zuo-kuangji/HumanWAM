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

NORMALIZATION_STRATEGY = (
    "fit_joint_reference_body29_orientation6d_q01q99_fixed_inspire_hands"
)
ROTATION6D_CONVENTION = "matrix_first_two_columns_R00_R10_R20_R01_R11_R21"


def schema_id(action_dim, body_target, recipe_version="v1"):
    require(
        action_dim in (41, 47)
        and body_target in ("wbc", "measured", "direct_reference")
        and recipe_version in ("v1", "v2")
        and not (recipe_version == "v1" and body_target == "direct_reference"),
        "Choose explicit joint41/joint47 and wbc/measured body labels",
    )
    return f"wholebody.joint_reference{action_dim}.{body_target}29.state44.p25.{recipe_version}"


class JointReferenceDeltaImageWAM(StateOnlyDeltaImageWAM):
    @staticmethod
    def accepts_schema(schema, action_dim):
        return any(
            schema == schema_id(dim, body, version) and action_dim == dim
            for dim in (41, 47)
            for version in ("v1", "v2")
            for body in (
                ("wbc", "measured")
                if version == "v1"
                else ("wbc", "measured", "direct_reference")
            )
        )

    def is_lookahead_v2(self):
        return self.state_only_schema_id.endswith(".v2")

    def reference_settings(self):
        require(
            self.accepts_schema(self.state_only_schema_id, self.expected_action_dim),
            "Joint checkpoint schema/dimension mismatch",
        )
        body = (
            "measured"
            if ".measured29." in self.state_only_schema_id
            else "direct_reference"
            if ".direct_reference29." in self.state_only_schema_id
            else "wbc"
        )
        for attribute in ("joint_statistics_sha256", "joint_instruction_sha256"):
            value = getattr(self, attribute, None)
            require(
                isinstance(value, str)
                and len(value) == 64
                and all(c in "0123456789abcdef" for c in value),
                "Joint checkpoint must bind its own dataset statistics and instruction identities",
            )
        result = {
            "body_target": body,
            "orientation_target": "external"
            if self.expected_action_dim == 41
            else "predicted_rotation6d",
            "action_stride": 2,
            "fps": 30,
            "reference_sample_dt": 2 / 30,
            "rotation6d_convention": ROTATION6D_CONVENTION,
            "statistics_sha256": self.joint_statistics_sha256,
            "instruction_sha256": self.joint_instruction_sha256,
        }
        if self.is_lookahead_v2():
            from imagewam.joint_reference_contract import (
                make_joint_recipe,
                source_contract_sha256,
            )

            recipe = getattr(self, "joint_recipe", None)
            require(
                isinstance(recipe, dict)
                and getattr(self, "joint_recipe_sha256", None)
                == source_contract_sha256(recipe),
                "Joint v2 requires its immutable pinned source/timing recipe",
            )
            expected = make_joint_recipe(
                recipe["reference_source_contract"],
                orientation_target=result["orientation_target"],
            )
            require(
                recipe == expected
                and recipe["reference_source_contract"]["body_target"] == body,
                "Joint v2 source/history/heading recipe differs from its schema",
            )
            result.update(recipe, joint_recipe_sha256=self.joint_recipe_sha256)
        else:
            require(
                not hasattr(self, "joint_recipe"),
                "Legacy joint v1 cannot adopt v2 source/timing metadata",
            )
        return result

    def validate_state_only_route(self):
        require(
            self.expected_action_horizon == (40 if self.is_lookahead_v2() else 16),
            "Joint v2 predicts40 with execution16; legacy joint v1 remains H16",
        )
        self.reference_settings()
        return super().validate_state_only_route()

    def build_inputs_flux2(self, sample, tiled=False):
        if self.is_lookahead_v2():
            sample = self._expanded_history_sample(sample)
        if self.expected_action_dim == 47:
            for field, tensor_key in (
                ("joint_orientation_is_pad", "action"),
                ("past_joint_orientation_is_pad", "past_action"),
            ):
                mask, values = sample.get(field), sample[tensor_key]
                require(
                    torch.is_tensor(mask)
                    and mask.dtype == torch.bool
                    and tuple(mask.shape) == tuple(values.shape[:2]),
                    "joint47 requires explicit per-frame recorded-orientation validity masks",
                )
                masked = values[..., 41:47][mask.to(values.device)]
                require(
                    bool((masked == 0).all()),
                    "Missing orientation must be zero in normalized target/history, not recorder identity fallback",
                )
        inputs = super().build_inputs_flux2(sample, tiled=tiled)
        if self.expected_action_dim == 47:
            action = inputs["action"]
            old_mask = inputs.get("action_dim_is_pad")
            if old_mask is None:
                mask = torch.zeros_like(action, dtype=torch.bool)
            else:
                require(
                    old_mask.dtype == torch.bool
                    and tuple(old_mask.shape) == (action.shape[0], 47),
                    "Joint dataset must provide the standard static dimension mask before merging orientation validity",
                )
                mask = old_mask[:, None, :].expand_as(action).clone()
            mask[..., 41:47] |= sample["joint_orientation_is_pad"].to(action.device)[
                ..., None
            ]
            inputs["action_dim_is_pad"] = mask
        return inputs

    def _expanded_history(self, history):
        require(
            torch.is_tensor(history)
            and history.ndim in (2, 3)
            and tuple(history.shape[-2:]) == (16, self.expected_action_dim)
            and bool(torch.isfinite(history).all()),
            "Joint v2 requires16 real normalized history points, not the predicted lookahead",
        )
        return torch.cat(
            (
                history,
                history[..., -1:, :].expand(
                    *history.shape[:-2], 24, self.expected_action_dim
                ),
            ),
            dim=-2,
        )

    def _expanded_history_sample(self, sample):
        result = dict(sample)
        result["past_action"] = self._expanded_history(sample["past_action"])
        for key in ("past_action_is_pad", "past_joint_orientation_is_pad"):
            if key in sample:
                mask = sample[key]
                require(
                    tuple(mask.shape[-1:]) == (16,),
                    "History validity mask must represent16 actual points",
                )
                result[key] = torch.cat(
                    (mask, mask[..., -1:].expand(*mask.shape[:-1], 24)), dim=-1
                )
        return result

    def infer_action_flux2(self, *args, **kwargs):
        if not self.is_lookahead_v2():
            return super().infer_action_flux2(*args, **kwargs)
        from imagewam.models.backbones.imagewam import ImageWAM

        bound = self._bind_inference(ImageWAM.infer_action_flux2, args, kwargs)
        bound.apply_defaults()
        bound.arguments["action_init"] = self._expanded_history(
            bound.arguments["action_init"]
        )
        forwarded = dict(bound.arguments)
        forwarded.pop("self")
        result = super().infer_action_flux2(**forwarded)
        return {
            **result,
            "prediction_horizon": 40,
            "execution_horizon": 16,
            "lookahead_tail_horizon": 24,
            "past_action_horizon": 16,
            "orientation_prediction_frame": "current_heading_relative"
            if self.expected_action_dim == 47
            else "external",
        }

    @staticmethod
    def _reduce_time_dimension_loss(loss, action_is_pad, action_dim_is_pad):
        require(
            tuple(action_dim_is_pad.shape) == tuple(loss.shape)
            and action_dim_is_pad.dtype == torch.bool,
            "Per-frame action dimension mask must match [B,T,D]",
        )
        valid_dim = (~action_dim_is_pad).to(device=loss.device, dtype=loss.dtype)
        counts = valid_dim.sum(dim=-1)
        token_loss = (loss * valid_dim).sum(dim=-1) / counts.clamp(min=1)
        valid_token = counts > 0
        if action_is_pad is not None:
            require(
                tuple(action_is_pad.shape) == tuple(loss.shape[:2]),
                "Action time mask must match [B,T]",
            )
            valid_token &= ~action_is_pad.to(device=loss.device, dtype=torch.bool)
        weights = valid_token.to(loss.dtype)
        return (token_loss * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)

    def _compute_action_loss_per_sample(
        self, pred_action, target_action, action_is_pad, action_dim_is_pad=None
    ):
        if action_dim_is_pad is None or action_dim_is_pad.ndim == 2:
            return super()._compute_action_loss_per_sample(
                pred_action, target_action, action_is_pad, action_dim_is_pad
            )
        loss = (pred_action.float() - target_action.float()).square()
        return self._reduce_time_dimension_loss(loss, action_is_pad, action_dim_is_pad)

    def _compute_action_ic_loss_per_sample(
        self, pred_action, target_action, action_is_pad, action_dim_is_pad=None
    ):
        if action_dim_is_pad is None or action_dim_is_pad.ndim == 2:
            return super()._compute_action_ic_loss_per_sample(
                pred_action, target_action, action_is_pad, action_dim_is_pad
            )
        require(self.action_a2a_ic_loss == "l1", "Joint endpoint consistency uses L1")
        loss = (pred_action.float() - target_action.float()).abs()
        return self._reduce_time_dimension_loss(loss, action_is_pad, action_dim_is_pad)

    def state_only_protocol(self):
        result = super().state_only_protocol()
        settings = self.reference_settings()
        dim = self.expected_action_dim
        result.update(
            format=f"wholebody_joint_reference{dim}_p25_flux2_v1",
            action_dim=dim,
            action_source=f"normalized_past_plus_0.5_gaussian_all{dim}",
            action_layout="body_joint29,inspire_left_measured6,inspire_right_measured6"
            + (",reference_root_rotation6d" if dim == 47 else ""),
            normalization_strategy=NORMALIZATION_STRATEGY,
            hand_target_offsets=[0, (self.expected_action_horizon - 1) * 2],
            future_action_frame_offsets=list(range(0, 32, 2)),
            past_action_frame_offsets=list(range(-32, 0, 2)),
            future_image_offset=32,
            future_reference_duration_seconds=1.0,
            root_orientation_source="external_required"
            if dim == 41
            else "predicted_teleop_body_quat_w",
            missing_orientation_policy="not_used"
            if dim == 41
            else "stream_mode_1_or_4_and_positive_smpl_frame_index;invalid_normalized_zero_and_6d_loss_mask",
            sonic_input_protocol="v1_joint_reference_not_v4_motion_token",
        )
        result.update(settings)
        if self.is_lookahead_v2():
            from imagewam.joint_reference_contract import (
                NORMALIZATION_V2,
                deployment_ready,
            )

            result.update(
                format=f"wholebody_joint_reference{dim}_p25_flux2_v2",
                normalization_strategy=NORMALIZATION_V2,
                hand_target_offsets=[0, 78],
                future_action_frame_offsets=list(range(0, 80, 2)),
                past_action_frame_offsets=list(range(-32, 0, 2)),
                future_reference_duration_seconds=78 / 30,
                root_orientation_source="external_required"
                if dim == 41
                else "predicted_current_heading_relative_then_world",
                missing_orientation_policy="contract_validity_mask_only_root6;current_measured_heading_required",
                source_provenance_deployment_ready=deployment_ready(
                    settings["reference_source_contract"]
                ),
                original_sonic_reference_equivalence_verified=False,
            )
        return result


def create_imagewam_flux2_wholebody_joint_reference(
    *,
    action_initializer_sha256,
    statistics_sha256,
    instruction_sha256,
    action_state_dim=44,
    expected_action_dim=41,
    expected_action_horizon=16,
    body_target="wbc",
    orientation_target="external",
    action_stride=2,
    reference_fps=30,
    state_only_schema_id=None,
    state_only_init_seed=42,
    recipe_version="v1",
    past_action_horizon=16,
    execution_horizon=16,
    future_image_offset=32,
    source_contract=None,
    source_contract_sha256=None,
    joint_recipe=None,
    **kwargs,
):
    statistics_sha256 = validate_expected_sha256(statistics_sha256)
    instruction_sha256 = validate_expected_sha256(instruction_sha256)
    require(
        (expected_action_dim, orientation_target)
        in ((41, "external"), (47, "predicted_rotation6d")),
        "joint41 requires external orientation; joint47 predicts its own reference rotation",
    )
    require(recipe_version in ("v1", "v2"), "Unknown joint recipe version")
    metadata = {
        "statistics_sha256": statistics_sha256,
        "instruction_sha256": instruction_sha256,
    }
    if recipe_version == "v2":
        from imagewam.joint_reference_contract import (
            make_joint_recipe,
            source_contract_sha256 as digest,
            plain_value,
        )

        source_contract = plain_value(source_contract)
        joint_recipe = plain_value(joint_recipe)
        require(
            isinstance(source_contract, dict),
            "Joint v2 requires explicit source contract",
        )
        require(
            source_contract.get("body_target") == body_target,
            "Source contract body target differs from requested model labels",
        )
        recipe = make_joint_recipe(
            source_contract,
            prediction_horizon=expected_action_horizon,
            past_action_horizon=past_action_horizon,
            execution_horizon=execution_horizon,
            future_image_offset=future_image_offset,
            reference_fps=reference_fps,
            action_stride=action_stride,
            orientation_target=orientation_target,
        )
        require(
            source_contract_sha256 == digest(source_contract),
            "Reference source contract SHA256 differs",
        )
        require(
            joint_recipe in (None, recipe),
            "Joint recipe differs from source/timing arguments",
        )
        metadata["joint_recipe"] = recipe
    else:
        require(
            expected_action_horizon == 16
            and action_stride == 2
            and reference_fps == 30
            and source_contract is None
            and joint_recipe is None,
            "Legacy joint v1 is H16/stride2 without a new source contract",
        )
    expected_schema = schema_id(expected_action_dim, body_target, recipe_version)
    require(
        state_only_schema_id in (None, expected_schema),
        "Explicit joint-reference schema mismatch",
    )
    return _create_state_only_flux2(
        action_initializer_sha256=action_initializer_sha256,
        action_state_dim=action_state_dim,
        expected_action_dim=expected_action_dim,
        expected_action_horizon=expected_action_horizon,
        state_only_init_seed=state_only_init_seed,
        state_only_schema_id=expected_schema,
        route_class=JointReferenceDeltaImageWAM,
        route_metadata=metadata,
        **kwargs,
    )
