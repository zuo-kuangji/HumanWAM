"""Explicit joint-reference inference; no robot command or token64 conversion."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .imagewam_delta import build_serve_command as build_serve_command
from .imagewam_delta import build_train_command as build_train_command

FACTORY = "imagewam.wholebody_joint_reference.create_imagewam_flux2_wholebody_joint_reference"


def validate_observation(observation, dimension, *, recipe_version="v1"):
    forbidden = {"future_state", "future_image", "normalized_past_actions", "actions"}
    if forbidden.intersection(observation):
        raise ValueError("Use current observation and strictly-past raw joint references only")
    state = np.asarray(observation["current_state"], dtype=np.float32)
    history = np.asarray(observation["past_actions"], dtype=np.float32)
    if history.size == 0:
        history = history.reshape(0, dimension)
    if state.shape != (44,) or not np.isfinite(state).all():
        raise ValueError("Expected current finite measured state44")
    if (
        history.ndim != 2
        or history.shape[1] != dimension
        or len(history) > 16
        or not np.isfinite(history).all()
    ):
        raise ValueError(
            "Expected at most16 already-strided raw past references of the trained dimension"
        )
    offsets = np.asarray(observation["past_action_frame_offsets"])
    expected = np.arange(-2 * len(history), 0, 2)
    if (
        (len(history) and offsets.dtype.kind not in "iu")
        or offsets.shape != expected.shape
        or not np.array_equal(offsets, expected)
    ):
        raise ValueError(
            "History must declare strictly-past30FPS offsets[-32,-30,...,-2], not contiguous history"
        )
    if not len(history) and observation.get("at_episode_start") is not True:
        raise ValueError("Empty history is only valid at an explicitly declared episode start")
    if np.any((state[29:41] < 0) | (state[29:41] > 1)) or np.any(
        (history[:, 29:41] < 0) | (history[:, 29:41] > 1)
    ):
        raise ValueError("Inspire measured feedback must be in physical[0,1]")
    images = {}
    for key in ("ego_rgb", "chest_rgb"):
        value = np.asarray(observation[key])
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"{key} must be current uint8 HWC RGB")
        images[key] = value
    result = dict(images, current_state=state, past_actions=history)
    if recipe_version == "v2":
        quat = np.asarray(observation["current_root_quaternion_wxyz"], dtype=np.float64)
        if (
            quat.shape != (4,)
            or not np.isfinite(quat).all()
            or abs(np.linalg.norm(quat) - 1) > 1e-3
        ):
            raise ValueError(
                "Joint v2 needs the current unit IMU-world wxyz quaternion as its heading anchor"
            )
        stamp = observation.get("observation_time_s")
        if (
            isinstance(stamp, bool)
            or not isinstance(stamp, (int, float))
            or not np.isfinite(stamp)
            or stamp < 0
            or stamp > time.monotonic()
            or observation.get("timestamp_clock") != "monotonic"
        ):
            raise ValueError(
                "Supply the actual observation time in the policy host's monotonic clock; do not use remote wall time"
            )
        valid = np.asarray(
            observation.get("past_root_orientation_valid", np.ones(len(history), dtype=bool))
        )
        if valid.shape == (0,) and not len(history):
            valid = valid.astype(bool)
        if valid.dtype.kind != "b" or valid.shape != (len(history),):
            raise ValueError("past_root_orientation_valid must match the provided history")
        result.update(
            current_root_quaternion_wxyz=quat / np.linalg.norm(quat),
            observation_time_s=float(stamp),
            past_root_orientation_valid=valid,
        )
    return result


def normalized_history(
    raw_history, dimension, normalizer, *, root_anchor=None, orientation_valid=None
):
    """Match dataset padding without inventing a measured history at episode start."""
    history = np.asarray(raw_history, dtype=np.float32)
    if not len(history):
        return np.zeros((16, dimension), dtype=np.float32)
    invalid = None
    if root_anchor is not None:
        from imagewam.joint_reference_rotation import (
            world_quaternion_to_heading_relative_rotation6d,
        )

        from humanwam.sonic_joint_reference import rotation6d_to_quaternions_wxyz

        history = history.copy()
        valid = (
            np.ones(len(history), dtype=bool)
            if orientation_valid is None
            else np.asarray(orientation_valid)
        )
        if dimension != 47 or valid.shape != (len(history),) or valid.dtype.kind != "b":
            raise ValueError(
                "Heading-relative history requires47D actions and an explicit valid-root mask"
            )
        history[valid, 41:47] = (
            world_quaternion_to_heading_relative_rotation6d(
                rotation6d_to_quaternions_wxyz(history[valid, 41:47]), root_anchor
            )
            if np.any(valid)
            else history[valid, 41:47]
        )
        invalid = ~valid
    history = normalizer.forward(history, "action")
    if invalid is not None:
        history[invalid, 41:47] = 0
    if len(history) < 16:
        history = np.concatenate(
            (np.zeros((16 - len(history), dimension), dtype=np.float32), history)
        )
    return history


class JointReferencePolicy:
    def __init__(self, config):
        self.config = dict(config)
        if config.get("schema") not in (
            "joint41_external_orientation",
            "joint47_predicted_rotation6d",
            "joint47_heading_relative_rotation6d_v2",
        ):
            raise ValueError(
                "Choose an explicit joint41 or joint47 schema; motion76 is a separate policy"
            )
        self.dimension = 41 if config["schema"].startswith("joint41") else 47
        self.recipe_version = "v2" if config["schema"].endswith("_v2") else "v1"
        self.horizon = 40 if self.recipe_version == "v2" else 16
        if config.get("horizon", self.horizon) != self.horizon:
            raise ValueError("Joint v2 predicts40 including lookahead; legacy v1 predicts16")
        if self.recipe_version == "v2" and config.get("execution_horizon", 16) != 16:
            raise ValueError("Joint v2 executes16 source intervals, not its40 predicted samples")
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from hydra.utils import instantiate
        from imagewam.datasets.wholebody import DEFAULT_PROMPT, resize_rgb_frame_uint8
        from imagewam.datasets.wholebody_joint_reference import JointReferenceNormalizer
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(self.config["model_config"])
        if cfg.model._target_ != FACTORY or cfg.model.expected_action_dim != self.dimension:
            raise ValueError("Model factory/dimension differs from the joint-reference schema")
        if (
            cfg.model.expected_action_horizon != self.horizon
            or cfg.model.action_stride != 2
            or cfg.model.reference_fps != 30
            or cfg.model.get("recipe_version", "v1") != self.recipe_version
        ):
            raise ValueError("Joint reference timing/version must match the trained recipe")
        stats_path = Path(self.config.get("stats_path", cfg.data.train.stats_path)).resolve(
            strict=True
        )
        if (
            hashlib.sha256(stats_path.read_bytes()).hexdigest()
            != cfg.wholebody_contract.statistics_sha256
        ):
            raise ValueError("Joint statistics changed after configuration generation")
        stats = json.loads(stats_path.read_text())
        if (
            cfg.model.statistics_sha256 != cfg.wholebody_contract.statistics_sha256
            or cfg.model.instruction_sha256
            != hashlib.sha256(stats["instruction"].encode()).hexdigest()
        ):
            raise ValueError(
                "Joint checkpoint identities must match this task's statistics and instruction"
            )
        if (
            stats["action_dim"] != self.dimension
            or stats["body_target"] != cfg.model.body_target
            or stats["orientation_target"] != cfg.model.orientation_target
            or stats["action_stride"] != 2
            or stats["fps"] != 30
            or stats["action_horizon"] != self.horizon
            or stats["past_action_size"] != 16
            or stats["future_image_offset"] != 32
        ):
            raise ValueError("Statistics and model reference semantics differ")
        if self.recipe_version == "v2":
            from imagewam.joint_reference_contract import deployment_ready, make_joint_recipe
            from imagewam.joint_reference_rotation import (
                heading_relative_rotation6d_to_world_quaternion,
                world_quaternion_to_heading_relative_rotation6d,
            )

            recipe = make_joint_recipe(stats["reference_source_contract"])
            if (
                stats.get("joint_recipe") != recipe
                or OmegaConf.to_container(cfg.model.joint_recipe, resolve=True) != recipe
                or stats.get("orientation_prediction_frame") != "current_heading_relative"
            ):
                raise ValueError(
                    "Joint v2 task-bound provenance, heading frame or lookahead differs"
                )
            self.joint_recipe = recipe
            self.source_contract = recipe["reference_source_contract"]
            self.reference_source_verified = deployment_ready(self.source_contract)
            self.to_world_quaternion = heading_relative_rotation6d_to_world_quaternion
            self.to_rotation6d = world_quaternion_to_heading_relative_rotation6d
        self.normalizer = JointReferenceNormalizer(stats)
        self.body_target = stats["body_target"]
        prompt = DEFAULT_PROMPT.format(task=stats["instruction"])
        cache_path = Path(self.config["text_cache"])
        if not cache_path.name.startswith(
            hashlib.sha256(prompt.encode()).hexdigest() + ".qwen3_flux2_len128"
        ):
            raise ValueError("Text cache does not belong to this task instruction")
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        hidden, mask = cached["text_hidden_states"], cached["text_attention_mask"]
        if (
            tuple(hidden.shape) != (128, 7680)
            or tuple(mask.shape) != (128,)
            or not torch.isfinite(hidden).all()
            or not mask.any()
        ):
            raise ValueError("Invalid pure-instruction text cache")
        device = self.config.get("device", "cuda:0")
        model = instantiate(cfg.model, device=device, model_dtype=torch.bfloat16)
        model.load_checkpoint(self.config["checkpoint"])
        model.eval().requires_grad_(False)
        self.hidden = hidden[None].to(device=device, dtype=torch.bfloat16)
        self.mask = mask[None].to(device=device, dtype=torch.bool)
        self.resize = resize_rgb_frame_uint8
        self._model = model

    def predict(self, observation):
        import torch

        raw = validate_observation(observation, self.dimension, recipe_version=self.recipe_version)
        self._load()
        history = normalized_history(
            raw["past_actions"],
            self.dimension,
            self.normalizer,
            root_anchor=raw.get("current_root_quaternion_wxyz"),
            orientation_valid=raw.get("past_root_orientation_valid"),
        )
        image = torch.cat([self.resize(raw[k]) for k in ("ego_rgb", "chest_rgb")], dim=-1) * 2 - 1
        state = self.normalizer.forward(raw["current_state"], "state")
        with torch.inference_mode():
            output = self._model.infer_action(
                prompt=None,
                input_image=image[None],
                proprio=torch.from_numpy(state[None]),
                context=self.hidden,
                context_mask=self.mask,
                action_init=torch.from_numpy(history[None]),
                action_horizon=self.horizon,
                num_inference_steps=1,
                action_init_noise_strength=0.5,
                action_init_noise_type="additive",
                action_init_noise_dim_mask=[True] * self.dimension,
                clamp_action_init_after_noise=False,
                seed=observation.get("seed"),
                rand_device="cpu",
            )
        normalized = output["action"].detach().float().cpu().numpy()
        if normalized.shape != (self.horizon, self.dimension) or not np.isfinite(normalized).all():
            raise ValueError("Invalid native joint-reference prediction")
        actions = self.normalizer.backward(normalized, "action")
        if not np.isfinite(actions).all():
            raise ValueError("Invalid denormalized joint-reference prediction")
        roots = None
        if self.recipe_version == "v2":
            roots = self.to_world_quaternion(actions[:, 41:47], raw["current_root_quaternion_wxyz"])
            actions = actions.copy()
            actions[:, 41:47] = self.to_rotation6d(roots, np.array([1.0, 0.0, 0.0, 0.0]))
        result = {
            "actions": actions,
            "normalized": False,
            "action_space": self.config["schema"],
            "body_joint_position29": actions[:, :29],
            "inspire_hand_state12": actions[:, 29:41],
            "root_rotation6d": actions[:, 41:47] if self.dimension == 47 else None,
            "relative_timestamps": np.arange(self.horizon, dtype=np.float64) * (2 / 30),
            "body_target_source": {
                "wbc": "wbc_motor_qtarget",
                "measured": "measured_joint_position",
                "direct_reference": "direct_joint_reference",
            }[self.body_target],
            "joint_order": "mujoco",
            "joint_units": "rad",
            "horizon": self.horizon,
            "root_orientation_required_externally": self.dimension == 41,
            "sonic_wire_protocol": "v1_requires_offline_bridge_not_v4_tokens",
            "robot_actuation_performed": False,
        }
        if self.recipe_version == "v2":
            from humanwam.sonic_joint_reference import G1_MUJOCO_JOINT_NAMES

            ready = time.monotonic()
            result.update(
                recipe_version="v2",
                execution_horizon=16,
                past_action_horizon=16,
                lookahead_tail_horizon=24,
                action_interval_s=2 / 30,
                prediction_horizon=40,
                root_quaternions_wxyz=roots,
                joint_names=list(G1_MUJOCO_JOINT_NAMES),
                model_orientation_frame="current_heading_relative",
                root_orientation_frame="world",
                root_orientation_source="predicted_future",
                joint_recipe=self.joint_recipe,
                reference_source_contract=self.source_contract,
                source_contract_sha256=self.joint_recipe["reference_source_contract_sha256"],
                reference_source_verified=self.reference_source_verified,
                observation_time_s=raw["observation_time_s"],
                prediction_ready_time_s=ready,
                timestamp_clock="monotonic",
                plan_id=f"joint-v2:{raw['observation_time_s'].hex()}:{ready.hex()}",
                action_semantics="absolute_joint_position_reference",
                sonic_wire_protocol="v1_joint_reference_local_encoder_50hz_requires_runtime",
                robot_execution_validated=False,
            )
        return result

    def reset(self):
        return None


def build_policy(config):
    return JointReferencePolicy(config)
