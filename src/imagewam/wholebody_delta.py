"""Fresh-base whole-body Delta-flow: action76 and a 25% image-latent prior.

The native FLUX.2 FM, endpoint-consistency losses, visual KI workspace and
sampling loops are inherited unchanged. A scoped CURRENT measured state is
read only by the action expert's pre_dit; it is never supplied to native text,
image, VAE or visual-workspace inputs. There is no pose-prediction head/loss.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
import hashlib
import inspect
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from imagewam.models.backbones.action_dit_flux2 import ActionDiTFlux2
from imagewam.models.backbones.imagewam import ImageWAM


CHECKPOINT_FORMAT = "wholebody_inspire76_p25_flux2_v1"
SCHEMA_ID = "wholebody.inspire_action76.state44.p25.v1"
NORMALIZATION_STRATEGY = "fit_wholebody_motion64_body29_gravity3_fixed_inspire_hands"
IMAGE_RESIDUAL_SCALE = 1.0 / math.sqrt(3.0)
UPSTREAM_SHA256 = {
    "imagewam.py": "7b27cfdaa6771cf67ab25f85cce93f53be12ed1bd320bec48a296317b4713af6",
    "action_dit_flux2.py": "2bfbdaea89c78f9f4872c9cf3e70e1560faf2fec78bf68755f342c1ca9290137",
}
STATE_KEYS = frozenset({"proprio", "state", "gt_pose", "predicted_proprio",
                        "proprio_is_pad", "proprio_dim_is_pad", "state_is_pad", "state_dim_is_pad"})
NEW_ACTION_KEYS = frozenset({"action_encoder.weight", "action_encoder.bias", "head.linear.weight",
                            "state_encoder.weight", "state_encoder.bias"})
PARTIAL_NEW_ACTION_KEYS = NEW_ACTION_KEYS | {"head.adaLN_modulation.1.weight"}
PARTIAL_INITIALIZER_META = {
    "variant": "klein-base-4b", "copied": 50, "resized": 64,
    "policy": "copy FLUX.2 img/single branches into slim action blocks; interpolate mismatched axes with alpha scaling",
    "alpha_scaling": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_expected_sha256(value):
    """Require a user-pinned artifact identity, without a machine-specific pin."""
    require(isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdefABCDEF" for character in value),
            "expected SHA256 must be exactly 64 hexadecimal characters")
    return value.lower()


@dataclass
class _ActionStateScope:
    owner: object
    current: torch.Tensor
    calls: int = 0


_CURRENT_ACTION_STATE: ContextVar[_ActionStateScope | None] = ContextVar(
    "wholebody_state_only_action_current_state", default=None)


class StateOnlyActionDiTFlux2(ActionDiTFlux2):
    """One registered Linear(44, hidden), without a reference to the parent."""

    def pre_dit(self, action_tokens, timestep, context=None, context_mask=None):
        scope = _CURRENT_ACTION_STATE.get()
        if scope is None or scope.owner is not self._state_only_owner:
            raise RuntimeError("Action state conditioning requires this model's explicit CURRENT-state scope")
        expected = (int(action_tokens.shape[0]), self.action_state_dim)
        require(tuple(scope.current.shape) == expected, f"Current action-side state must be {expected}")
        pre = super().pre_dit(action_tokens, timestep, context, context_mask)
        condition = scope.current.to(device=pre["tokens"].device, dtype=pre["tokens"].dtype)
        # Recompute this projection for every FM/IC forward. Never retain its
        # autograd graph across the native independently-backpropagated losses.
        pre["tokens"] = pre["tokens"] + self.state_encoder(condition)[:, None, :]
        scope.calls += 1
        self.last_action_condition = condition.detach().clone()
        return pre


class StateOnlyDeltaImageWAM(ImageWAM):
    state_only_delta_enabled = True

    def validate_state_only_route(self):
        require(self.stack == "flux2" and self.mot.block_protocol == "flux2", "Whole-body adapter requires FLUX.2 MoT")
        require(self.proprio_dim is None and self.proprio_encoder is None
                and self._modules.get("proprio_encoder") is None and not self.pack_proprio_after_text,
                "A visual/text proprio encoder must never exist")
        require(isinstance(self.action_expert, StateOnlyActionDiTFlux2)
                and self.mot.mixtures["action"] is self.action_expert
                and self.mot.mixtures["video"] is self.video_expert and self.dit is self.mot,
                "The conditioned action expert must be registered inside the optimized/saved MoT")
        require(self.action_state_dim == 44 and self.action_expert.action_state_dim == 44
                and self.action_expert.action_dim == 76 and self.expected_action_horizon == 40,
                "Whole-body policy requires action76/current-state44/horizon40")
        require(self.action_expert.state_encoder.in_features == 44
                and self.action_expert.state_encoder.out_features == self.action_expert.hidden_dim,
                "Wrong action-side state projection")
        require(self.action_expert.max_action_horizon >= 40, "Action expert horizon is shorter than forty")
        require(not hasattr(self, "pose_head") and not self.action_aura_enabled and self.action_aura_router is None,
                "This architecture has neither C-Pred/pose supervision nor AURA")
        require(self.action_a2a_enabled and self.action_a2a_fm_weight == 1.0
                and self.action_a2a_source_noise_std == 0.5
                and tuple(self.action_a2a_source_noise_dim_mask or ()) == (True,) * 76
                and self.action_a2a_ic_weight == 0.5 and self.action_a2a_ic_steps == 1
                and self.action_a2a_ic_loss == "l1", "Wrong action Delta/FM/IC source configuration")
        require(self.image_i2i_enabled and self.image_i2i_source_mode == "gaussian"
                and math.isclose(self.image_i2i_current_residual_scale, IMAGE_RESIDUAL_SCALE, rel_tol=1e-12)
                and self.image_i2i_variance_normalize
                and self.image_i2i_ic_weight == 1.0 and self.image_i2i_ic_steps == 1
                and self.image_i2i_ic_loss == "l1", "Wrong current-image Delta/FM/IC configuration")
        require(self.flux2_action_image_context_mode == "noise_idm"
                and self.flux2_action_include_future_tokens and self.flux2_action_knowledge_insulation
                and not self.flux2_action_reuse_image_fm_workspace,
                "Expected full noise_idm visual conditioning with hard KI, not a live visual action graph")
        require(self.loss_lambda_video == 0.5 and self.loss_lambda_action == 1.0,
                "Expected visual/action objective weights 0.5/1.0")
        require(not bool(getattr(self.video_expert, "flux2_lora_enabled", False)),
                "This architecture trains the visual DiT, not a LoRA-only variant")
        return self.state_only_protocol()

    def state_only_protocol(self):
        return {"format": CHECKPOINT_FORMAT, "schema_id": self.state_only_schema_id,
                "action_dim": 76, "current_state_dim": 44, "action_horizon": 40,
                "input_size": [224, 448], "state_source": "measured_current_normalized",
                "state_index": 0, "visual_state_route": "none", "text_state_route": "none",
                "action_state_route": "additive_hidden_linear", "detach_measured_state": True,
                "pose_prediction_head": False, "pose_prediction_loss": False,
                "action_source": "normalized_past_plus_0.5_gaussian_all76",
                "action_layout": "motion64,inspire_left_measured6,inspire_right_measured6",
                "hand_target_time_offset": 0,
                "hand_target_offsets": [0, 39],
                "hand_target_is_issued_command": False,
                "normalization_strategy": NORMALIZATION_STRATEGY,
                "training_initialization": "FLUX.2-klein-base-4B-and-base-derived-action114",
                "parent_task_checkpoint_loaded": False, "initial_optimizer_source": "fresh",
                "base_action_initializer_is_task_trained": False,
                "image_source": "0.5*current_latent+sqrt(0.75)*standard_gaussian",
                "image_prior_squared_coefficient": 0.25,
                "action_history_dropout_probability": 0.0,
                "action_fm_weight": 1.0, "action_ic_weight": 0.5, "action_ic_steps": 1,
                "image_ic_weight": 1.0, "image_ic_steps": 1, "ic_loss": "l1",
                "visual_loss_weight": 0.5, "action_loss_weight": 1.0,
                "action_to_visual_gradients": False, "knowledge_insulation": True,
                "action_image_context": "noise_idm", "future_image_token_slots": True,
                "aura": False, "upstream_source_sha256": dict(UPSTREAM_SHA256)}

    def _append_proprio_to_context(self, context, context_mask, proprio):
        require(proprio is None and self.proprio_encoder is None, "Numerical state reached a visual/text context helper")
        return context, context_mask

    def _append_proprio_to_context_if_enabled(self, context, context_mask, proprio, source):
        del source
        return self._append_proprio_to_context(context, context_mask, proprio)

    @staticmethod
    def _without_state(sample):
        return {key: value for key, value in sample.items() if key not in STATE_KEYS}

    def _current_measured_state(self, value, *, sample=None, inference=False):
        require(torch.is_tensor(value) and value.is_floating_point(), "CURRENT normalized state must be a floating tensor")
        if not inference and value.ndim == 3:
            require(value.shape[1] > 0, "Empty state time axis")
            value = value[:, 0, :]
        elif inference and value.ndim == 1:
            value = value.unsqueeze(0)
        require(value.ndim == 2 and value.shape[-1] == 44 and value.shape[0] > 0,
                "State must be current [B,44]; training may supply [B,T,44], inference never a future sequence")
        if inference:
            require(value.shape[0] == 1, "Native FLUX.2 action inference uses one observation at a time")
        require(bool(torch.isfinite(value).all()), "Current state contains NaN/Inf")
        if sample is not None:
            for key in ("proprio_dim_is_pad", "state_dim_is_pad"):
                mask = sample.get(key)
                if mask is not None:
                    require(tuple(mask.shape) == tuple(value.shape) and not bool(mask.any()),
                            "The task requires all forty-four real state dimensions, not dimension padding")
            mask = sample.get("proprio_is_pad", sample.get("state_is_pad"))
            if mask is not None:
                require(mask.ndim in (1, 2), "State time padding must be [B] or [B,T]")
                current_mask = mask[:, 0] if mask.ndim == 2 else mask
                require(tuple(current_mask.shape) == (value.shape[0],) and not bool(current_mask.any()),
                        "Current measured state is padded")
        # Measurements are not a learnable prefix graph. Only state_encoder
        # parameters learn from this observation, independently per objective.
        return value.detach().clone()

    @contextmanager
    def action_state_scope(self, current=None):
        existing = _CURRENT_ACTION_STATE.get()
        if existing is not None:
            require(existing.owner is self._state_only_owner, "Nested action state belongs to a different model")
            if current is not None:
                require(torch.equal(current, existing.current), "A nested call changed the current observation")
            yield existing
            return
        require(current is not None, "Missing CURRENT measured state; never infer a zero/future condition")
        scope = _ActionStateScope(self._state_only_owner, current)
        token = _CURRENT_ACTION_STATE.set(scope)
        try:
            yield scope
        finally:
            _CURRENT_ACTION_STATE.reset(token)

    def _scope_for_sample(self, sample):
        value = sample.get("proprio")
        if value is None:
            scope = _CURRENT_ACTION_STATE.get()
            require(scope is not None and scope.owner is self._state_only_owner,
                    "Training requires normalized CURRENT sample['proprio']")
            return None
        return self._current_measured_state(value, sample=sample)

    def build_inputs_flux2(self, sample, tiled=False):
        self.validate_state_only_route()
        safe = self._without_state(sample)
        inputs = super().build_inputs_flux2(safe, tiled=tiled)
        require(tuple(inputs["action"].shape[1:]) == (40, 76)
                and inputs["past_action"] is not None
                and tuple(inputs["past_action"].shape) == tuple(inputs["action"].shape),
                "Current task needs matching forty-step past/future 76D action chunks")
        require(tuple(inputs["ref_image_latents"].shape[1:]) == (392, 128)
                and tuple(inputs["target_latent"].shape) == tuple(inputs["ref_image_latents"].shape),
                "Two224px views must form native224x448 FLUX.2 current/future token grids")
        inputs["_state_only_visual_owner"] = self._state_only_owner
        return inputs

    def _checked_inputs(self, inputs):
        if inputs is not None:
            require(inputs.get("_state_only_visual_owner") is self._state_only_owner,
                    "Only this model's state-free native visual inputs may be reused")
            require(not STATE_KEYS.intersection(inputs), "Numerical state was smuggled into a visual workspace")
        return inputs

    def _pack_flux2_action_workspace(self, video_pre, video_kv_cache):
        workspace = super()._pack_flux2_action_workspace(video_pre, video_kv_cache)
        workspace["_state_only_visual_owner"] = self._state_only_owner
        return workspace

    def _checked_workspace(self, workspace):
        if workspace is not None:
            require(workspace.get("_state_only_visual_owner") is self._state_only_owner,
                    "A shared visual workspace must come from this state-free model")
            def detached(value):
                if torch.is_tensor(value):
                    return not value.requires_grad and value.grad_fn is None
                if isinstance(value, dict):
                    return all(detached(item) for item in value.values())
                if isinstance(value, (tuple, list)):
                    return all(detached(item) for item in value)
                return True
            require(detached(workspace["video_kv_cache"]), "Action conditioning received a live visual graph despite KI")
        return workspace

    def _refresh_flux2_trainable_condition(self, inputs, sample):
        del sample
        return self._checked_inputs(inputs)

    def iter_training_losses(self, sample, tiled=False):
        self.validate_state_only_route()
        current = self._scope_for_sample(sample)
        iterator = super().iter_training_losses(self._without_state(sample), tiled=tiled)
        try:
            while True:
                # Reset BEFORE yielding to the trainer/backward, not after.
                # Native checkpoint recomputation is inside MoT blocks and
                # never re-enters action pre_dit, so no live scope is needed.
                try:
                    with self.action_state_scope(current):
                        item = next(iterator)
                except StopIteration:
                    return
                yield item
        finally:
            iterator.close()

    def _training_loss_flux2(self, sample, tiled=False, include_action_ic=True, inputs=None,
                             action_video_workspace=None, return_action_workspace=False):
        self.validate_state_only_route()
        with self.action_state_scope(self._scope_for_sample(sample)):
            return super()._training_loss_flux2(self._without_state(sample), tiled=tiled,
                include_action_ic=include_action_ic, inputs=self._checked_inputs(inputs),
                action_video_workspace=self._checked_workspace(action_video_workspace), return_action_workspace=return_action_workspace)

    def _training_loss_flux2_action_ic(self, sample, tiled=False, inputs=None, action_video_workspace=None):
        self.validate_state_only_route()
        with self.action_state_scope(self._scope_for_sample(sample)):
            return super()._training_loss_flux2_action_ic(self._without_state(sample), tiled=tiled,
                inputs=self._checked_inputs(inputs), action_video_workspace=self._checked_workspace(action_video_workspace))

    def _training_loss_flux2_image_ic(self, sample, tiled=False, inputs=None):
        self.validate_state_only_route()
        return super()._training_loss_flux2_image_ic(self._without_state(sample), tiled=tiled,
                                                    inputs=self._checked_inputs(inputs))

    def _bind_inference(self, method, args, kwargs):
        bound = inspect.signature(method).bind(self, *args, **kwargs)
        defaults = {"num_inference_steps": 1, "action_init_noise_strength": 0.5,
                    "action_init_noise_type": "additive", "action_init_noise_dim_mask": [True] * 76,
                    "clamp_action_init_after_noise": False}
        for key, value in defaults.items():
            if key not in bound.arguments:
                bound.arguments[key] = value
        return bound

    @wraps(ImageWAM.infer_action)
    def infer_action(self, *args, **kwargs):
        bound = self._bind_inference(ImageWAM.infer_action, args, kwargs)
        return ImageWAM.infer_action(*bound.args, **bound.kwargs)

    @wraps(ImageWAM.infer)
    def infer(self, *args, **kwargs):
        bound = self._bind_inference(ImageWAM.infer, args, kwargs)
        return ImageWAM.infer(*bound.args, **bound.kwargs)

    @wraps(ImageWAM.infer_action_flux2)
    def infer_action_flux2(self, *args, **kwargs):
        self.validate_state_only_route()
        bound = self._bind_inference(ImageWAM.infer_action_flux2, args, kwargs)
        bound.apply_defaults()
        current = self._current_measured_state(bound.arguments.pop("proprio"), inference=True)
        require(int(bound.arguments["action_horizon"]) == 40, "Expected forty-step whole-body action prediction")
        history = bound.arguments["action_init"]
        require(torch.is_tensor(history) and tuple(history.shape) in {(40, 76), (1, 40, 76)}
                and bool(torch.isfinite(history).all()), "Inference requires finite normalized40x76 recent action history")
        require(float(bound.arguments["action_init_noise_strength"]) == 0.5
                and bound.arguments["action_init_noise_type"] == "additive"
                and tuple(bound.arguments["action_init_noise_dim_mask"] or ()) == (True,) * 76
                and bound.arguments["clamp_action_init_after_noise"] is False,
                "Inference must match training: history+0.5*Gaussian on all76 dimensions, without clipping")
        image = bound.arguments["input_image"]
        require(tuple(image.shape) in {(3, 224, 448), (1, 3, 224, 448)}, "Expected two-view224x448 current image")
        bound.arguments["proprio"] = None
        self.action_expert.last_action_condition = None
        with self.action_state_scope(current) as scope:
            result = ImageWAM.infer_action_flux2(*bound.args, **bound.kwargs)
            require(scope.calls >= int(result["num_inference_steps"]) > 0,
                    "Native action solver did not consume the CURRENT action-side state")
            require(tuple(result["action"].shape) == (40, 76)
                    and bool(torch.isfinite(result["action"]).all()), "Native solver returned invalid whole-body actions")
        self.last_action_condition = self.action_expert.last_action_condition
        return {**result, "state_only_visual_state_route": "none", "state_only_action_state_source": "measured_current",
                "state_only_action_condition_calls": scope.calls, "state_only_action_dim": 76}

    @wraps(ImageWAM.infer_video_flux2)
    def infer_video_flux2(self, *args, **kwargs):
        self.validate_state_only_route()
        bound = inspect.signature(ImageWAM.infer_video_flux2).bind(self, *args, **kwargs)
        # Video prediction deliberately ignores numerical state, even when a
        # caller shares observation kwargs with action inference.
        bound.arguments["proprio"] = None
        if "num_inference_steps" not in bound.arguments:
            bound.arguments["num_inference_steps"] = 1
        return ImageWAM.infer_video_flux2(*bound.args, **bound.kwargs)

    def infer_joint(self, *args, **kwargs):
        raise NotImplementedError("Use native FLUX.2 infer_action and infer_video_flux2 independently; legacy joint sampler is not this task's route")

    def save_checkpoint(self, path, optimizer=None, step=None):
        self.validate_state_only_route()
        require(step is None or (type(step) is int and step >= 0), "Invalid checkpoint optimizer step")
        payload = {"checkpoint_format": CHECKPOINT_FORMAT, "state_only_delta": self.state_only_protocol(),
                   "mot": self.mot.state_dict(), "step": step, "torch_dtype": str(self.torch_dtype),
                   "initialization_audit": self.compatible_action_initialization_audit}
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def validate_checkpoint_payload(self, payload):
        require(isinstance(payload, dict) and payload.get("checkpoint_format") == CHECKPOINT_FORMAT,
                "Restore requires the whole-body checkpoint schema, not a legacy/native/C-Pred model")
        require(payload.get("state_only_delta") == self.state_only_protocol(), "Checkpoint action/state/source/route protocol differs")
        require(isinstance(payload.get("initialization_audit"), dict), "Checkpoint lacks initialization provenance")
        state = payload.get("mot")
        expected = self.mot.state_dict()
        require(isinstance(state, dict) and set(state) == set(expected), "Strict MoT checkpoint keys differ, including action state_encoder")
        for key, target in expected.items():
            require(torch.is_tensor(state[key]) and state[key].shape == target.shape and state[key].dtype == target.dtype,
                    f"Strict checkpoint tensor schema differs: {key}")
            if state[key].is_floating_point() or state[key].is_complex():
                require(bool(torch.isfinite(state[key]).all()), f"Non-finite checkpoint tensor: {key}")
        require(not {"proprio_encoder", "pose_head", "action_aura_router"}.intersection(payload),
                "Checkpoint contains a forbidden visual-state/pose/AURA module")
        return payload

    def load_checkpoint(self, path, optimizer=None):
        self.validate_state_only_route()
        payload = self.validate_checkpoint_payload(torch.load(path, map_location="cpu", weights_only=False))
        require(optimizer is None or "optimizer" in payload, "Training resume requested without optimizer state")
        self.mot.load_state_dict(payload["mot"], strict=True)
        self.compatible_action_initialization_audit = payload["initialization_audit"]
        if optimizer is not None:
            optimizer.load_state_dict(payload["optimizer"])
        return payload


def adopt_state_only_delta(model, *, action_state_dim=44, expected_action_dim=76,
                           expected_action_horizon=40, state_only_init_seed=42,
                           state_only_schema_id=SCHEMA_ID):
    """Register only new input/output/state projections in an initialized MoT."""
    require(type(model) is ImageWAM and type(model.action_expert) is ActionDiTFlux2,
            "Only the unchanged native ImageWAM and ActionDiTFlux2 can be adopted")
    require(model.proprio_encoder is None and model.proprio_dim is None and not model.pack_proprio_after_text,
            "The base factory must construct a state-free visual branch from the start")
    require((int(action_state_dim), int(expected_action_dim), int(expected_action_horizon)) == (44, 76, 40),
            "This task fixes action76/state44/horizon40")
    require(model.action_expert.action_dim == 76, "Configure the native action expert for76 dimensions before adoption")
    parameter = next(model.action_expert.parameters())
    model.__class__ = StateOnlyDeltaImageWAM
    model.action_expert.__class__ = StateOnlyActionDiTFlux2
    model.action_state_dim = model.action_expert.action_state_dim = 44
    model.expected_action_horizon = 40
    model.state_only_schema_id = str(state_only_schema_id)
    model.state_only_init_seed = int(state_only_init_seed)
    model._state_only_owner = object()
    model.action_expert._state_only_owner = model._state_only_owner
    hidden = int(model.action_expert.hidden_dim)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(int(state_only_init_seed))
        model.action_expert.action_encoder = nn.Linear(76, hidden)
        model.action_expert.head.linear = nn.Linear(hidden, 76, bias=False)
        model.action_expert.state_encoder = nn.Linear(44, hidden)
        # The canonical FLUX.2 partial initializer has no output modulation.
        # Initialize it explicitly, not from the base factory's incidental RNG
        # stream. Complete7D/14D initializers overwrite it with their shared key.
        model.action_expert.head.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 2 * hidden, bias=False))
    for module in (model.action_expert.action_encoder, model.action_expert.head.linear,
                   model.action_expert.state_encoder, model.action_expert.head.adaLN_modulation):
        module.to(device=parameter.device, dtype=parameter.dtype)
    model.action_expert.last_action_condition = None
    model.last_action_condition = None
    model.compatible_action_initialization_audit = {"source": None, "status": "new_action_layers_not_yet_loaded",
        "newly_initialized": sorted(PARTIAL_NEW_ACTION_KEYS), "init_seed": int(state_only_init_seed)}
    model.validate_state_only_route()
    return model


def _file_identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _byte_equal(left, right):
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    a = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    b = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(a, b))


def load_compatible_action_initializer(model, path, *, expected_sha256):
    """Load a user-pinned 114-tensor FLUX.2 base-derived partial initializer.

    Generate this artifact directly from the FLUX.2 base, not task weights.
    The explicit expected digest binds this run to the caller's local artifact;
    no digest or storage path from an author's machine is required.
    Every expected shared key must be present with exact shape. The actual
    artifact is float32 and the production expert is BF16: this one explicit
    dtype conversion is recorded and verified, not a silently skipped layer.
    Only the six declared new projections may be absent; they remain
    bit-identical to their freshly seeded factory initialization.
    """
    expected_sha256 = validate_expected_sha256(expected_sha256)
    source = Path(path).expanduser().resolve(strict=True)
    require(source.is_file(), "Base action initializer must be a regular file")
    before = _file_identity(source)
    actual_sha256 = file_sha256(source)
    require(actual_sha256 == expected_sha256,
            "Base action initializer SHA256 differs from the explicitly expected digest")
    require(_file_identity(source) == before, "Base initializer changed during hashing")
    payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    require(_file_identity(source) == before, "Base initializer changed while loading")
    require(isinstance(payload, dict) and set(payload) == {"state_dict", "meta"},
            "Only the original base-derived partial initializer is allowed; no task/optimizer checkpoint")
    state, metadata = payload["state_dict"], payload["meta"]
    require(isinstance(state, dict) and len(state) == 114,
            "The base initializer must contain exactly 114 shared tensors")
    require(isinstance(metadata, dict)
            and set(metadata) == set(PARTIAL_INITIALIZER_META) | {"source"}
            and all(metadata[key] == value for key, value in PARTIAL_INITIALIZER_META.items())
            and isinstance(metadata["source"], str)
            and Path(metadata["source"]).name == "flux-2-klein-base-4b.safetensors",
            "Base initializer conversion metadata does not match the audited FLUX.2 artifact")
    allowed = {"double_blocks", "double_stream_modulation_img", "single_blocks",
               "single_stream_modulation", "time_in"}
    require(all(isinstance(key, str) for key in state), "Initializer contains a non-string tensor key")
    require({key.split(".", 1)[0] for key in state} <= allowed,
            "Base initializer contains a non-shared/task-specific tensor")
    require(model.action_expert.action_dim == 76 and model.action_expert.action_state_dim == 44
            and model.state_only_init_seed == 42, "Expected the freshly seeded action76/state44 expert")
    expected = model.action_expert.state_dict()
    require(set(expected) - set(state) == set(PARTIAL_NEW_ACTION_KEYS)
            and not set(state) - set(expected),
            "Initializer keys must equal all shared action keys; only six new projections may be absent")
    new_before = {key: expected[key].detach().cpu().clone() for key in PARTIAL_NEW_ACTION_KEYS}
    retained, conversions = {}, []
    for key, tensor in state.items():
        target = expected[key]
        require(torch.is_tensor(tensor) and tensor.layout == torch.strided
                and tensor.device.type != "meta", f"Invalid shared initializer tensor: {key}")
        require(tensor.shape == target.shape,
                f"Shared action tensor shape/dtype mismatch, never silently skipped: {key}")
        require(tensor.dtype == target.dtype or
                (tensor.dtype == torch.float32 and target.dtype == torch.bfloat16),
                f"Unreviewed shared initializer dtype conversion: {key}")
        if tensor.is_floating_point() or tensor.is_complex():
            require(bool(torch.isfinite(tensor).all()), f"Non-finite shared initializer tensor: {key}")
        retained[key] = tensor.to(dtype=target.dtype, device="cpu")
        require(bool(torch.isfinite(retained[key]).all()), f"Shared initializer overflow after dtype conversion: {key}")
        if tensor.dtype != target.dtype:
            conversions.append({"key": key, "source_dtype": str(tensor.dtype), "target_dtype": str(target.dtype)})
    require(_file_identity(source) == before, "Base initializer changed before parameter loading")
    result = model.action_expert.load_state_dict(retained, strict=False)
    require(set(result.missing_keys) == set(PARTIAL_NEW_ACTION_KEYS) and not result.unexpected_keys,
            "Partial action initialization omitted keys beyond the six declared new projections")
    actual = model.action_expert.state_dict()
    for key, tensor in retained.items():
        require(_byte_equal(actual[key], tensor), f"Shared base action tensor was not copied bit-for-bit: {key}")
    for key, tensor in new_before.items():
        require(_byte_equal(actual[key], tensor), f"Fresh task projection was overwritten by initializer: {key}")
    require(_file_identity(source) == before, "Base initializer changed during exact-copy verification")
    audit = {
        "source": str(source), "source_sha256": actual_sha256,
        "expected_sha256": expected_sha256,
        "source_format": "flux2_shared_partial_initializer_v1", "source_action_dim": None,
        "target_action_dim": 76, "current_state_dim": 44,
        "retained_keys": sorted(state), "retained_tensor_count": 114,
        "ignored_source_projections": [], "newly_initialized": sorted(PARTIAL_NEW_ACTION_KEYS),
        "new_parameter_shapes": {key: list(actual[key].shape) for key in sorted(PARTIAL_NEW_ACTION_KEYS)},
        "source_meta": dict(metadata), "init_seed": 42,
        "status": "shared_layers_loaded_exactly_new_projections_recorded",
        "shared_tensors_copied_bitwise_after_dtype_conversion": True,
        "dtype_policy": "same dtype or explicitly recorded float32-to-bfloat16",
        "dtype_conversions": conversions, "new_projections_unchanged": True,
        "source_is_task_trained": False, "parent_task_checkpoint_loaded": False,
        "optimizer_loaded": False, "scheduler_loaded": False, "training_step_resumed": False,
    }
    model.compatible_action_initialization_audit = audit
    return audit


def create_imagewam_flux2_wholebody_delta(*, action_initializer_sha256, action_state_dim=44, expected_action_dim=76,
        expected_action_horizon=40, state_only_init_seed=42,
        state_only_schema_id=SCHEMA_ID, **kwargs):
    """Fresh FLUX.2 base video + base-derived shared action layers, no parent task."""
    from imagewam.runtime import create_imagewam_flux2_klein

    forbidden = {"parent_checkpoint", "additional_training_epochs", "resume", "checkpoint", "pretrained_checkpoint"}
    require(not forbidden.intersection(kwargs),
            "Fresh whole-body initialization rejects parent/continuation/resume arguments")
    action_initializer_sha256 = validate_expected_sha256(action_initializer_sha256)
    for cls in (ImageWAM, ActionDiTFlux2):
        source = Path(inspect.getfile(cls))
        require(file_sha256(source) == UPSTREAM_SHA256[source.name], f"Unreviewed native source: {source}")
    require(state_only_schema_id == SCHEMA_ID, "Whole-body checkpoint schema cannot masquerade as another architecture")
    require(int(state_only_init_seed) == 42, "Fresh task projections must use seed42")
    require(kwargs.get("variant") == "klein-base-4b", "Use the FLUX.2-klein-base-4B base variant")
    require(Path(str(kwargs.get("flux2_model_path"))).name == "flux-2-klein-base-4b.safetensors",
            "Initialize video from the FLUX.2 base safetensors, not a trained task checkpoint")
    require(kwargs.get("proprio_dim") in (None, 44), "proprio_dim must represent the real44D current state")
    require(not bool(kwargs.get("pack_proprio_after_text", False)), "State text-packing must be disabled")
    require(not bool(kwargs.get("zero_init_action_head", False)) and float(kwargs.get("action_head_init_scale", 1.0)) == 1.0,
            "New whole-body output head must use its recorded nonzero initialization")
    action_config = kwargs.get("action_dit_config")
    require(action_config is not None and int(action_config.get("action_dim", -1)) == 76,
            "Configure native action_dit_config.action_dim=76")
    initializer = kwargs.get("action_dit_pretrained_path")
    require(initializer is not None, "Fresh whole-body training requires a base-derived action114 initializer")
    # Validate the artifact before allocating the full backbone, and validate it
    # again during its actual strict shared-only load after construction.
    require(file_sha256(Path(initializer).expanduser().resolve(strict=True)) == action_initializer_sha256,
            "Action initializer SHA256 differs from action_initializer_sha256")
    kwargs["proprio_dim"] = None
    kwargs["pack_proprio_after_text"] = False
    kwargs["action_dit_pretrained_path"] = None
    model = adopt_state_only_delta(create_imagewam_flux2_klein(**kwargs),
        action_state_dim=action_state_dim, expected_action_dim=expected_action_dim,
        expected_action_horizon=expected_action_horizon, state_only_init_seed=state_only_init_seed,
        state_only_schema_id=state_only_schema_id)
    load_compatible_action_initializer(model, initializer, expected_sha256=action_initializer_sha256)
    model.model_paths["action_dit"] = str(Path(initializer).expanduser().resolve(strict=True))
    model.model_paths["parent_task"] = None
    model.validate_state_only_route()
    return model
