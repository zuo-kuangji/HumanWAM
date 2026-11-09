"""Verified OpenPI LIBERO contract, using the pinned native inference transforms.

History is RAW executed actions, never normalized twice or inferred from planned
actions. The patched OpenPI factory loads the checkpoint's own normalization
assets and aliases action statistics to ``past_actions`` for Delta-flow.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

DELTA_CONFIGS = frozenset({"pi05_libero_a2a_jax_lora", "pi05_libero_a2a_jax_lora_warmup_cosine"})
BASELINE_CONFIGS = frozenset({"pi05_libero"})
LIBERO_SCHEMA = "libero_eef_delta7_state8"


def validate_config(config: dict[str, Any], *, require_checkpoint: bool = True) -> dict:
    config = dict(config)
    method = config.get("method", "delta_flow")
    if method not in {"delta_flow", "gaussian"}:
        raise ValueError("pi05 method must be delta_flow or gaussian")
    expected = DELTA_CONFIGS if method == "delta_flow" else BASELINE_CONFIGS
    name = config.get("config_name")
    if name not in expected:
        raise ValueError(f"Unsupported {method} config {name!r}; supported: {sorted(expected)}")
    if config.get("schema", LIBERO_SCHEMA) != LIBERO_SCHEMA:
        raise ValueError("This native OpenPI adapter supports LIBERO7/state8, not SONIC76/state44")
    if config.get("action_horizon", 10) != 10 or config.get("action_dim", 7) != 7:
        raise ValueError("The pinned LIBERO checkpoint contract is horizon10/action7")
    steps = config.get("num_steps", 1 if method == "delta_flow" else 10)
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ValueError("num_steps must be a positive integer")
    if method == "delta_flow" and steps != 1:
        raise ValueError("This released LIBERO Delta-flow configuration uses one Euler update")
    if require_checkpoint:
        checkpoint = Path(config.get("checkpoint_dir", "")).expanduser()
        if not config.get("checkpoint_dir") or not checkpoint.is_dir():
            raise FileNotFoundError(
                "checkpoint_dir must point to a downloaded native OpenPI checkpoint"
            )
        if not (checkpoint / "assets").is_dir():
            raise FileNotFoundError("Checkpoint normalization assets/ are required")
        if method == "delta_flow" and not (checkpoint / "params").exists():
            raise ValueError(
                "The maintained A2A patch supports JAX params checkpoints; not an arbitrary Torch release"
            )
        config["checkpoint_dir"] = str(checkpoint.resolve())
    config.update(
        method=method, num_steps=steps, action_horizon=10, action_dim=7, schema=LIBERO_SCHEMA
    )
    return config


class Pi05Policy:
    """Persistent, native OpenPI policy; raw native observations remain explicit."""

    def __init__(self, config: dict[str, Any]):
        self.config = validate_config(config)
        self._native = None

    def _load(self):
        if self._native is not None:
            return self._native
        native_configs = importlib.import_module("openpi.training.config")
        policy_config = importlib.import_module("openpi.policies.policy_config")
        training = native_configs.get_config(self.config["config_name"])
        delta = self.config["method"] == "delta_flow"
        if not training.model.pi05 or bool(getattr(training.model, "a2a_enabled", False)) != delta:
            raise ValueError(
                "Installed OpenPI model method differs from the requested checkpoint contract"
            )
        if training.model.action_horizon != 10 or training.model.action_dim != 32:
            raise ValueError(
                "Installed OpenPI model dimensions differ from the pinned configuration"
            )
        if delta and not callable(getattr(policy_config, "input_norm_stats_for_model", None)):
            raise RuntimeError("Apply the pinned A2A patch before loading Delta-flow")
        self._native = policy_config.create_trained_policy(
            training,
            self.config["checkpoint_dir"],
            sample_kwargs={"num_steps": self.config["num_steps"]},
            default_prompt=self.config.get("default_prompt"),
            pytorch_device=self.config.get("device"),
        )
        return self._native

    def predict(self, observation: dict[str, Any]) -> dict:
        import numpy as np

        inputs = dict(observation)
        forbidden = {
            "actions",
            "action",
            "future_actions",
            "future_state",
            "normalized_past_actions",
        }
        if forbidden.intersection(inputs):
            raise ValueError(
                "Prediction inputs may not contain target actions/future state/normalized history"
            )
        for key in ["observation/image", "observation/wrist_image"]:
            image = np.asarray(inputs.get(key))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{key} must be HWC uint8 RGB")
            inputs[key] = image.copy()
        state = np.asarray(inputs.get("observation/state"), dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("observation/state must be raw finite state8")
        inputs["observation/state"] = state.copy()
        if "prompt" not in inputs and not self.config.get("default_prompt"):
            raise ValueError("A prompt or configured default_prompt is required")
        if "prompt" in inputs and (
            not isinstance(inputs["prompt"], str) or not inputs["prompt"].strip()
        ):
            raise ValueError("prompt must be a nonempty instruction")
        if self.config["method"] == "delta_flow":
            past = np.asarray(inputs.get("past_actions"), dtype=np.float32)
            valid = np.asarray(inputs.get("past_action_valid_mask"))
            if past.shape != (10, 7) or not np.isfinite(past).all():
                raise ValueError(
                    "Delta-flow needs raw, oldest-to-newest, strictly executed past_actions[10,7]"
                )
            if (
                valid.shape != (10,)
                or valid.dtype != np.bool_
                or not valid.any()
                or np.any(valid[:-1] & ~valid[1:])
            ):
                raise ValueError(
                    "past_action_valid_mask must be bool[10], with an optional invalid prefix and nonempty valid suffix"
                )
            inputs["past_actions"] = past.copy()
            inputs["past_action_valid_mask"] = valid.copy()
        elif {"past_actions", "past_action_valid_mask", "past_action"}.intersection(inputs):
            raise ValueError("Gaussian baseline takes no action-history source")
        result = self._load().infer(inputs)
        actions = np.asarray(result.get("actions"))
        if actions.shape != (10, 7) or not np.isfinite(actions).all():
            raise RuntimeError(
                "Native OpenPI returned an invalid action chunk; expected finite[10,7]"
            )
        # Native output transforms have already unnormalized and sliced to7D.
        return {**result, "actions": actions.copy()}

    def reset(self) -> None:
        if self._native is not None:
            reset = getattr(self._native, "reset", None)
            if callable(reset):
                reset()


def build_policy(config: dict[str, Any]) -> Pi05Policy:
    return Pi05Policy(config)


def pad_executed_history(history, horizon: int = 10):
    """Same prefix rule as the pinned LIBERO evaluator; at least one past action."""
    import numpy as np

    values = np.asarray(history, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 7 or len(values) < 1 or not np.isfinite(values).all():
        raise ValueError(
            "Provide at least one finite, already executed, raw seven-dimensional action"
        )
    if horizon != 10:
        raise ValueError("The pinned native LIBERO configuration requires10 history steps")
    values = values[-horizon:]
    if len(values) < horizon:
        values = np.concatenate(
            [np.repeat(values[:1], horizon - len(values), axis=0), values], axis=0
        )
    return values, np.ones(horizon, dtype=np.bool_)
