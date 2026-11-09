"""Lazy adapter for the pinned, official DiT4DiT native inference API.

DiT4DiT returns normalized trajectories. This adapter preserves those units
and does not present them as SONIC motion tokens or robot joint commands.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

UPSTREAM_COMMIT = "66a6f3a12e2c8740157c1e478795952c040d31dd"
ACTION_SCHEMA = "dit4dit_native_normalized"


def _value(obj, *names):
    for name in names:
        obj = obj[name] if isinstance(obj, dict) else getattr(obj, name)
    return obj


class DiT4DiTPolicy:
    """Expose the upstream ``predict_action(examples=...)`` without changing units.

    Inputs are one current RGB camera, a prompt, and the checkpoint's already
    normalized/encoded/padded state. Multiple images are rejected: the native
    Cosmos interface interprets image1 onward as FUTURE FRAMES, not cameras.
    """

    def __init__(self, config: dict, *, model_factory=None):
        if config.get("action_schema", ACTION_SCHEMA) != ACTION_SCHEMA:
            raise ValueError(
                "DiT4DiT native output is not SONIC76; choose dit4dit_native_normalized"
            )
        self.config = dict(config)
        checkpoint = Path(config["checkpoint"]).expanduser().resolve()
        if not checkpoint.is_file() or checkpoint.suffix != ".pt":
            raise FileNotFoundError("DiT4DiT needs an existing .pt checkpoint in RUN/checkpoints/")
        run = checkpoint.parents[1]
        for name in ("config.yaml", "dataset_statistics.json"):
            if not (run / name).is_file():
                raise FileNotFoundError(
                    f"DiT4DiT native checkpoint companion is missing: {run / name}"
                )
        injected_factory = model_factory is not None
        if model_factory is None:
            root = Path(config["upstream_root"]).expanduser().resolve()
            if not (root / "DiT4DiT/model/framework/base_framework.py").is_file():
                raise FileNotFoundError(f"Pinned DiT4DiT checkout not found: {root}")
            root_text = str(root)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            module = importlib.import_module("DiT4DiT.model.framework.base_framework")
            model_factory = module.baseframework.from_pretrained
        self.model = model_factory(str(checkpoint))
        # Use the upstream loader/strict state_dict matching and own config.
        if not injected_factory:
            if config.get("use_bf16", False):
                import torch

                self.model = self.model.to(torch.bfloat16)
            self.model = self.model.to(config.get("device", "cuda")).eval()
        action = _value(self.model.config, "framework", "action_model")
        self.action_dim = int(_value(action, "action_dim"))
        self.state_dim = int(_value(action, "state_dim"))
        self.horizon = int(_value(action, "action_horizon"))
        if min(self.action_dim, self.state_dim, self.horizon) < 1:
            raise ValueError("Checkpoint action/state/horizon dimensions must be positive")
        for name, actual in (
            ("action_dim", self.action_dim),
            ("state_dim", self.state_dim),
            ("horizon", self.horizon),
        ):
            if name in config and int(config[name]) != actual:
                raise ValueError(f"Configured {name} differs from the checkpoint ({actual})")
        data = _value(self.model.config, "datasets", "vla_data")
        image_size = (
            data.get("image_size") if isinstance(data, dict) else getattr(data, "image_size", None)
        )
        # The official LIBERO release omits image_size. Native predict_action
        # preserves the current RGB resolution in this case; do not invent a
        # training resolution or reject an otherwise valid published config.
        self.image_size = None if image_size is None else tuple(int(x) for x in image_size)
        if self.image_size is not None and (len(self.image_size) != 2 or min(self.image_size) < 1):
            raise ValueError("Checkpoint image_size must contain height,width")

    def reset(self):
        # The pinned upstream DiT4DiT model has no recurrent state/cache to reset.
        return None

    def predict(self, observation: dict) -> dict:
        import numpy as np
        from PIL import Image

        if not isinstance(observation, dict):
            raise TypeError("Observation must be a dictionary")
        if "past_actions" in observation or "past_action" in observation:
            raise ValueError("Native DiT4DiT is Gaussian-source and has no action-history input")
        images = observation.get("images")
        # The common offline CLI reads a dense [camera,H,W,C] NPZ array.
        if isinstance(images, np.ndarray):
            if images.ndim != 4 or images.shape[0] != 1:
                raise ValueError(
                    "NPZ images must be [1,H,W,3]; extra images mean future frames upstream"
                )
            images = [images[0]]
        if not isinstance(images, (list, tuple)) or len(images) != 1:
            raise ValueError(
                "Provide exactly one current RGB image; extra images mean future frames upstream"
            )
        image = np.asarray(images[0])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("The current image must be uint8 HWC RGB")
        image = Image.fromarray(image)
        if self.image_size is not None:
            height, width = self.image_size
            image = image.resize((width, height))
        prompt = observation.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty prompt for DiT4DiT")
        if "state" in observation:
            raise ValueError(
                "Use state_normalized after the checkpoint-specific state encoding/normalization"
            )
        state = np.asarray(observation.get("state_normalized"), dtype=np.float32)
        if state.shape == (self.state_dim,):
            state = state[None, :]
        if state.shape != (1, self.state_dim) or not np.isfinite(state).all():
            raise ValueError(f"state_normalized must be finite [1,{self.state_dim}]")
        result = self.model.predict_action(
            examples=[{"image": [image], "lang": prompt, "state": state}]
        )
        if not isinstance(result, dict) or "normalized_actions" not in result:
            raise ValueError("Native DiT4DiT did not return normalized_actions")
        actions = np.asarray(result["normalized_actions"])
        if actions.shape != (1, self.horizon, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError("Native DiT4DiT output shape/finite check failed")
        return {
            "actions": actions[0].tolist(),
            "action_space": ACTION_SCHEMA,
            "normalized": True,
            "horizon": self.horizon,
            "native": {
                "output_key": "normalized_actions",
                "action_dim": self.action_dim,
                "state_dim": self.state_dim,
                "upstream_commit": UPSTREAM_COMMIT,
                "robot_execution_verified": False,
            },
        }


def build_policy(config: dict):
    return DiT4DiTPolicy(config)
