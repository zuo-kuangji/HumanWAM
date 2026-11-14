"""Lazy official Fast-WAM adapter; native Gaussian sources and task action spaces.

No Torch, CUDA, FastWAM or model download is imported by this module. Released
LIBERO and RoboTwin profiles have different action/state/camera contracts.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

UPSTREAM_COMMIT = "7faa71108368fbb3b6885649f112af607427a2d4"
RELEASE_REVISION = "8eaceeb24c3cc92ff2a9c9a9d266a4941b836705"
PROFILES = {
    "libero_uncond_2cam224": {
        "task": "libero_uncond_2cam224_1e-4",
        "action_dim": 7,
        "state_dim": 8,
        "horizon": 32,
        "cameras": 2,
        "action_space": "libero_eef_delta7_gripper",
        "stats_sha256": "30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638",
        "checkpoint_sha256": "1000437cfcf55c000094f79a2600634c502bcb5b492476b94bf8509883a49579",
    },
    "robotwin_uncond_3cam_384": {
        "task": "robotwin_uncond_3cam_384_1e-4",
        "action_dim": 14,
        "state_dim": 14,
        "horizon": 32,
        "cameras": 3,
        "action_space": "robotwin_joint14",
        "stats_sha256": "7a02c46cfc8c5e746c0afbe41fca73f723eda34cbc083f8ca54f76d8f7468095",
        "checkpoint_sha256": "776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63",
    },
}
_ASSET_LOAD_LOCK = threading.Lock()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hash(path, expected):
    if expected is not None and _sha256(path) != expected:
        raise ValueError(f"FastWAM asset SHA256 mismatch: {path}")


def _validate_stats(stats, profile):
    """Check the native JSON structure before allocating any model memory."""
    import math

    for group, dim in (("action", profile["action_dim"]), ("state", profile["state_dim"])):
        if not isinstance(stats.get(group), dict) or set(stats[group]) != {"default"}:
            raise ValueError(f"Release profile requires one {group}.default stats field")
        field = stats[group]["default"]
        for name in ("global_min", "global_max", "global_mean", "global_std"):
            values = field.get(name)
            if not isinstance(values, list) or len(values) != dim:
                raise ValueError(f"{group}.default.{name} must have {dim} values")
            if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
                raise ValueError("Statistics must contain finite numeric values")
    stepwise = stats["action"]["default"].get("stepwise_min")
    if not isinstance(stepwise, list) or len(stepwise) != profile["horizon"]:
        raise ValueError("Action statistics horizon does not match the release profile")
    if any(not isinstance(row, list) or len(row) != profile["action_dim"] for row in stepwise):
        raise ValueError("Stepwise action statistics have the wrong action dimension")


def _verified_stats_bytes(config, profile):
    payload = Path(config["stats_path"]).read_bytes()
    expected = config.get("stats_sha256")
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("FastWAM statistics SHA256 mismatch")
    _validate_stats(json.loads(payload), profile)
    return payload


def validate_local_assets(cache_root):
    """Only public, original Wan assets; never trigger native auto-downloads."""
    root = Path(cache_root).expanduser().resolve()
    wan = root / "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer = root / "Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"
    required = [
        wan / "Wan2.2_VAE.pth",
        wan / "models_t5_umt5-xxl-enc-bf16.pth",
        tokenizer / "tokenizer_config.json",
        tokenizer / "tokenizer.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Prepare local FastWAM assets first: " + ", ".join(missing))
    return root


@contextmanager
def _local_asset_environment(root):
    # ModelConfig reads these process variables while constructing components.
    # Restore them after construction so another policy's cache is not changed.
    with _ASSET_LOAD_LOCK:
        changes = {"DIFFSYNTH_MODEL_BASE_PATH": str(root), "DIFFSYNTH_SKIP_DOWNLOAD": "true"}
        previous = {key: os.environ.get(key) for key in changes}
        os.environ.update(changes)
        try:
            yield
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _validate_model_payload(model, payload):
    """The official loader is permissive; require a complete task checkpoint."""
    if not isinstance(payload, dict) or not isinstance(payload.get("mot"), dict):
        raise ValueError("FastWAM requires a complete task checkpoint with mot weights")
    modules = [("mot", model.mot)]
    if model.proprio_encoder is not None:
        modules.append(("proprio_encoder", model.proprio_encoder))
    for name, module in modules:
        actual = payload.get(name)
        expected = module.state_dict()
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Incomplete or incompatible FastWAM {name} checkpoint keys")
        if any(tuple(actual[key].shape) != tuple(expected[key].shape) for key in expected):
            raise ValueError(f"Incompatible FastWAM {name} checkpoint tensor shape")


class _NativeRuntime:
    def __init__(self, config, profile):
        stats_bytes = _verified_stats_bytes(config, profile)
        root = Path(config["source_dir"]).expanduser().resolve()
        if not (root / "src/fastwam/runtime.py").is_file():
            raise FileNotFoundError(f"Official FastWAM checkout missing: {root}")
        cache = validate_local_assets(config["model_cache_root"])
        _check_hash(config["checkpoint"], config.get("checkpoint_sha256"))
        source = str(root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        native = importlib.import_module("fastwam.runtime")
        if Path(native.__file__).resolve() != root / "src/fastwam/runtime.py":
            raise RuntimeError("A different FastWAM installation shadows the pinned source")

        import torch
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        from fastwam.utils.config_resolvers import register_default_resolvers
        from hydra.utils import instantiate

        register_default_resolvers()
        cfg = self._compose_config(root, profile)
        if cfg.model._target_ != "fastwam.runtime.create_fastwam":
            raise ValueError("Release adapter only supports the official uncond FastWAM factory")
        cfg.model.load_text_encoder = True
        cfg.model.skip_dit_load_from_pretrain = True
        cfg.model.action_dit_pretrained_path = None
        cfg.model.redirect_common_files = False
        device = config.get("device", "cuda")
        dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[config.get("dtype", "bf16")]
        with _local_asset_environment(cache):
            model = instantiate(cfg.model, model_dtype=dtype, device=device)
        if (
            model.action_expert.action_dim != profile["action_dim"]
            or model.proprio_dim != profile["state_dim"]
        ):
            raise ValueError("Native model dimensions disagree with the maintained release profile")
        payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=True)
        _validate_model_payload(model, payload)
        model.mot.load_state_dict(payload["mot"], strict=True)
        model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        del payload
        self.model = model.to(device).eval().requires_grad_(False)
        self.processor = instantiate(cfg.data.train.processor).eval()
        # Native conversion reads a path. Give it exactly the verified bytes,
        # rather than rereading a potentially replaced task statistics file.
        with tempfile.NamedTemporaryFile(suffix=".json") as verified:
            verified.write(stats_bytes)
            verified.flush()
            stats = load_dataset_stats_from_json(verified.name)
        self.processor.set_normalizer_from_stats(stats)
        self.torch = torch

    def _compose_config(self, root, profile):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
            return compose(config_name="train", overrides=["task=" + profile["task"]])

    def _prepare_image(self, image):
        torch = self.torch
        image = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=self.model.device, dtype=self.model.torch_dtype)
        return image * (2.0 / 255.0) - 1.0

    def infer(self, image, state, prompt, **kwargs):
        torch = self.torch
        image = self._prepare_image(image)
        batch = {"state": {"default": torch.as_tensor(state.copy()).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        with torch.inference_mode():
            result = self.model.infer_action(
                prompt=prompt,
                input_image=image,
                proprio=batch["state"]["default"],
                **kwargs,
            )
        action = result["action"]
        if action.ndim == 2:
            action = action.unsqueeze(0)
        normalizer = self.processor.normalizer.normalizers["action"]["default"]
        return normalizer.backward(action.to(dtype=torch.float32, device="cpu")).numpy()


def _resize_rgb(image, width, height, *, crop):
    import numpy as np
    from PIL import Image

    image = Image.fromarray(image)
    if crop:
        scale = max(width / image.width, height / image.height)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR
        )
        left, top = (image.width - width) // 2, (image.height - height) // 2
        image = image.crop((left, top, left + width, top + height))
    else:
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(image)


class FastWAMPolicy:
    """Current RGB and raw task state in; native denormalized task actions out."""

    def __init__(self, config: dict, *, runtime_factory=None):
        self.config = dict(config)
        name = config.get("profile")
        if name not in PROFILES:
            raise ValueError(
                "Choose a maintained FastWAM release profile; no implicit SONIC conversion"
            )
        self.profile_name = name
        self.profile = dict(PROFILES[name])
        for field in ("action_dim", "state_dim", "horizon"):
            if field in config and int(config[field]) != self.profile[field]:
                raise ValueError(
                    f"{field} differs from the trained release; retrain a matching model"
                )
        if (
            config.get("action_schema", self.profile["action_space"])
            != self.profile["action_space"]
        ):
            raise ValueError("FastWAM task actions cannot be relabeled as SONIC or another robot")
        if config.get("method", "gaussian_fm") != "gaussian_fm":
            raise ValueError(
                "Official FastWAM is Gaussian FM; Delta-flow needs a separate audited patch"
            )
        for field in ("checkpoint", "stats_path"):
            path = Path(config[field]).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            self.config[field] = str(path)
        if Path(self.config["checkpoint"]).suffix != ".pt":
            raise ValueError("FastWAM task checkpoint must be a .pt file")
        _verified_stats_bytes(self.config, self.profile)
        if config.get("dtype", "bf16") not in ("bf16", "fp32"):
            raise ValueError("FastWAM dtype must be bf16 or fp32")
        if int(config.get("num_inference_steps", 10)) < 1:
            raise ValueError("num_inference_steps must be positive")
        shift = float(config.get("sigma_shift", 5.0))
        if not 0 < shift < float("inf"):
            raise ValueError("sigma_shift must be positive and finite")
        self._runtime_factory = runtime_factory or _NativeRuntime
        self._runtime = None

    def reset(self):
        # No action queue or planned history: the caller owns real execution.
        return None

    def predict(self, observation: dict) -> dict:
        import numpy as np

        if not isinstance(observation, dict):
            raise TypeError("Observation must be a dict")
        forbidden = {
            "past_actions",
            "past_action",
            "future_state",
            "future_images",
            "state_normalized",
        }
        if forbidden.intersection(observation):
            raise ValueError(
                "Provide current images and raw task state; native Gaussian source has no action history"
            )
        prompt = observation.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Provide a nonempty task instruction")
        images = observation.get("images")
        if (
            not isinstance(images, (list, tuple, np.ndarray))
            or len(images) != self.profile["cameras"]
        ):
            raise ValueError(
                f"Provide exactly {self.profile['cameras']} current cameras in profile order"
            )
        arrays = [np.asarray(value) for value in images]
        if any(
            x.dtype != np.uint8 or x.ndim != 3 or x.shape[2] != 3 or min(x.shape[:2]) < 1
            for x in arrays
        ):
            raise ValueError("Each current camera must be nonempty uint8 HWC RGB")
        state = np.asarray(observation.get("state"), dtype=np.float32)
        if state.shape != (self.profile["state_dim"],) or not np.isfinite(state).all():
            raise ValueError(f"Raw state must be finite [{self.profile['state_dim']}]")
        if self.profile_name.startswith("libero"):
            image = np.concatenate([_resize_rgb(x, 224, 224, crop=True) for x in arrays], axis=1)
        else:
            head = _resize_rgb(arrays[0], 320, 256, crop=False)
            wrists = [_resize_rgb(x, 160, 128, crop=False) for x in arrays[1:]]
            image = np.concatenate([head, np.concatenate(wrists, axis=1)], axis=0)
        if self._runtime is None:
            self._runtime = self._runtime_factory(self.config, self.profile)
        output = self._runtime.infer(
            image,
            state,
            "A video recorded from a robot's point of view executing the following instruction: "
            + prompt,
            action_horizon=self.profile["horizon"],
            num_inference_steps=int(self.config.get("num_inference_steps", 10)),
            sigma_shift=float(self.config.get("sigma_shift", 5.0)),
            compile_action_infer=bool(self.config.get("compile_action_infer", False)),
            seed=self.config.get("seed"),
            rand_device="cpu",
            tiled=False,
            text_cfg_scale=1.0,
        )
        actions = np.asarray(output, dtype=np.float32)
        expected = (1, self.profile["horizon"], self.profile["action_dim"])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Native FastWAM output must be finite {expected}")
        actions = actions[0].copy()
        if self.profile_name.startswith("libero"):
            # Exact official evaluation convention: [0,1] -> [-1,1], then invert.
            actions[:, -1] = -(actions[:, -1] * 2 - 1)
            if self.config.get("binarize_gripper", True):
                actions[:, -1] = np.sign(actions[:, -1])
        return {
            "actions": actions.tolist(),
            "action_space": self.profile["action_space"],
            "normalized": False,
            "horizon": self.profile["horizon"],
            "native": {
                "profile": self.profile_name,
                "upstream_commit": UPSTREAM_COMMIT,
                "method": "gaussian_fm",
                "robot_execution_verified": False,
            },
        }


def build_policy(config):
    return FastWAMPolicy(config)
