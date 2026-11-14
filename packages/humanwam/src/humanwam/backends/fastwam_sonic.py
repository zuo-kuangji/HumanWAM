"""G1 feedback76/H16 Fast-WAM, with the GR00T SONIC modality contract.

This requires a separately trained checkpoint. LIBERO/RoboTwin release weights
cannot produce motion tokens or Inspire feedback by padding or relabeling.
Heavy native imports occur only in the first valid prediction.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from .fastwam import (
    UPSTREAM_COMMIT,
    _NativeRuntime,
    _sha256,
    _verified_stats_bytes,
)

STATE_FIELDS = (
    ("left_leg", 6), ("right_leg", 6), ("waist", 3),
    ("left_arm", 7), ("right_arm", 7),
    ("left_hand", 6), ("right_hand", 6), ("projected_gravity", 3),
)
ACTION_FIELDS = (("motion_token", 64), ("left_hand_joints", 6), ("right_hand_joints", 6))
CAMERAS = ["ego_view", "chest"]
LANGUAGE_KEY = "annotation.human.task_description"
CONTRACT = {
    "version": 1,
    "profile": "feedback76_h16",
    "action_space": "sonic_motion64_inspire12",
    "action_dim": 76,
    "state_dim": 44,
    "horizon": 16,
    "camera_order": CAMERAS,
    "hand_semantics": "measured_inspire6_per_hand",
    "method": "gaussian_fm",
    "sigma_shift": 1.0,
    "upstream_commit": UPSTREAM_COMMIT,
}


def validate_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("FastWAM SONIC checkpoint manifest must be a mapping")
    for key, expected in CONTRACT.items():
        if contract.get(key) != expected:
            raise ValueError(f"FastWAM SONIC checkpoint manifest mismatch: {key}")
    hz = contract.get("action_frequency_hz")
    if isinstance(hz, bool) or not isinstance(hz, (int, float)) or not 0 < hz < math.inf:
        raise ValueError("Checkpoint manifest needs finite positive action_frequency_hz")
    for key in ("checkpoint_sha256", "stats_sha256"):
        digest = contract.get(key)
        if not isinstance(digest, str) or len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError(f"Checkpoint manifest requires {key}")
    return copy.deepcopy(contract)


def checkpoint_manifest(checkpoint, stats_path, *, action_frequency_hz):
    """Bind an operator-declared, trained feedback76 checkpoint to its stats.

    This records provenance, not evidence that training or a rollout succeeded.
    Tensor key/shape validation still runs in the native loader.
    """
    profile = dict(CONTRACT)
    _verified_stats_bytes({"stats_path": stats_path}, profile)
    return validate_contract({
        **CONTRACT,
        "action_frequency_hz": action_frequency_hz,
        "checkpoint_sha256": _sha256(checkpoint),
        "stats_sha256": _sha256(stats_path),
    })


class _SonicNativeRuntime(_NativeRuntime):
    def _compose_config(self, root, profile):
        from .fastwam_sonic_training import compose_sonic_config

        return compose_sonic_config(root)

    def _prepare_image(self, images):
        # Reuse the actual native processor's float32 ToTensor/Resize sequence.
        # PIL uint8 resizing would round before the model sees the pixels.
        torch = self.torch
        views = []
        for image in images:
            value = torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0)
            for transform in self.processor.val_transforms:
                value = transform(value)
            views.append(value)
        # Concatenation already has the configured 224x448 final aspect ratio,
        # so the dataset's final resize/crop is identity for this recipe.
        image = (torch.cat(views, dim=-1) - 0.5) / 0.5
        return image.to(device=self.model.device, dtype=self.model.torch_dtype)


class FastWAMSonicPolicy:
    """Same nested observation and batched action keys as GR00T feedback76.

    Only B=1, T=1 observations and H16 output are supported. The caller owns
    execution timing and hands; reset does not reset SONIC or a robot.
    """

    def __init__(self, config, *, runtime_factory=None):
        self.config = dict(config)
        manifest_path = Path(config["checkpoint_manifest"]).expanduser().resolve(strict=True)
        self.manifest = validate_contract(json.loads(manifest_path.read_text()))
        for key in ("profile", "action_dim", "state_dim", "horizon", "method", "sigma_shift"):
            if key in config and config[key] != CONTRACT[key]:
                raise ValueError(f"FastWAM SONIC {key} must match its trained checkpoint")
        if config.get("action_schema", CONTRACT["action_space"]) != CONTRACT["action_space"]:
            raise ValueError("SONIC feedback76 cannot be relabeled as stock SONIC78")
        for key in ("checkpoint", "stats_path"):
            self.config[key] = str(Path(config[key]).expanduser().resolve(strict=True))
        if Path(self.config["checkpoint"]).suffix != ".pt":
            raise ValueError("FastWAM SONIC checkpoint must be a complete native .pt")
        for key in ("checkpoint_sha256", "stats_sha256"):
            if key in config and config[key] != self.manifest[key]:
                raise ValueError(f"Configuration conflicts with checkpoint manifest: {key}")
            self.config[key] = self.manifest[key]
        self.profile = dict(CONTRACT)
        _verified_stats_bytes(self.config, self.profile)
        if config.get("dtype", "bf16") not in ("bf16", "fp32"):
            raise ValueError("FastWAM dtype must be bf16 or fp32")
        steps = config.get("num_inference_steps", 10)
        if type(steps) is not int or steps < 1:
            raise ValueError("num_inference_steps must be a positive integer")
        self._runtime_factory = runtime_factory or _SonicNativeRuntime
        self._runtime = None

    def get_modality_config(self):
        # The dedicated SONIC wire server wraps these in native ModalityConfig
        # markers, so GR00T PolicyClient reconstructs real native dataclasses.
        return {
            "video": {"delta_indices": [0], "modality_keys": CAMERAS.copy()},
            "state": {"delta_indices": [0], "modality_keys": [k for k, _ in STATE_FIELDS]},
            "action": {
                "delta_indices": list(range(16)),
                "modality_keys": [k for k, _ in ACTION_FIELDS],
                "action_configs": [
                    {"rep": "ABSOLUTE", "type": "NON_EEF", "format": "DEFAULT"}
                    for _ in ACTION_FIELDS
                ],
            },
            "language": {"delta_indices": [0], "modality_keys": [LANGUAGE_KEY]},
        }

    def _observation(self, observation):
        import numpy as np

        if not isinstance(observation, dict) or set(observation) != {"video", "state", "language"}:
            raise ValueError("Use GR00T nested video/state/language observations")
        video, state, language = (observation[k] for k in ("video", "state", "language"))
        for group, keys in ((video, CAMERAS), (state, [k for k, _ in STATE_FIELDS]),
                            (language, [LANGUAGE_KEY])):
            if not isinstance(group, dict) or set(group) != set(keys):
                raise ValueError(f"Observation modality must contain exactly {keys}")
        images = []
        for key in CAMERAS:
            image = np.asarray(video[key])
            if (image.dtype != np.uint8 or image.ndim != 5 or image.shape[:2] != (1, 1)
                    or image.shape[-1] != 3 or min(image.shape[2:4]) < 1):
                raise ValueError(f"video.{key} must be current uint8 RGB [1,1,H,W,3]")
            images.append(image[0, 0].copy())
        pieces = []
        for key, dim in STATE_FIELDS:
            value = np.asarray(state[key])
            if value.dtype != np.float32 or value.shape != (1, 1, dim) or not np.isfinite(value).all():
                raise ValueError(f"state.{key} must be raw finite float32 [1,1,{dim}]")
            if key in ("left_hand", "right_hand") and ((value < 0) | (value > 1)).any():
                raise ValueError("Measured Inspire state must use [0,1], not 7D commands or 0..1000")
            pieces.append(value[0, 0])
        text = language[LANGUAGE_KEY]
        if (not isinstance(text, list) or len(text) != 1 or not isinstance(text[0], list)
                or len(text[0]) != 1 or not isinstance(text[0][0], str) or not text[0][0].strip()):
            raise ValueError("language task instruction must be a nonempty [[str]]")
        return tuple(images), np.concatenate(pieces), text[0][0]

    def get_action(self, observation, options=None):
        import numpy as np

        if options is not None and (not isinstance(options, dict) or options):
            raise ValueError("FastWAM SONIC supports only empty native request options")
        image, state, prompt = self._observation(observation)
        if self._runtime is None:
            self._runtime = self._runtime_factory(self.config, self.profile)
        output = self._runtime.infer(
            image, state,
            "A video recorded from a robot's point of view executing the following instruction: " + prompt,
            action_horizon=16, num_inference_steps=self.config.get("num_inference_steps", 10),
            sigma_shift=1.0, compile_action_infer=bool(self.config.get("compile_action_infer", False)),
            seed=self.config.get("seed"), rand_device="cpu", tiled=False, text_cfg_scale=1.0,
        )
        actions = np.asarray(output, dtype=np.float32)
        if actions.shape != (1, 16, 76) or not np.isfinite(actions).all():
            raise ValueError("Native SONIC FastWAM output must be finite [1,16,76]")
        # Native processor already denormalized these values. No gripper sign,
        # body reordering, token normalization or hand width conversion here.
        result = {}
        offset = 0
        for key, width in ACTION_FIELDS:
            result[key] = actions[:, :, offset:offset + width].copy()
            offset += width
        return result, {
            "profile": CONTRACT["profile"], "action_space": CONTRACT["action_space"],
            "normalized": False, "horizon": 16,
            "action_frequency_hz": self.manifest["action_frequency_hz"],
            "hand_semantics": CONTRACT["hand_semantics"], "robot_execution_verified": False,
        }

    def predict(self, observation):
        import numpy as np

        actions, info = self.get_action(observation)
        return {"actions": actions, "info": info,
                "action": np.concatenate([actions[k][0] for k, _ in ACTION_FIELDS], axis=-1)}

    def reset(self, options=None):
        if options is not None and not isinstance(options, dict):
            raise ValueError("reset options must be a mapping or null")
        return {}


def build_policy(config):
    return FastWAMSonicPolicy(config)


def build_train_command(config, source_dir):
    from .fastwam_sonic_training import build_train_command as build

    return build(config, source_dir)


def build_serve_command(config, source_dir):
    source = Path(source_dir).expanduser().resolve(strict=True)
    launcher = Path(config["launcher_config"]).expanduser().resolve(strict=True)
    port = config.get("port", 5555)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be in 1..65535")
    return {
        "argv": [config.get("python", "python3"), "-m", "humanwam.sonic_policy_server",
                 "--config", str(launcher), "--host", config.get("host", "127.0.0.1"),
                 "--port", str(port)],
        "cwd": str(source), "env": {},
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Record a trained feedback76/H16 checkpoint contract")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--action-frequency-hz", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    manifest = checkpoint_manifest(args.checkpoint, args.stats,
                                   action_frequency_hz=args.action_frequency_hz)
    with Path(args.output).open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
