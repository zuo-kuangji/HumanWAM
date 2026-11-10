"""Pinned Psi-Zero SONIC runtime, preserving checkpoint-owned transformations."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

UPSTREAM_COMMIT = "cda3fdfafe47cb6fa75400d013de1bd36e3cea05"
ACTION_SPACE = "psi0_sonic_native"
NECKLESS_PROFILE = "g1_neckless43_action78"
NATIVE_PROFILE = "checkpoint_native"


def validate_config(config: dict, *, checkpoint: bool = True) -> dict:
    config = dict(config)
    profile = config.get("embodiment_profile", NECKLESS_PROFILE)
    if not isinstance(profile, str) or profile not in {NECKLESS_PROFILE, NATIVE_PROFILE}:
        raise ValueError("Unknown Psi-Zero embodiment_profile")
    config["embodiment_profile"] = profile
    if profile == NECKLESS_PROFILE:
        for key, expected in (("raw_state_dim", 43), ("action_dim", 78)):
            value = config.get(key, expected)
            if type(value) is not int or value != expected:
                raise ValueError("The neckless G1 profile requires state43/action78; not ImageWAM44/76 or neck45/80")
            config[key] = expected
    if config.get("action_schema", ACTION_SPACE) != ACTION_SPACE:
        raise ValueError("Psi-Zero native SONIC actions are not ImageWAM feedback76")
    if checkpoint:
        root = Path(config.get("run_dir", "")).expanduser().resolve()
        step = config.get("ckpt_step")
        if type(step) is not int or step < 0:
            raise ValueError(
                "Native SONIC server requires an explicit nonnegative integer ckpt_step"
            )
        if not config.get("run_dir") or not root.is_dir():
            raise FileNotFoundError("run_dir must be a complete downloaded Psi-Zero run")
        for name in ("argv.txt", "run_config.json", f"checkpoints/ckpt_{step}"):
            if not (root / name).exists():
                raise FileNotFoundError(root / name)
        saved = json.loads((root / "run_config.json").read_text())
        model = saved.get("model", {})
        for key in ("action_dim", "action_chunk_size", "odim"):
            if type(model.get(key)) is not int or model[key] < 1:
                raise ValueError("Saved Psi-Zero model config lacks positive " + key)
        if profile == NECKLESS_PROFILE:
            if (model["odim"], model["action_dim"]) != (43, 78):
                raise ValueError("The neckless G1 profile needs a trained state43/action78 checkpoint; neck45/action80 weights cannot be relabeled or cropped")
            transforms = saved.get("data", {}).get("transform", {})
            for section, keys in (("repack", ("action_keys", "state_keys")),
                                  ("field", ("stat_action_keys", "stat_state_keys"))):
                for key in keys:
                    declared = transforms.get(section, {}).get(key, [])
                    if not isinstance(declared, list) or any(not isinstance(v, str) for v in declared):
                        raise ValueError("Checkpoint modality keys must be string lists")
                    if any("neck" in value.lower() for value in declared):
                        raise ValueError("The neckless G1 checkpoint cannot declare neck input or action channels")
            field = transforms.get("field", {})
            for key, width in (("state_min", 43), ("state_max", 43),
                               ("action_min", 78), ("action_max", 78)):
                if key in field and (not isinstance(field[key], list) or len(field[key]) != width):
                    raise ValueError("Neckless checkpoint normalization must match state43/action78: " + key)
        for requested, key in (
            ("action_dim", "action_dim"),
            ("horizon", "action_chunk_size"),
        ):
            if requested in config and config[requested] != model[key]:
                raise ValueError(f"Configured {requested} differs from saved checkpoint")
        config.update(
            run_dir=str(root),
            action_dim=model["action_dim"],
            horizon=model["action_chunk_size"],
            model_state_dim=model["odim"],
        )
    if "raw_state_dim" in config and (
        type(config["raw_state_dim"]) is not int or config["raw_state_dim"] < 1
    ):
        raise ValueError("raw_state_dim must be a positive exact integer")
    if checkpoint and config.get("raw_state_dim", 0) > config["model_state_dim"]:
        raise ValueError("raw_state_dim cannot exceed checkpoint model_state_dim")
    if checkpoint and "raw_state_dim" in config:
        # Native padding is valid only for genuinely zero-padded state channels.
        # The official Sonic80 release has real neck states in channels43/44;
        # padding an old43D observation would silently discard those inputs.
        field = saved.get("data", {}).get("transform", {}).get("field", {})
        low, high = field.get("state_min"), field.get("state_max")
        width, raw = config["model_state_dim"], config["raw_state_dim"]
        if isinstance(low, list) and isinstance(high, list) and len(low) == len(high) == width:
            if any(float(low[i]) != 0.0 or float(high[i]) != 0.0 for i in range(raw, width)):
                raise ValueError(
                    "Missing checkpoint state channels are measured inputs, not zero padding; use the full raw state layout"
                )
    steps = config.get("num_inference_steps", 8)
    if type(steps) is not int or steps < 1:
        raise ValueError("num_inference_steps must be a positive exact integer")
    config["num_inference_steps"] = steps
    if "horizon" in config and (type(config["horizon"]) is not int or config["horizon"] < 1):
        raise ValueError("horizon must be a positive integer")
    return config


def _run_native(server, prepared, steps):
    import torch

    with torch.inference_mode():
        result = server.model.predict_action(
            observations=prepared["imgs"],
            states=torch.from_numpy(prepared["obs"]).to(server.device),
            traj2ds=None,
            instructions=prepared["text_instructions"],
            num_inference_steps=steps,
            pooled_projections=prepared.get("pooled_projections"),
        )
    return result.float().detach().cpu().numpy()


class Psi0Policy:
    """Offline chunk prediction through the actual native SONIC Server internals.

    Server construction loads its model/config/transforms but does not call
    ``run`` or start an RTC control thread. ``predict`` is a fresh complete
    chunk, not an imitation of the asynchronous native RTC controller.
    """

    def __init__(self, config: dict, *, server_factory=None, message_factory=None, inference=None):
        self.config = validate_config(config)
        if "raw_state_dim" not in self.config:
            raise ValueError(
                "Declare the recorded raw_state_dim; checkpoint padding is not raw state layout"
            )
        self._server = None
        self._server_factory = server_factory
        self._message_factory = message_factory
        self._inference = inference or _run_native

    def _load(self):
        if self._server is not None:
            return self._server
        factory = self._server_factory
        if factory is None:
            module = importlib.import_module("psi.deploy.serve_psi0_sonic")
            if self.config.get("source_dir"):
                expected = (
                    Path(self.config["source_dir"]).resolve() / "src/psi/deploy/serve_psi0_sonic.py"
                )
                if Path(module.__file__).resolve() != expected:
                    raise ValueError(
                        "Installed Psi-Zero runtime does not come from the configured pinned source"
                    )
            factory = module.Server
        message = self._message_factory
        if message is None:
            message = importlib.import_module("psi.deploy.helpers").RequestMessage
        server = factory(
            policy="psi",
            run_dir=Path(self.config["run_dir"]),
            ckpt_step=self.config["ckpt_step"],
            device=self.config.get("device", "cuda:0"),
            enable_rtc=False,
            action_exec_horizon=None,
            dashboard=False,
        )
        if (server.Da, server.Tp) != (
            self.config["action_dim"],
            self.config["horizon"],
        ):
            raise ValueError("Loaded native Psi-Zero dimensions differ from run_config.json")
        if getattr(server.launch_cfg.model, "odim", None) != self.config["model_state_dim"]:
            raise ValueError("Loaded native Psi-Zero state dimension differs from run_config.json")
        keys = list(server.launch_cfg.data.transform.repack.image_keys)
        if len(keys) != 1:
            raise ValueError(
                "Pinned native SONIC preprocessing only consumes cam0; multi-camera checkpoints require their own runtime"
            )
        self.image_key = keys[0]
        self._message_factory = message
        self._server = server
        return server

    def predict(self, observation: dict[str, Any]) -> dict:
        import numpy as np

        forbidden = {
            "past_actions",
            "normalized_past_actions",
            "future_state",
            "actions",
            "future_images",
            "gt_action",
        }
        if forbidden.intersection(observation):
            raise ValueError(
                "Offline Psi-Zero chunk input must contain only current RGB/raw state/instruction"
            )
        image = np.asarray(observation.get("image_rgb"))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("image_rgb must be the current uint8 HWC image")
        states = np.asarray(observation.get("state_raw"), dtype=np.float32)
        if states.shape != (self.config["raw_state_dim"],) or not np.isfinite(states).all():
            raise ValueError("state_raw must match the explicitly declared raw_state_dim")
        instruction = observation.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a nonempty string")
        server = self._load()
        message = self._message_factory(
            image={self.image_key: image.copy()},
            instruction=instruction,
            history={},
            state={"states": states.copy()},
            condition={},
            gt_action=[],
            dataset_name=self.config.get("dataset_name", "sonic"),
            timestamp="0",
        )
        # Preserve exact native state normalization, checkpoint image transforms
        # and optional pooled text embeddings, with inference augmentation OFF.
        prepared = server._parse_obs_payload(message.serialize())
        prepared_state = np.asarray(prepared["obs"])
        if (prepared_state.shape != (1, 1, self.config["model_state_dim"])
                or not np.isfinite(prepared_state).all()):
            raise ValueError("Native preprocessed state shape/values differ from the checkpoint state dimension")
        normalized = np.asarray(
            self._inference(server, prepared, self.config["num_inference_steps"])
        )
        expected = (1, self.config["horizon"], self.config["action_dim"])
        if normalized.shape != expected or not np.isfinite(normalized).all():
            raise ValueError("Native Psi-Zero returned invalid normalized action shape/values")
        actions = np.asarray(server._postprocess_action(normalized[0]))
        if actions.shape != expected[1:] or not np.isfinite(actions).all():
            raise ValueError("Native checkpoint denormalization returned invalid actions")
        return dict(
            actions=actions,
            normalized=False,
            action_space=ACTION_SPACE,
            horizon=self.config["horizon"],
            native=dict(
                action_dim=self.config["action_dim"],
                embodiment_profile=self.config["embodiment_profile"],
                raw_state_dim=self.config["raw_state_dim"],
                checkpoint_step=self.config["ckpt_step"],
                upstream_commit=UPSTREAM_COMMIT,
                inference_mode="offline_full_chunk_no_rtc_thread",
                robot_execution_verified=False,
            ),
        )

    def reset(self):
        # No open-loop RTC execution counter/thread is created in this adapter.
        # Keep the loaded model, and clear the optional pending pseudo action.
        if self._server is not None:
            self._server._pending_init_prev = None
        return None


def build_policy(config: dict) -> Psi0Policy:
    return Psi0Policy(config)
