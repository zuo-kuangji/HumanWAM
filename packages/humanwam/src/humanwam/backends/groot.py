"""Lazy adapter for pinned native GR00T N1.7; no model imports at module import."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

# Names whose native enum values differ from their lowercase spelling in the
# pinned N1.7 source. These are aliases, never fallback embodiments.
_EMBODIMENT_VALUES = {
    "xdof": "xdof_relative_eef_relative_joint",
    "xdof_subtask": "xdof_relative_eef_relative_joint_subtask",
    "real_g1": "real_g1_relative_eef_relative_joints",
    "real_r1_pro_sharpa": "real_r1_pro_sharpa_relative_eef",
    "real_r1_pro_sharpa_human": "real_r1_pro_sharpa_relative_eef_human",
    "real_r1_pro_sharpa_maxinsights": "real_r1_pro_sharpa_relative_eef_maxinsights",
    "real_r1_pro_sharpa_mecka": "real_r1_pro_sharpa_relative_eef_mecka",
    "unitree_g1": "unitree_g1_full_body_with_waist_height_nav_cmd",
    "libero_panda": "libero_sim",
}


def _required(config: dict, key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"GR00T requires a nonempty {key}")
    return value


def validate_local_checkpoint(config: dict) -> dict | None:
    """Reject an absent local embodiment before importing/loading a model.

    HF identifiers and incomplete local fixture directories remain owned by
    native loading; this guard does not fetch metadata from the network.
    """
    root = Path(_required(config, "model_path")).expanduser()
    if not root.is_dir():
        return None
    path = root / "processor_config.json"
    if not path.is_file():
        path = root / "processor/processor_config.json"
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("processor_kwargs"), dict):
        raise ValueError(f"Local GR00T processor config has an invalid structure: {path}")
    modalities = raw["processor_kwargs"].get("modality_configs")
    if not isinstance(modalities, dict) or not modalities:
        raise ValueError(f"Local GR00T processor lacks modality_configs: {path}")
    requested = config.get("embodiment_tag", "UNITREE_G1_SONIC")
    if not isinstance(requested, str) or not requested.strip():
        raise ValueError("embodiment_tag must be a nonempty native GR00T tag")
    normalized = requested.strip().lower()
    normalized = _EMBODIMENT_VALUES.get(normalized, normalized)
    if normalized not in modalities:
        raise ValueError(
            f"GR00T checkpoint {root} does not contain requested embodiment {requested!r} "
            f"({normalized}). Supported tags: {', '.join(sorted(modalities))}. "
            "Choose a checkpoint trained for the requested embodiment; no fallback was applied."
        )
    return {
        "processor_config": str(path.resolve()),
        "embodiment_tag": normalized,
        "supported_tags": sorted(modalities),
    }


class GrootPolicy:
    """Preserves the actual native modality schema instead of guessing a flat action space.

    ``predict`` accepts the native nested ``video/state/language`` dictionary.
    Native arrays carry both batch and temporal axes. The checkpoint processor,
    not the adapter, selects normalization, camera names, and prediction horizon.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.checkpoint_metadata = validate_local_checkpoint(config)
        self.output_format = config.get("output_format", "native")
        if self.output_format not in {"native", "feedback76"}:
            raise ValueError("GR00T output_format must be native or feedback76")
        if self.output_format == "feedback76" and config.get("embodiment_tag") != "NEW_EMBODIMENT":
            raise ValueError("feedback76 requires its separately trained NEW_EMBODIMENT checkpoint")
        cls = importlib.import_module("gr00t.policy.gr00t_policy").Gr00tPolicy
        self.native = cls(
            embodiment_tag=config.get("embodiment_tag", "UNITREE_G1_SONIC"),
            model_path=_required(config, "model_path"),
            device=config.get("device", "cuda:0"),
            strict=True,
        )
        if self.output_format == "feedback76":
            # Opt-in for checkpoints actually trained with this recipe. It does
            # not reinterpret the stock SONIC 78D hand-command policy as 76D.
            modalities = self.native.get_modality_config()
            action = modalities["action"]
            if list(action.modality_keys) != [
                "motion_token",
                "left_hand_joints",
                "right_hand_joints",
            ]:
                raise ValueError("Checkpoint does not declare the feedback76 action-key order")
            if list(action.delta_indices) != list(range(16)):
                raise ValueError("feedback76 requires a checkpoint trained at horizon16")

    def get_action(self, observation: dict, options: dict | None = None) -> tuple[dict, dict]:
        """Native BasePolicy-compatible call for GR00T's ZeroMQ PolicyServer."""
        if set(observation) != {"video", "state", "language"}:
            raise ValueError("Use the native GR00T video/state/language schema")
        return self.native.get_action(observation, options)

    def predict(self, observation: dict) -> dict:
        action, info = self.get_action(observation)
        result = {"actions": action, "info": info}
        if self.output_format == "feedback76":
            import numpy as np

            pieces = []
            for key, size in [
                ("motion_token", 64),
                ("left_hand_joints", 6),
                ("right_hand_joints", 6),
            ]:
                value = np.asarray(action[key])
                if (
                    value.shape != (1, 16, size)
                    or value.dtype != np.float32
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(
                        f"feedback76 checkpoint output {key} must be finite float32 [1,16,{size}]"
                    )
                pieces.append(value[0])
            result["action"] = np.concatenate(pieces, axis=-1)
        return result

    def reset(self, options: dict | None = None) -> dict:
        return self.native.reset(options)

    def get_modality_config(self) -> dict:
        return self.native.get_modality_config()


def build_policy(config: dict) -> GrootPolicy:
    return GrootPolicy(config)


def build_train_command(config: dict, source_dir: str | Path) -> dict:
    """Build the real pinned upstream fine-tuning command; never execute it here."""
    root = Path(source_dir).resolve()
    entry = root / "examples/finetune.sh"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    argv = [
        "bash",
        str(entry),
        "--base-model-path",
        _required(config, "base_model_path"),
        "--dataset-path",
        _required(config, "dataset_path"),
        "--embodiment-tag",
        config.get("embodiment_tag", "UNITREE_G1_SONIC"),
        "--output-dir",
        _required(config, "output_dir"),
    ]
    if config.get("modality_config_path"):
        path = Path(config["modality_config_path"]).resolve()
        if not path.is_file() or path.suffix != ".py":
            raise ValueError(
                "Native GR00T training modality_config_path must be an existing Python config"
            )
        argv += ["--modality-config-path", str(path)]
    count = int(config.get("num_gpus", 8))
    steps = int(config.get("max_steps", 10000))
    batch = int(config.get("global_batch_size", 128))
    if min(count, steps, batch) < 1:
        raise ValueError("num_gpus, max_steps and global_batch_size must be positive")
    env = {
        "NUM_GPUS": str(count),
        "MAX_STEPS": str(steps),
        "GLOBAL_BATCH_SIZE": str(batch),
        "USE_WANDB": "0",
        "SAVE_STEPS": str(int(config.get("save_steps", 1000))),
    }
    return {"argv": argv, "cwd": str(root), "env": env}


def build_serve_command(config: dict, source_dir: str | Path, python: str | None = None) -> dict:
    """Use native ZeroMQ/msgpack SONIC transport, not an HTTP imitation."""
    root = Path(source_dir).resolve()
    validate_local_checkpoint(config)
    entry = root / "gr00t/eval/run_gr00t_server.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    argv = [
        python or config.get("python", "python"),
        str(entry),
        "--model-path",
        _required(config, "model_path"),
        "--embodiment-tag",
        config.get("embodiment_tag", "UNITREE_G1_SONIC"),
        "--device",
        config.get("device", "cuda:0"),
        "--host",
        config.get("host", "127.0.0.1"),
        "--port",
        str(int(config.get("port", 5555))),
    ]
    return {"argv": argv, "cwd": str(root), "env": {}}
