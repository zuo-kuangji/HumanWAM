"""Native whole-body ImageWAM Delta-flow and exact published release adapters."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path

SCHEMA = "motion64_measured_inspire6x2_state44"


def validate_observation(observation):
    import numpy as np

    forbidden = {"future_state", "future_image", "normalized_past_actions", "actions"}
    if forbidden.intersection(observation):
        raise ValueError("Use only current images/state and raw strictly-past action history")
    state = np.asarray(observation["current_state"], dtype=np.float32)
    history = np.asarray(observation["past_actions"], dtype=np.float32)
    if state.shape != (44,) or not np.isfinite(state).all():
        raise ValueError("current_state must be finite raw measured state44")
    if history.ndim != 2 or history.shape[1] != 76 or not np.isfinite(history).all():
        raise ValueError("past_actions must be finite raw [N,76], N may be0 at episode start")
    if np.any((state[29:41] < 0) | (state[29:41] > 1)) or np.any(
        (history[:, 64:] < 0) | (history[:, 64:] > 1)
    ):
        raise ValueError("Measured hand feedback must already be in physical [0,1]")
    views = {}
    for key in ("ego_rgb", "chest_rgb"):
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"{key} must be current uint8 HWC RGB")
        views[key] = image
    return dict(views, current_state=state, past_actions=history)


def prepare_inputs(observation, horizon, normalizer, resize):
    import numpy as np
    import torch

    raw = validate_observation(observation)
    state, history = raw["current_state"], raw["past_actions"]
    views = [resize(raw[key]) for key in ("ego_rgb", "chest_rgb")]
    if not len(history):
        history = np.zeros((horizon, 76), dtype=np.float32)
    else:
        history = normalizer.forward(history[-horizon:], "action")
        if len(history) < horizon:
            history = np.concatenate(
                (np.repeat(history[:1], horizon - len(history), axis=0), history)
            )
    image = (torch.cat(views, dim=-1) * 2 - 1)[None]
    return (
        image,
        torch.from_numpy(normalizer.forward(state, "state")[None]),
        torch.from_numpy(history[None]),
    )


class WholeBodyDeltaPolicy:
    """Load once; retain exact H16/H40 checkpoint and normalization contracts."""

    def __init__(self, config):
        if config.get("schema", SCHEMA) != SCHEMA:
            raise ValueError("ImageWAM feedback76 cannot use SONIC78 command-hand normalization")
        self.config = dict(config)
        self.horizon = int(config.get("horizon", 16))
        if self.horizon not in (16, 40):
            raise ValueError(
                "Use H16 for new models; H40 only for explicitly matching legacy models"
            )
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from hydra.utils import instantiate
        from imagewam.datasets.wholebody import WholeBodyNormalizer, resize_rgb_frame_uint8
        from omegaconf import OmegaConf

        model_config = OmegaConf.load(self.config["model_config"])
        kwargs = model_config.model
        if kwargs._target_ != "imagewam.wholebody_delta.create_imagewam_flux2_wholebody_delta":
            raise ValueError("Use the maintained wholebody model factory or a pinned release entry")
        if int(kwargs.expected_action_horizon) != self.horizon:
            raise ValueError("Configured inference horizon differs from the trained model")
        stats_path = Path(self.config.get("stats_path", model_config.data.train.stats_path))
        stats = json.loads(stats_path.read_text())
        self.normalizer = WholeBodyNormalizer(stats)
        from imagewam.datasets.wholebody import DEFAULT_PROMPT

        prompt_hash = hashlib.sha256(
            DEFAULT_PROMPT.format(task=stats["instruction"]).encode()
        ).hexdigest()
        if not Path(self.config["text_cache"]).name.startswith(prompt_hash + ".qwen3_flux2_len128"):
            raise ValueError(
                "Text cache filename does not match the task instruction in the statistics"
            )
        self.resize = resize_rgb_frame_uint8
        model = instantiate(
            kwargs, device=self.config.get("device", "cuda:0"), model_dtype=torch.bfloat16
        )
        model.load_checkpoint(self.config["checkpoint"])
        model.eval().requires_grad_(False)
        cached = torch.load(self.config["text_cache"], map_location="cpu", weights_only=True)
        hidden, mask = cached["text_hidden_states"], cached["text_attention_mask"]
        if tuple(hidden.shape) != (128, 7680) or tuple(mask.shape) != (128,):
            raise ValueError("Use the instruction-only text cache belonging to this task")
        device = self.config.get("device", "cuda:0")
        self.hidden = hidden[None].to(device=device, dtype=torch.bfloat16)
        self.mask = mask[None].to(device=device, dtype=torch.bool)
        self._model = model

    def predict(self, observation):
        import numpy as np
        import torch

        validate_observation(observation)
        self._load()
        image, state, history = prepare_inputs(
            observation, self.horizon, self.normalizer, self.resize
        )
        with torch.inference_mode():
            result = self._model.infer_action(
                prompt=None,
                input_image=image,
                proprio=state,
                context=self.hidden,
                context_mask=self.mask,
                action_init=history,
                action_horizon=self.horizon,
                num_inference_steps=1,
                action_init_noise_strength=0.5,
                action_init_noise_type="additive",
                action_init_noise_dim_mask=[True] * 76,
                clamp_action_init_after_noise=False,
                seed=observation.get("seed"),
                rand_device="cpu",
            )
        normalized = result["action"].detach().float().cpu().numpy()
        if normalized.shape != (self.horizon, 76) or not np.isfinite(normalized).all():
            raise ValueError("ImageWAM returned an invalid action chunk")
        action = self.normalizer.backward(normalized, "action")
        if not np.isfinite(action).all():
            raise ValueError("ImageWAM denormalization returned nonfinite actions")
        return dict(
            actions=action,
            normalized=False,
            action_space=SCHEMA,
            horizon=self.horizon,
            motion64=action[:, :64],
            left_hand_state6=action[:, 64:70],
            right_hand_state6=action[:, 70:76],
        )

    def reset(self):
        return None


class PublishedDeltaPolicy:
    """Use the release's own verified persistent predictor; do not alter its protocol."""

    ENTRIES = {
        "trash": ("infer_trash.py", "TrashPolicy"),
        "pillow": ("infer_pillow.py", "PillowPolicy"),
    }

    def __init__(self, config):
        self.config = dict(config)
        release = Path(config["release"]).expanduser().resolve(strict=True)
        name, cls_name = self.ENTRIES[config["release_kind"]]
        path = release / name
        manifest = json.loads((release / "release_manifest.json").read_text())
        row = next((row for row in manifest["files"] if row["path"] == name), None)
        if row is None or hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("The release inference entry must match its pinned manifest")
        self.horizon = int(manifest["protocol"]["action_horizon"])
        if self.horizon != int(config.get("horizon", 16)):
            raise ValueError("Release horizon differs from the requested policy")
        module_name = "humanwam_release_" + hashlib.sha256(str(path).encode()).hexdigest()[:16]
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self.native = getattr(module, cls_name)(
            release, config["base_model"], config["ae"], config.get("device", "cuda:0")
        )

    def predict(self, observation):
        validate_observation(observation)
        result = self.native.predict(
            observation["ego_rgb"],
            observation["chest_rgb"],
            observation["current_state"],
            observation["past_actions"],
            seed=observation.get("seed"),
        )
        return dict(
            result,
            actions=result["action76"],
            normalized=False,
            action_space=SCHEMA,
            horizon=self.horizon,
        )

    def reset(self):
        return None


class LegacyClothesPolicy(WholeBodyDeltaPolicy):
    """Exact published Inspire76/p25 final25 release; keep its H40/parent quantiles."""

    def __init__(self, config):
        if config.get("horizon") != 40:
            raise ValueError("The published clothes final25 checkpoint explicitly requires H40")
        super().__init__(config)

    def _load(self):
        if self._model is not None:
            return
        import torch

        root = Path(self.config["release"]).expanduser().resolve(strict=True)
        for path in (root, root / "source", root / "source/src", root / "third_party/flux2/src"):
            sys.path.insert(0, str(path))
        verifier_path = root / "verify_release.py"
        name = "humanwam_clothes_verifier_" + hashlib.sha256(str(root).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(name, verifier_path)
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        integrity = verifier.verify(root, None, include_weight=True)
        config = json.loads((root / "config.json").read_text())
        kwargs = dict(config["model"])
        expected = "imagewam.sonic_inspire_delta.create_imagewam_flux2_inspire_delta"
        if kwargs.pop("_target_") != expected:
            raise ValueError("Not the published Inspire76/p25 clothes factory")
        dataset = importlib.import_module("sonic_inspire_dataset")
        if Path(dataset.__file__).resolve() != root / "sonic_inspire_dataset.py":
            raise ValueError("Clothes dataset runtime was shadowed by another policy")
        self.normalizer = dataset.SonicNormalizer(
            json.loads((root / "inputs/dataset_stats.json").read_text())
        )
        self.resize = dataset.resize_rgb_frame_uint8
        kwargs.pop("device", None)
        kwargs.pop("model_dtype", None)
        kwargs.update(
            flux2_src_path=str(root / "third_party/flux2"),
            flux2_model_path=str(Path(self.config["base_model"]).resolve(strict=True)),
            ae_model_path=str(Path(self.config["ae"]).resolve(strict=True)),
            parent_checkpoint=None,
            action_dit_pretrained_path=None,
            additional_training_epochs=25,
            qwen3_model_spec=None,
            load_text_encoder=False,
        )
        factory = importlib.import_module(
            "imagewam.sonic_inspire_delta"
        ).create_imagewam_flux2_inspire_delta
        device = self.config.get("device", "cuda:0")
        model = factory(**kwargs, device=device, model_dtype=torch.bfloat16)
        payload = model.load_checkpoint(integrity["checkpoint_path"])
        if payload.get("step") != integrity["checkpoint"]["step"]:
            raise ValueError("Loaded clothes checkpoint step differs from the verified release")
        model.eval().requires_grad_(False)
        cached = torch.load(root / integrity["text_cache"], map_location="cpu", weights_only=True)
        hidden, mask = cached["text_hidden_states"], cached["text_attention_mask"]
        if tuple(hidden.shape) != (128, 7680) or tuple(mask.shape) != (128,):
            raise ValueError("Wrong clothes text cache")
        self.hidden = hidden[None].to(device=device, dtype=torch.bfloat16)
        self.mask = mask[None].to(device=device, dtype=torch.bool)
        self._model = model


def build_policy(config):
    if config.get("release_kind"):
        if config["release_kind"] == "clothes_legacy":
            return LegacyClothesPolicy(config)
        if config["release_kind"] not in PublishedDeltaPolicy.ENTRIES:
            raise ValueError("Use a maintained trash, pillow or clothes_legacy release entry")
        return PublishedDeltaPolicy(config)
    return WholeBodyDeltaPolicy(config)


def build_train_command(config, source_dir):
    root = Path(source_dir).resolve()
    entry = root / "scripts/wholebody/train.py"
    if not entry.is_file():
        raise FileNotFoundError(entry)
    native = Path(config["native_config"]).resolve(strict=True)
    count = int(config.get("num_processes", 8))
    if count < 1:
        raise ValueError("num_processes must be positive")
    return dict(
        argv=[
            config.get("python", "python3"),
            "-m",
            "accelerate.commands.launch",
            "--config_file",
            str(root / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"),
            "--num_processes",
            str(count),
            str(entry),
            "--config",
            str(native),
        ],
        cwd=str(root),
        env={"PYTHONPATH": str(root / "src"), "WANDB_MODE": "disabled"},
    )


def build_serve_command(config, source_dir):
    # Explicit JSON/NPZ service. It is not the native SONIC78 actuator protocol.
    if not config.get("launcher_config"):
        raise ValueError("ImageWAM serve requires launcher_config pointing to this HumanWAM config")
    return dict(
        argv=[
            config.get("python", "python3"),
            "-m",
            "humanwam.http_server",
            "--config",
            str(config["launcher_config"]),
            "--host",
            config.get("host", "127.0.0.1"),
            "--port",
            str(int(config.get("port", 8090))),
        ],
        cwd=str(Path(source_dir).resolve()),
        env={},
    )
