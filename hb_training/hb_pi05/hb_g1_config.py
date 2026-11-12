"""OpenPI TrainConfigs for the HB tasks. Importing this module registers in openpi.training.config._CONFIGS_DICT:
  pi05_hb_g1_b6, pi05_hb_g1_basket, pi05_hb_g1_a3, pi05_hb_g1_a7   (G1: state 44, actions 76)
  pi05_hb_h2_a18                                                    (H2: state 78, actions 75 + 1 zero pad)
so get_config()/cli() and policy_config.create_trained_policy() can use them.

Model (all tasks): Pi0Config(pi05=True, action_dim=76, action_horizon=16) and one shared 76-D init checkpoint.
For pi05, ModelTransformFactory (training/config.py) runs TokenizePrompt(discrete_state_input=True) BEFORE
PadStatesAndActions(76): the normalized state is discretized into 256 bins and written into the prompt
("Task: ..., State: s1 ... sS;\\nAction: "), then state/actions are zero-padded up to 76 (pad_to_dim never truncates, so
the H2 state stays 78 wide; the pi05 network only uses observation.state for its batch shape). H2 actions (75) get one
zero dim, sliced off again by HBOutputs. Each state value takes ~4 tokens, so max_token_len is per robot layout
(g1 256, h2 384; default pi05 is 200); verify_cpu.py checks nothing is truncated.
"""
import dataclasses
import os
import pathlib

import numpy as np
from typing_extensions import override

import openpi.models.pi0_config as pi0_config
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

import hb_g1_data as D


def _uint8_hwc(img) -> np.ndarray:  # same as libero_policy._parse_image
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.shape[0] == 3:
        img = np.moveaxis(img, 0, -1)
    return img


@dataclasses.dataclass(frozen=True)
class HBInputs(_transforms.DataTransformFn):
    """Training and inference input mapping (cf. policies/libero_policy.py:LiberoInputs). Inference clients send
    observation/<cam> (raw 480x640 RGB uint8; ResizeImages(224, 224) does the identical resize_with_pad),
    observation/state and prompt."""

    layout: str = "g1"

    def __call__(self, data: dict) -> dict:
        L = D.LAYOUTS[self.layout]
        base = _uint8_hwc(data[f"observation/{L.cameras[0]}"])
        wrist = _uint8_hwc(data[f"observation/{L.cameras[1]}"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if state.shape[-1] != L.state_dim:
            raise ValueError(f"{self.layout}: state dim {state.shape} != {L.state_dim}")
        out = {
            "state": state,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {"base_0_rgb": np.True_, "left_wrist_0_rgb": np.True_, "right_wrist_0_rgb": np.False_},
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != L.action_dim:
                raise ValueError(f"{self.layout}: action dim {actions.shape} != {L.action_dim}")
            out["actions"] = actions
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class HBOutputs(_transforms.DataTransformFn):
    """Slices the 76-D model output back to the robot's raw action dim (h2: drops the zero pad dim). Runs after
    Unnormalize, whose quantile branch leaves dims beyond the 75-D stats untouched."""

    action_dim: int = D.MODEL_ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.action_dim])}


@dataclasses.dataclass(frozen=True)
class HBDataConfig(_config.DataConfigFactory):
    """repo_id 'hb_<layout>_<task>' is resolved to HBCachedDataset by hb_g1_data.install_dataset_hook (training only).
    The dataset already yields the HBInputs keys and the prompt, so no repack transform / PromptFromLeRobotTask."""

    repo_id: str = "hb_g1_b6"
    layout: str = "g1"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config) -> _config.DataConfig:
        L = D.LAYOUTS[self.layout]
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),  # loads assets/<asset_id>/norm_stats.json, quantile norm for pi05
            data_transforms=_transforms.Group(inputs=[HBInputs(self.layout)], outputs=[HBOutputs(L.action_dim)]),
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


# 2026-11-12: A6 door-pull has a ~90-token instruction; prompt + 44 discretized state values reach 257 tokens, so its
# sequence capacity is raised (no parameter changes, only padding length; every other task keeps its layout value).
TASK_MAX_TOKEN_LEN = {"a6": 320}


def model_config(layout: str = "g1", max_token_len: int | None = None, **kw) -> pi0_config.Pi0Config:
    return pi0_config.Pi0Config(pi05=True, action_dim=D.MODEL_ACTION_DIM, action_horizon=D.HORIZON,
                                max_token_len=max_token_len or D.LAYOUTS[layout].max_token_len, **kw)


def make_train_config(task: str, *, exp_name: str = "default", batch_size: int = 32 * 32, num_train_steps: int = 1000,
                      peak_lr: float | None = None, warmup_frac: float | None = None, num_workers: int = 8,
                      checkpoint_base_dir: str | None = None, pytorch_weight_path: str | None = "default",
                      resume: bool = False, log_interval: int = 10) -> _config.TrainConfig:
    L = D.layout_of(task)
    peak_lr = float(os.environ.get("HB_PEAK_LR", "5e-5")) if peak_lr is None else peak_lr
    warmup_frac = float(os.environ.get("HB_WARMUP_FRAC", "0.05")) if warmup_frac is None else warmup_frac
    if pytorch_weight_path == "default":
        pytorch_weight_path = os.environ.get("HB_PI05_INIT", str(D.G / "models/openpi/pi05_base_g1a76_pytorch"))
    runs = os.environ.get("HB_PI05_RUNS", str(D.G / "pi05-runs"))
    return _config.TrainConfig(
        name=D.config_name(task),
        exp_name=exp_name,
        model=model_config(L.name, max_token_len=TASK_MAX_TOKEN_LEN.get(task)),
        data=HBDataConfig(repo_id=D.repo_id(task), layout=L.name,
                          assets=_config.AssetsConfig(assets_dir=str(D.task_dir(task) / "assets"), asset_id=D.asset_id(task))),
        pytorch_weight_path=pytorch_weight_path,
        pytorch_training_precision="bfloat16",
        # train_pytorch.py: linear warmup then cosine to decay_lr at decay_steps (= end of training here).
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=max(1, round(warmup_frac * num_train_steps)),
                                                   peak_lr=peak_lr, decay_steps=num_train_steps, decay_lr=peak_lr / 10),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,  # EMA is not supported by the PyTorch trainer
        batch_size=batch_size,  # GLOBAL batch; train_pytorch/data_loader divide by world size
        num_workers=num_workers,
        num_train_steps=num_train_steps,
        log_interval=log_interval,
        save_interval=10**9,  # hb_train.py decides the save steps
        keep_period=None,
        wandb_enabled=False,
        assets_base_dir=str(pathlib.Path(runs) / "assets"),
        checkpoint_base_dir=checkpoint_base_dir or str(pathlib.Path(runs) / "checkpoints"),
        resume=resume,
        policy_metadata={"robot": L.name, "task": task, "state": L.state_desc, "actions": L.action_desc,
                         "action_dim": L.action_dim, "action_horizon": D.HORIZON},
    )


for _task in D.TASKS:
    _cfg = make_train_config(_task)
    if _cfg.name not in _config._CONFIGS_DICT:  # noqa: SLF001
        _config._CONFIGS.append(_cfg)  # noqa: SLF001
        _config._CONFIGS_DICT[_cfg.name] = _cfg  # noqa: SLF001
