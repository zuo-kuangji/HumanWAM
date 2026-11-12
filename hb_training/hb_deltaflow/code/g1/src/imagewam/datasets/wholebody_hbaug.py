"""ImageWAM Delta-flow loader with Humanoid Benchmark augmentation spec v2 (project CLAUDE.md), proprio aligned with
the GR00T A20 recipe (2026-11-12).

Installed by prepare_deltaflow.py as imagewam/datasets/wholebody_hbaug.py, next to an unchanged copy of
$G/hb_groot/hb_aug_v2.py (the implementation shared by GR00T / pi0.5 / Psi-Zero / DiT4DiT).

Base class: WholeBodyRobustLeRobotDataset (wholebody_augmented_v3), the loader of the reference run
HF humanoid-benchmark/b6-imagewam-deltaflow76 real30-from-sim60-20261108 (proprio_dim 44, velocity_channels []).
NO joint velocity (user rule 2026-11-10): the model input is exactly 44-D = body joints 29 + measured hands 12 +
projected gravity 3. Any velocity argument or any proprio width other than 44 is an error.

Visual: hb_aug_v2.augment on [T=2, V=2, 3, 224, 224]: one of the 15 effects uniformly boosted to 1.5x per sample;
colour shared over both cameras and both times; rotation/crop, light patches and single-channel white balance drawn
per camera and applied identically to the observation (t) and the future target (t+16); noise / gaussian blur / JPEG /
motion blur / sharpen / defocus only on the observation frame.
Proprio (2026-11-12, user: align every model with GR00T's A20 basket-transfer training, hb_groot/hb_launch.py):
hb_aug_v2.perturb_joints on the 29 body joints of the CURRENT frame, i.e. a fresh uniform +-2 deg bias per joint and
per sample plus N(0, 0.25 deg) noise clipped at 3 sigma; NO proprio delay, NO action-history drop (the parent class's
per-episode bias, 15 % delay and 15 % history drop are replaced). Labels (action, measured hands) are never modified.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import hb_aug_v2
from .wholebody_augmented_v3 import WholeBodyRobustLeRobotDataset

SPEC_VERSION = "hb-aug-v2-20261109"
PROPRIO_DIM = 44  # body 29 + hands 12 + gravity 3; NO velocity
BODY = 29
PROPRIO_SPEC = {"enabled": True, "joint_bias_degrees": 2.0, "joint_noise_std_degrees": 0.25,
                "delay_probability": 0.0, "max_delay_control_cycles": 0, "recording_hz": 30,
                "history_drop_probability": 0.0, "bias_scope": "per_sample",
                "implementation": "hb_aug_v2.perturb_joints"}


class WholeBodyHBAugV2LeRobotDataset(WholeBodyRobustLeRobotDataset):
    def __init__(self, *, image_augmentation, proprio_augmentation, **kwargs):
        velocity_args = sorted(k for k in kwargs if "velocity" in k)
        if velocity_args:
            raise ValueError(f"Joint velocity is not allowed in the model input (got {velocity_args})")
        image_policy = dict(image_augmentation)
        proprio_policy = dict(proprio_augmentation)
        if hb_aug_v2.VERSION != SPEC_VERSION or image_policy.get("version") != SPEC_VERSION:
            raise ValueError(f"Expected augmentation {SPEC_VERSION}, got module {hb_aug_v2.VERSION} / "
                             f"config {image_policy.get('version')}")
        if image_policy.get("enabled") is not True or "seed" not in image_policy:
            raise ValueError("Augmentation v2 must be enabled and seeded")
        for key, value in PROPRIO_SPEC.items():
            if proprio_policy.get(key) != value:
                raise ValueError(f"proprio_augmentation.{key}={proprio_policy.get(key)!r}, aligned spec needs {value!r}")
        if list(proprio_policy.get("affected_angle_channels", [])) != list(range(BODY)):
            raise ValueError("Bias/noise must hit exactly the 29 body joints")
        if list(proprio_policy.get("velocity_channels") or []):
            raise ValueError("velocity_channels must be empty (no joint velocity)")
        super().__init__(image_augmentation=image_policy, proprio_augmentation=proprio_policy, **kwargs)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        width = int(sample["proprio"].shape[-1])
        pad = sample.get("proprio_dim_is_pad")
        if width != PROPRIO_DIM or (pad is not None and int(pad.shape[-1]) != PROPRIO_DIM):
            raise ValueError(f"proprio must be {PROPRIO_DIM}-D (no velocity), got {width}")
        return sample

    def augment_numeric(self, sample, rng):
        episode = int(sample["episode_index"])
        anchor = int(sample["frame_index"])
        states, _ = self.episodes[episode]
        state = states[anchor].copy()                                  # no delay: always the current frame
        raw = state[:BODY].astype(np.float64)
        state[:BODY] = hb_aug_v2.perturb_joints(state[:BODY], rng)     # per-sample bias +-2 deg + noise 0.25 deg
        current = self.normalizer.forward(state, "state")
        sample["proprio"] = torch.from_numpy(np.repeat(current[None], self.action_horizon, axis=0)).to(
            dtype=sample["proprio"].dtype)
        past = sample["past_action"].clone()
        pad = sample["past_action_is_pad"].clone()
        past[pad] = 0                                                  # unchanged history, no drop
        sample["past_action"] = past
        sample["past_action_is_pad"] = pad
        return dict(delay_recorded_frames=0, history_drop_rows=[],
                    joint_perturbation_degrees=(np.degrees(state[:BODY].astype(np.float64) - raw)).tolist())

    def augment_video(self, video, rng):
        # video: [3, T, 224, 2*224] in [-1, 1]; T=2 = (anchor t, future target t+16); width = ego_view | chest.
        if (video.ndim != 4 or video.shape[0] != 3 or video.shape[1] != 2
                or video.shape[-1] != 2 * video.shape[-2]):
            raise ValueError(f"Expected [3,2,H,2H] ego|chest video, got {tuple(video.shape)}")
        x = ((video.permute(1, 0, 2, 3) + 1.0) * 0.5).clamp(0, 1)
        views = torch.stack(x.chunk(2, dim=-1), dim=1)              # [T, V=2, 3, H, W]
        observation = torch.ones(views.shape[0], dtype=torch.bool)
        observation[-1] = False                                       # t+16 is the supervised image target
        out, params = hb_aug_v2.augment(views, rng, obs_mask=observation)
        joined = torch.cat(list(out.unbind(1)), dim=-1)              # [T, 3, H, 2W]
        return (joined * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous().to(video.dtype), params


assert math.isclose(PROPRIO_SPEC["joint_bias_degrees"], 2.0)
