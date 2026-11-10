"""ImageWAM Delta-flow loader with Humanoid Benchmark augmentation spec v2 (project CLAUDE.md).

Installed by prepare_deltaflow.py as imagewam/datasets/wholebody_hbaug.py, next to an unchanged copy of
$G/hb_groot/hb_aug_v2.py (the implementation shared by GR00T / pi0.5 / Psi-Zero / DiT4DiT).

Base class: WholeBodyRobustLeRobotDataset (wholebody_augmented_v3), the loader of the reference run
HF humanoid-benchmark/b6-imagewam-deltaflow76 real30-from-sim60-20261108 (proprio_dim 44, velocity_channels []).
NO joint velocity (user rule 2026-11-10): the model input is exactly 44-D = body joints 29 + measured hands 12 +
projected gravity 3. Any velocity argument or any proprio width other than 44 is an error.

Only the visual part changes relative to the reference run:
  reference  WholeBodyRobustLeRobotDataset.augment_video ("b6-visual-aug-v2" = spec v1, 10 effects, no 1.5x boost)
  now        hb_aug_v2.augment on [T=2, V=2, 3, 224, 224]: one of the 15 effects uniformly boosted to 1.5x per sample;
             colour shared over both cameras and both times; rotation/crop, light patches and single-channel white
             balance drawn per camera and applied identically to the observation (t) and the future target (t+16);
             noise / gaussian blur / JPEG / motion blur / sharpen / defocus only on the observation frame.
Proprio (29-joint bias + noise, 1-2 control-cycle delay, history drop) is inherited unchanged from
WholeBodyRobustLeRobotDataset; this class only checks its parameters. Labels (action, measured hands) are never modified.
"""
from __future__ import annotations

import torch

from . import hb_aug_v2
from .wholebody_augmented_v3 import WholeBodyRobustLeRobotDataset

SPEC_VERSION = "hb-aug-v2-20261109"
PROPRIO_DIM = 44  # body 29 + hands 12 + gravity 3; NO velocity
PROPRIO_SPEC = {"enabled": True, "joint_bias_degrees": 2.0, "joint_noise_std_degrees": 0.25,
                "delay_probability": 0.15, "max_delay_control_cycles": 2, "recording_hz": 30,
                "history_drop_probability": 0.15, "bias_scope": "fixed_per_episode"}


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
                raise ValueError(f"proprio_augmentation.{key}={proprio_policy.get(key)!r}, spec v2 needs {value!r}")
        if int(proprio_policy.get("control_hz", 0)) not in (30, 50):
            raise ValueError("control_hz must be 50 (B6 SONIC) or 30 (A3/A7)")
        if list(proprio_policy.get("affected_angle_channels", [])) != list(range(29)):
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
