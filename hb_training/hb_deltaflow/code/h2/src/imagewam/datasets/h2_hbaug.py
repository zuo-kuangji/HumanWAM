"""A18 (H2 + Sharpa hands) Delta-flow loader with augmentation spec v2. NO joint velocity.

Installed by prepare_deltaflow_a18.py as imagewam/datasets/h2_hbaug.py in the H2 code copy (HumanWAM-h2), next to
an unchanged copy of $G/hb_groot/hb_aug_v2.py. Builds on H2PourDataset of a18-h2-pour-from-scratch-20261108:
same cameras (head + chest; the collection also records wrist and third_person, unused as before), same
normalization file format (q01/q99 of state78 = 75 joints + gravity3, action75 = body_target31 + hands44),
same 16-step action chunk / history and t+16 image target. Changes:
  - visual: hb_aug_v2.augment on [T=2, V=2, 3, 224, 224] (15 effects, one boosted to 1.5x per sample, colour shared,
    rotation/light/WB per camera on t and t+16, sensor/optics only on the observation frame t);
  - proprio (2026-11-12, aligned with GR00T A20; H2 body = state[0:31] incl. waist and head):
    hb_aug_v2.perturb_joints per sample (+-2 deg bias + N(0, 0.25 deg) noise clipped at 3 sigma), no delay,
    no history drop; hands, gravity and every label untouched.
Model input: proprio[16, 78] = normalized(75 joints + gravity 3), exactly the state of the original H2 model
(h2_pour_delta.py, HF g1-basket-transfer-imagewam-deltaflow76 h2-a18-pour-water/epoch-040: proprio_dim 78).
User rule 2026-11-10: no joint-velocity channels in any training; any velocity argument is an error.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from . import hb_aug_v2
from .h2_pour import ACTION_DIM, CAMERAS, STATE_DIM, H2PourDataset
from .wholebody import window_indices

BODY = 31
SPEC_VERSION = "hb-aug-v2-20261109"
# 2026-11-12 (user): proprio aligned with GR00T's A20 basket-transfer training (hb_groot/hb_launch.py):
# hb_aug_v2.perturb_joints on the 31 body joints per sample, no delay, no history drop.
PROPRIO_SPEC = {"enabled": True, "joint_bias_degrees": 2.0, "joint_noise_std_degrees": 0.25,
                "delay_probability": 0.0, "max_delay_control_cycles": 0, "control_hz": 30, "recording_hz": 30,
                "history_drop_probability": 0.0, "bias_scope": "per_sample",
                "implementation": "hb_aug_v2.perturb_joints"}
if STATE_DIM != 78:
    raise ImportError(f"H2 state must be 78-D (75 joints + gravity 3), h2_pour says {STATE_DIM}")


class H2HBAugV2Dataset(H2PourDataset):
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
        if list(proprio_policy.get("affected_angle_channels", [])) != list(range(BODY)) or "seed" not in proprio_policy:
            raise ValueError("Bias/noise must hit exactly the 31 H2 body joints (state[0:31])")
        if list(proprio_policy.get("velocity_channels") or []):
            raise ValueError("velocity_channels must be empty (no joint velocity)")
        # The parent's own (spec v1) image augmentation stays disabled; hb_aug_v2 replaces it below.
        super().__init__(image_augmentation=None, **kwargs)
        self.image_policy, self.proprio_policy = image_policy, proprio_policy
        self._aug_rng, self._aug_key = None, None
        self.last_augmentation = {}

    def _rng(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        rank = int(os.environ.get("RANK", "0"))
        initial = int(torch.initial_seed())
        key = (os.getpid(), rank, worker_id, initial)
        if key != self._aug_key:
            self._aug_rng = np.random.default_rng(np.random.SeedSequence(
                [int(self.image_policy["seed"]), rank, worker_id, initial % (2 ** 32), initial // (2 ** 32)]))
            self._aug_key = key
        return self._aug_rng

    def augment_numeric(self, episode, anchor, state, past, pad, rng):
        raw = state[anchor].copy()                                  # 75 joints + gravity 3, radians; no delay
        before = raw[:BODY].astype(np.float64)
        raw[:BODY] = hb_aug_v2.perturb_joints(raw[:BODY], rng)      # per-sample bias +-2 deg + noise 0.25 deg
        proprio = self.normalize(raw, "state").astype(np.float32)
        if proprio.shape != (STATE_DIM,):
            raise ValueError(f"proprio must be {STATE_DIM}-D (no velocity), got {proprio.shape}")
        past[pad] = 0                                               # unchanged history, no drop
        info = dict(delay_recorded_frames=0, state_frame=anchor, anchor_frame=anchor, history_drop_rows=[],
                    joint_perturbation_degrees=np.degrees(raw[:BODY].astype(np.float64) - before).tolist())
        return proprio, past, pad, info

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_position = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episode_ids[ep_position]
        anchor = int(index - (self.ends[ep_position - 1] if ep_position else 0))
        state, action = self.episodes[episode]
        window = window_indices(anchor, len(action), 16, 16)
        views = torch.stack([self._frames(episode, c, window["image"].tolist()) for c in CAMERAS], dim=1)
        rng = self._rng()
        observation = torch.tensor([True, False])                   # t+16 is the supervised image target
        views, visual = hb_aug_v2.augment(views, rng, obs_mask=observation)
        video = (torch.cat(list(views.unbind(1)), dim=-1) * 2 - 1).permute(1, 0, 2, 3).contiguous()
        future = self.normalize(action[window["action"]], "action")
        past = self.normalize(action[window["past_action"]], "action")
        if window["past_action_is_pad"].all():
            past = np.zeros_like(past)
        proprio, past, pad, numeric = self.augment_numeric(
            episode, anchor, state, torch.from_numpy(past), torch.from_numpy(window["past_action_is_pad"].copy()), rng)
        self.last_augmentation = dict(visual=visual, numeric=numeric)
        return dict(video=video, action=torch.from_numpy(future), past_action=past,
                    proprio=torch.from_numpy(np.repeat(proprio[None], 16, axis=0)),
                    action_is_pad=torch.from_numpy(window["action_is_pad"]), past_action_is_pad=pad,
                    image_is_pad=torch.from_numpy(window["image_is_pad"]),
                    proprio_is_pad=torch.zeros(16, dtype=torch.bool),
                    action_dim_is_pad=torch.zeros(ACTION_DIM, dtype=torch.bool),
                    proprio_dim_is_pad=torch.zeros(STATE_DIM, dtype=torch.bool),
                    text_hidden_states=self.text_hidden_states, text_attention_mask=self.text_attention_mask,
                    prompt=self.prompt, instruction=self.prompt, idx=int(index), episode_index=int(episode),
                    frame_index=anchor, task_index=0, dataset_index=0)
