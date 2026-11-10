"""A18 (H2 + Sharpa hands) Delta-flow loader with augmentation spec v2 and 31-joint velocity.

Installed by prepare_deltaflow_a18.py as imagewam/datasets/h2_hbaug.py in the H2 code copy (HumanWAM-h2), next to
an unchanged copy of $G/hb_groot/hb_aug_v2.py. Builds on H2PourDataset of a18-h2-pour-from-scratch-20261108:
same cameras (head + chest; the collection also records wrist and third_person, unused as before), same
normalization file format (q01/q99 of state78 = 75 joints + gravity3, action75 = body_target31 + hands44),
same 16-step action chunk / history and t+16 image target. Changes:
  - visual: hb_aug_v2.augment on [T=2, V=2, 3, 224, 224] (15 effects, one boosted to 1.5x per sample, colour shared,
    rotation/light/WB per camera on t and t+16, sensor/optics only on the observation frame t);
  - proprio (spec section 3, H2 body = state[0:31] incl. waist and head): per-episode fixed +-2 deg bias, N(0, 0.25 deg)
    noise clipped at 3 sigma, 15 % delay of 1-2 control cycles (30 Hz -> 1-2 frames) applied to the whole
    observation and the action history, 15 % history drop; hands, gravity and every label untouched;
  - velocity (spec section 4): 31 body joints, (q[t]-q[t-1])/(ts[t]-ts[t-1]) from a prepared cache, taken at the
    delayed frame, N(0, 0.05 rad/s) noise clipped at 3 sigma, (v-mean)/std (std floor 0.05) clipped to [-5, 5],
    zeros at episode frame 0.
Model input: proprio[16, 109] = normalized(75 joints + gravity 3) followed by normalized velocity 31.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from . import hb_aug_v2
from .h2_pour import ACTION_DIM, CAMERAS, STATE_DIM as RAW_STATE_DIM, H2PourDataset
from .wholebody import window_indices

BODY = 31
STATE_DIM = RAW_STATE_DIM + BODY          # 78 + 31 = 109
SPEC_VERSION = "hb-aug-v2-20261109"
PROPRIO_SPEC = {"enabled": True, "joint_bias_degrees": 2.0, "joint_noise_std_degrees": 0.25,
                "delay_probability": 0.15, "max_delay_control_cycles": 2, "control_hz": 30, "recording_hz": 30,
                "history_drop_probability": 0.15, "bias_scope": "fixed_per_episode"}


class H2HBAugV2Dataset(H2PourDataset):
    def __init__(self, *, image_augmentation, proprio_augmentation, velocity_cache, velocity_noise_std=0.05, **kwargs):
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
        if float(velocity_noise_std) != 0.05:
            raise ValueError("Velocity noise must be 0.05 rad/s")
        # The parent's own (spec v1) image augmentation stays disabled; hb_aug_v2 replaces it below.
        super().__init__(image_augmentation=None, **kwargs)
        self.image_policy, self.proprio_policy = image_policy, proprio_policy
        self.velocity_noise_std = float(velocity_noise_std)
        with np.load(velocity_cache) as cache:
            self.velocity_mean = cache["mean"].astype(np.float32)
            self.velocity_scale = cache["scale"].astype(np.float32)
            self.velocities = {int(key.split("_")[1]): cache[key].astype(np.float32)
                               for key in cache.files if key.startswith("episode_")}
        if self.velocity_mean.shape != (BODY,) or self.velocity_scale.shape != (BODY,) or (self.velocity_scale < 0.05).any():
            raise ValueError("Invalid velocity normalization")
        if sorted(self.velocities) != sorted(self.episodes):
            raise ValueError("Velocity cache episodes differ from the collection")
        for episode, (state, _) in self.episodes.items():
            if self.velocities[episode].shape != (len(state), BODY) or self.velocities[episode][0].any():
                raise ValueError(f"Invalid velocity cache for episode {episode}")
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
        policy = self.proprio_policy
        delay = 0
        if anchor > 0 and rng.random() < policy["delay_probability"]:
            cycles = int(rng.integers(1, policy["max_delay_control_cycles"] + 1))
            delay = max(1, round(cycles * policy["recording_hz"] / policy["control_hz"]))
        frame = max(0, anchor - delay)
        raw = state[frame].copy()                                   # 75 joints + gravity 3, radians
        bias_rng = np.random.default_rng(np.random.SeedSequence([int(policy["seed"]), int(episode), 3719]))
        limit = math.radians(policy["joint_bias_degrees"])
        bias = bias_rng.uniform(-limit, limit, BODY)
        sigma = math.radians(policy["joint_noise_std_degrees"])
        noise = np.clip(rng.normal(0, sigma, BODY), -3 * sigma, 3 * sigma)
        raw[:BODY] = (raw[:BODY] + bias + noise).astype(np.float32)
        current = self.normalize(raw, "state")
        if frame > 0:
            s = self.velocity_noise_std
            jitter = np.clip(rng.normal(0, s, BODY), -3 * s, 3 * s)
            velocity = np.clip((self.velocities[episode][frame] + jitter - self.velocity_mean) / self.velocity_scale,
                               -5, 5).astype(np.float32)
        else:
            velocity = np.zeros(BODY, dtype=np.float32)
        if delay:
            shifted, shift_pad = torch.zeros_like(past), torch.ones_like(pad)
            if delay < len(past):
                shifted[delay:], shift_pad[delay:] = past[:-delay], pad[:-delay]
            past, pad = shifted, shift_pad
        dropped = []
        if rng.random() < policy["history_drop_probability"]:
            candidates = [i for i in range(1, len(past)) if not bool(pad[i - 1]) and not bool(pad[i])]
            if candidates:
                i = int(rng.choice(candidates))
                past[i], pad[i] = past[i - 1], pad[i - 1]
                dropped.append(i)
        past[pad] = 0
        proprio = np.concatenate([current, velocity]).astype(np.float32)
        info = dict(delay_recorded_frames=delay, state_frame=frame, bias_degrees=np.degrees(bias).tolist(),
                    noise_degrees=np.degrees(noise).tolist(), history_drop_rows=dropped)
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
