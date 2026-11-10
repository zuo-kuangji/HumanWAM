"""Psi0 launch config for the Humanoid Benchmark tasks: G1 (A76 / S44 / H30) and H2 a18 (A75 / S78 / H30).

Registered by hb_train.py as `psi.config.train.hb_psi0_config`, so the pinned upstream entry
`scripts/train.py` (which does importlib.import_module(f"psi.config.train.{sys.argv[1]}")) loads it
unchanged. Structure mirrors src/psi/config/train/finetune_sonic_psi0_config.py; differences:

* repack  = HbSonicRepackTransform (subclass of transform_psi0_sonic.SonicRepackTransform)
    - native state-temporal-jitter must be 0; replaced by the HB v2 proprio spec (CLAUDE.md §三):
      15 % delay of 1..hb_delay_max_frames frames (window clamped to the episode by LeRobot), then on
      the first hb_body_dims raw-radian dims (G1: 29 body joints; H2: 31 = 29 + head 2), before
      normalization:
        bias  U(+-2 deg) per joint, FIXED PER TRAJECTORY within an epoch: drawn from
              default_rng([hb_bias_seed, epoch, episode_index]). episode_index is a column of every
              LeRobot item (lerobot_dataset.py __getitem__: item = hf_dataset[idx]); the epoch is the
              Psi0 loop epoch, pushed by hb_patches (Trainer.next_epoch) into a fork-shared RawValue
              that the persistent DataLoader workers read. Same (epoch, episode) -> same bias on every
              rank/worker; a new epoch draws new biases.
        noise N(0, 0.25 deg) clipped at 3 sigma, per sample (hb_rng).
      Same constants as hb_aug_v2.perturb_joints, which cannot be reused here because it draws the bias
      inside each call. Skipped when no_aug=True (val / deploy).
    - asserts that the repacked widths equal pad_*_dim, i.e. NO zero padding ever happens.
* field   = native SonicActionStateTransform (bounds normalization), state_noise_std must be 0.
* model   = HbPsi0ModelTransform (subclass of transform.Psi0ModelTransform)
    - native img_aug / view_aug must be off; HB v2 visual augmentation (hb_aug_v2.augment) is applied
      jointly to all cameras of the sample at the model input resolution (after the native
      resize + center crop), then the native Qwen3-VL input builder runs unchanged.
"""
# No `from __future__ import annotations` (same as the upstream config modules): tyro/pydantic then see
# real classes, not strings that must be resolved through the aliased module name.
import json
import math
import multiprocessing
import os
import sys
from typing import Any

import numpy as np
import torch
from pydantic import model_validator
from torchvision.transforms import v2
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from typing_extensions import Self

from psi.config import transform as pt
from psi.config import transform_psi0_sonic as ps
from psi.config.config import LaunchConfig
from psi.config.data_lerobot import LerobotDataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform

_HERE = os.path.dirname(os.path.abspath(__file__))
_AUG_DIR = os.environ.get("HB_AUG_DIR", os.path.join(os.path.dirname(_HERE), "hb_groot"))
if _AUG_DIR not in sys.path:
    sys.path.insert(0, _AUG_DIR)
import hb_aug_v2 as A  # noqa: E402  (augment(x[T,V,3,H,W] in 0..1, rng), perturb_joints(q, rng))

HB_AUG_VERSION = "hb-aug-v2-20261109"
assert A.VERSION == HB_AUG_VERSION, f"hb_aug_v2 at {_AUG_DIR} is {A.VERSION}, expected {HB_AUG_VERSION}"

_RNG: dict[str, Any] = {"pid": None, "rng": None, "count": 0}
# Current training epoch, shared with forked DataLoader workers (created in the main process at import,
# i.e. before any worker is forked; torch DataLoader uses fork on Linux).
_EPOCH = multiprocessing.RawValue("q", 0)
BIAS_DEG, NOISE_DEG = 2.0, 0.25  # = hb_aug_v2.perturb_joints constants


def set_epoch(epoch: int) -> None:
    _EPOCH.value = int(epoch)


def get_epoch() -> int:
    return int(_EPOCH.value)


def traj_bias(seed: int, epoch: int, episode_index: int, dims: int) -> np.ndarray:
    """Per-trajectory joint bias (rad), deterministic in (seed, epoch, episode_index)."""
    r = np.random.default_rng([int(seed), int(epoch), int(episode_index)])
    return r.uniform(-math.radians(BIAS_DEG), math.radians(BIAS_DEG), size=dims)


def hb_rng() -> np.random.Generator:
    """Per-process generator (re-seeded after fork from pid + urandom). Psi0's worker_init_fn seeds
    np.random from a generator that is identical on every rank, so we do not reuse np.random."""
    pid = os.getpid()
    if _RNG["pid"] != pid:
        _RNG["rng"] = np.random.default_rng([pid, int.from_bytes(os.urandom(8), "little")])
        _RNG["pid"], _RNG["count"] = pid, 0
    return _RNG["rng"]


def _log_aug(p: dict, shape) -> None:
    _RNG["count"] += 1
    log = os.environ.get("HB_AUG_LOG")
    if log and _RNG["count"] in (1, 100):
        with open(f"{log}.{os.getpid()}", "a") as f:
            f.write(json.dumps({"count": _RNG["count"], "shape": list(shape), "boost": p["boost"],
                                "version": A.VERSION}) + "\n")


class HbSonicRepackTransform(ps.SonicRepackTransform):
    hb_delay_prob: float = 0.15       # HB v2: delay proprio by 1..2 control periods with p=0.15
    hb_delay_max_frames: int = 2      # 30 fps recordings -> 1..2 frames
    hb_body_dims: int = 29            # leading joint-angle dims of the state (G1 29, H2 31 incl. head)
    hb_joint_perturb: bool = True     # HB v2: +-2 deg per-trajectory bias + N(0, 0.25 deg) clipped 3 sigma
    hb_bias_seed: int = 292285        # bias = traj_bias(hb_bias_seed, epoch, episode_index)

    def delta_timestamps(self, fps):
        assert self.state_temporal_jitter == 0 and self.num_past_frames == 0, \
            "native state-temporal-jitter / past frames must be 0 (replaced by the HB v2 proprio spec)"
        delta = super().delta_timestamps(fps)
        for s_key in self.state_keys:
            base, _ = ps.parse_modality_key(s_key)
            delta[base] = [t / fps for t in range(-self.hb_delay_max_frames, 1)]  # [-D .. 0]
        return delta

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        no_aug = kwargs.get("no_aug", False)
        D = self.hb_delay_max_frames
        k = 0
        if not no_aug and D > 0 and self.hb_delay_prob > 0 and hb_rng().random() < self.hb_delay_prob:
            k = int(hb_rng().integers(1, D + 1))
        data = dict(data)
        for s_key in self.state_keys:
            base, _ = ps.parse_modality_key(s_key)
            w = np.asarray(data[base])
            if w.ndim != 2 or w.shape[0] != D + 1:
                raise ValueError(f"{base}: expected a ({D + 1}, Ds) state window, got {w.shape}")
            data[base] = w[D - k:D - k + 1]  # s_{t-k}; LeRobot already clamped it to the episode
        out = super().__call__(data, **kwargs)
        if out["states"].shape[-1] != self.pad_state_dim or out["actions"].shape[-1] != self.pad_action_dim \
                or not bool(np.all(out["actions_mask"])):
            raise ValueError(f"repacked state {out['states'].shape} / action {out['actions'].shape} must equal "
                             f"pad dims {self.pad_state_dim}/{self.pad_action_dim} exactly (no zero padding)")
        nb = self.hb_body_dims
        if not no_aug and self.hb_joint_perturb and nb > 0:
            if "episode_index" not in data:
                raise KeyError("episode_index missing from the LeRobot item (needed for the per-trajectory bias)")
            ep = int(np.asarray(data["episode_index"]).reshape(-1)[0])
            s = out["states"]
            sigma = math.radians(NOISE_DEG)
            noise = np.clip(hb_rng().normal(0.0, sigma, size=s[..., :nb].shape), -3 * sigma, 3 * sigma)
            s[..., :nb] = (s[..., :nb] + traj_bias(self.hb_bias_seed, get_epoch(), ep, nb) + noise).astype(np.float32)
        return out


class HbPsi0ModelTransform(pt.Psi0ModelTransform):
    hb_aug: bool = True

    def __call__(self, data: dict[str, Any], vlm_processor=None, no_aug=False, **kwargs) -> dict[str, Any]:
        assert not self.img_aug and not self.view_aug and not self.adaptive_resize, \
            "Psi0 native image augmentation must stay off (HB v2 replaces it)"
        if self.hb_aug and not no_aug:
            to_input = v2.Compose([self.resize(), self.center_crop()])  # native geometry, model input res
            imgs = [to_input(im) for im in data["observations"]]
            x = torch.stack([pil_to_tensor(im) for im in imgs]).float().div(255.0)[None]  # [1, V, 3, H, W]
            x, p = A.augment(x, hb_rng())
            u8 = (x[0] * 255.0 + 0.5).clamp(0, 255).to(torch.uint8)
            data = dict(data)
            data["observations"] = [to_pil_image(u8[i]) for i in range(u8.shape[0])]
            _log_aug(p, x.shape)
        # Native path; images are already at resize/center_crop size, so its NEAREST resize and
        # center crop are identities, and no_aug=True keeps its (disabled) augmentation off.
        return super().__call__(data, vlm_processor=vlm_processor, no_aug=True, **kwargs)


class HbDataTransform(DataTransform):
    repack: HbSonicRepackTransform
    field: ps.SonicActionStateTransform
    model: HbPsi0ModelTransform


class HbDataConfig(LerobotDataConfig):
    # MUST re-annotate `transform` (as finetune_sonic_psi0_config.DynamicDataConfig does): tyro builds the
    # CLI from the annotation, and the inherited one is the base DataTransform (repack: RepackTransform,
    # field: FieldTransform, model: ModelTransform), which has none of the repack/field/model flags.
    transform: HbDataTransform
    # read via getattr() in trainers/finetune.py:create_dataloaders (default 12 workers / prefetch 4)
    num_workers: int | None = None
    prefetch_factor: int | None = None


class DynamicLaunchConfig(LaunchConfig):
    data: HbDataConfig
    model: Psi0ModelConfig

    @model_validator(mode="after")
    def check_hb_dims(self, __context: Any) -> Self:  # same signature as the native validator
        repack, field, model = self.data.transform.repack, self.data.transform.field, self.model
        # same checks as finetune_sonic_psi0_config.DynamicLaunchConfig.check_observation_dim
        assert repack.pad_action_dim == field.pad_action_dim == model.action_dim, \
            (repack.pad_action_dim, field.pad_action_dim, model.action_dim)
        assert repack.pad_state_dim == field.pad_state_dim == model.odim, \
            (repack.pad_state_dim, field.pad_state_dim, model.odim)
        assert model.action_chunk_size == repack.action_chunk_size
        if model.vlm_layer_indices is not None:
            assert model.num_blocks == len(model.vlm_layer_indices)
        # HB: statistics must have exactly the model widths (pad_to_len would otherwise zero-pad)
        assert field.action_min is not None and len(field.action_min) == model.action_dim, \
            f"action stats missing or wrong width ({field.stat_path})"
        assert field.normalize_state and field.state_min is not None and len(field.state_min) == model.odim, \
            f"state stats missing or wrong width ({field.stat_path})"
        # HB v2: native augmentations off, ours on
        m = self.data.transform.model
        assert not m.img_aug and not m.view_aug, "disable --data.transform.model.img-aug / view-aug"
        assert field.state_noise_std == 0.0 and repack.state_temporal_jitter == 0
        assert 0 <= repack.hb_body_dims <= model.odim, (repack.hb_body_dims, model.odim)
        return self
