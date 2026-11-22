"""2026-11-16: Humanoid Benchmark augmentation v2 (project CLAUDE.md) inside the native FastWAM data pipeline.
- Visual: hb_aug_v2.augment on every training sample's frames before the two cameras are concatenated; frame 0 is the
  current observation (sensor / optics effects only there), the later frames are the future video targets (colour,
  per-camera geometry, light patches and white balance shared with frame 0, as the spec requires).
- Proprio: hb_aug_v2.perturb_joints on the raw body joints (G1 29, H2 31 = body 29 + head 2) before normalization;
  hands and gravity untouched; supervision labels untouched. Normalization statistics are computed on the raw data.
FastWAM's own image augmentation is only a resize (no colour jitter / crop), so nothing native is stacked.
Installed in the main process before the dataloaders fork; HB_FW_JOINTS sets the number of perturbed joints.
HB_AUG_PROFILE=unified (or unified_v2) explicitly selects the shared v2 recipe.
unified_v3 remains an opt-in recipe; unified_v1 preserves the earliest recipe. Its identity is the real
episode/frame plus the sampler's resumed epoch, carried with each index through worker prefetch.
The legacy default remains hb_v2."""
import importlib
import os
import sys
import time
from contextvars import ContextVar

import numpy as np
import torch

NJ = int(os.environ.get("HB_FW_JOINTS", "29"))
_seen = {"img": False, "state": False}
_epoch = ContextVar("hb_fastwam_augmentation_epoch", default=0)
_SAMPLE = "__hb_augmentation_sample"


def _rng(idx=0):
    return np.random.default_rng([os.getpid(), int(idx) & 0xFFFFFFFF, time.time_ns() & 0xFFFFFFFF])


def _train(proc):
    v = getattr(proc, "is_train", False)
    return bool(v() if callable(v) else v)


def _identity(data, SampleKey):
    values = []
    for name in ("episode_index", "frame_index"):
        if name not in data:
            raise ValueError(f"Unified FastWAM augmentation needs the actual dataset {name}")
        array = np.asarray(data[name]).reshape(-1)
        if not len(array) or not np.issubdtype(array.dtype, np.integer) or np.any(array < 0):
            raise ValueError(f"Dataset {name} must contain nonnegative integer IDs")
        values.append(int(array[0]))
    return SampleKey(*values, epoch=_epoch.get())


def install(profile=None, seed=None):
    profile = profile or os.environ.get("HB_AUG_PROFILE", "hb_v2")
    if profile not in {"hb_v2", "unified", "unified_v1", "unified_v2", "unified_v3"}:
        raise ValueError("HB_AUG_PROFILE must be hb_v2, unified, unified_v1, unified_v2 or unified_v3")
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor

    installed = getattr(FastWAMProcessor, "_hb_aug_profile", None)
    if installed is not None:
        if installed != profile:
            raise ValueError("Start a fresh process when changing FastWAM augmentation profiles")
        return FastWAMProcessor._hb_aug_version
    engine = None
    if profile != "hb_v2":
        from humanwam.augmentation import SampleKey, UnifiedAugmentation

        seed = int(os.environ.get("HB_AUG_SEED", "42")) if seed is None else seed
        engine = UnifiedAugmentation.from_hb_profile(profile, seed=seed)
        version = engine.profile
    else:
        sys.path.insert(0, os.environ.get("HB_AUG_DIR", ""))
        A = importlib.import_module("hb_aug_v2")
        version = A.VERSION
    pre0 = FastWAMProcessor.preprocess

    def preprocess(self, data):
        sample = _identity(data, SampleKey) if engine is not None and _train(self) else None
        if _train(self) and isinstance(data.get("state"), dict) and "default" in data["state"]:
            st = data["state"]["default"]
            if engine is None:
                q = st[..., :NJ].detach().cpu().numpy()
                st = st.clone()
                st[..., :NJ] = torch.from_numpy(A.perturb_joints(q, _rng())).to(st.dtype)
                data["state"]["default"] = st
            else:
                values = engine.apply_state(st.detach().cpu().numpy(), body_dims=NJ, sample=sample)
                data = dict(data, state={**data["state"], "default": torch.from_numpy(values).to(st)})
            if not _seen["state"]:
                _seen["state"] = True
                print(f"HB_FW_AUG proprio {version} on state[..., :{NJ}] shape {tuple(st.shape)}", flush=True)
        result = pre0(self, data)
        if sample is not None:
            result[_SAMPLE] = sample
        return result

    get0 = BaseLerobotDataset.__getitem__

    def getitem(self, idx):
        s = get0(self, idx)
        sample = s.pop(_SAMPLE, None)
        if self.processor is not None and _train(self.processor) and "pixel_values" in s:
            pv = s["pixel_values"]
            five = pv.ndim == 5
            x = (pv if five else pv[None]).permute(1, 0, 2, 3, 4).float()  # [T, V, C, H, W]
            hi = float(x.max())
            scale = 255.0 if hi > 1.5 else 1.0
            mask = torch.zeros(x.shape[0], dtype=torch.bool)
            mask[0] = True
            if engine is None:
                y, p = A.augment((x / scale).clamp(0, 1), _rng(idx), mask)
            else:
                if sample is None:
                    raise ValueError("Unified FastWAM visual/state sample identity was lost")
                y = engine.apply_rgb((x / scale).clamp(0, 1), sample=sample, observation_mask=mask)
                p = {"sample": [sample.episode, sample.frame, sample.epoch]}
            y = (y * scale).permute(1, 0, 2, 3, 4)
            s["pixel_values"] = (y if five else y[0]).to(pv.dtype)
            if not _seen["img"]:
                _seen["img"] = True
                details = f"boost={p['boost']}" if engine is None else str(p)
                print(f"HB_FW_AUG visual {version} on pixel_values {tuple(pv.shape)} dtype {pv.dtype} max {hi:.3f} "
                      f"(T={x.shape[0]}, obs frame 0) {details}", flush=True)
        return s

    if engine is not None:
        from fastwam.datasets.lerobot.robot_video_dataset import RobotVideoDataset
        from fastwam.utils.samplers import ResumableEpochSampler

        sampler_iter = ResumableEpochSampler.__iter__

        def epoch_indices(self):
            epoch = int(self.epoch) + int(self.epoch_offset)
            if epoch < 0:
                raise ValueError("Augmentation epoch must be nonnegative")
            return ((int(index), epoch) for index in sampler_iter(self))

        def video_getitem(self, index):
            epoch = _epoch.get()
            if isinstance(index, tuple):
                if len(index) != 2 or any(type(value) is not int or value < 0 for value in index):
                    raise ValueError("Unified FastWAM sampler requires (index, epoch) nonnegative integers")
                index, epoch = index
            token = _epoch.set(int(epoch))
            try:
                # Native _get retains its padding retry. Surface contract failures
                # instead of the native catch-all's recursive random fallback.
                return self._get(index)
            finally:
                _epoch.reset(token)

        ResumableEpochSampler.__iter__ = epoch_indices
        RobotVideoDataset.__getitem__ = video_getitem
    FastWAMProcessor.preprocess = preprocess
    BaseLerobotDataset.__getitem__ = getitem
    FastWAMProcessor._hb_aug_profile = profile
    FastWAMProcessor._hb_aug_version = version
    print(f"HB_FW_AUG installed: {version}, joints {NJ}, seed {seed}", flush=True)
    return version
