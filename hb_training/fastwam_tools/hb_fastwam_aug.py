"""2026-11-16: Humanoid Benchmark augmentation v2 (project CLAUDE.md) inside the native FastWAM data pipeline.
- Visual: hb_aug_v2.augment on every training sample's frames before the two cameras are concatenated; frame 0 is the
  current observation (sensor / optics effects only there), the later frames are the future video targets (colour,
  per-camera geometry, light patches and white balance shared with frame 0, as the spec requires).
- Proprio: hb_aug_v2.perturb_joints on the raw body joints (G1 29, H2 31 = body 29 + head 2) before normalization;
  hands and gravity untouched; supervision labels untouched. Normalization statistics are computed on the raw data.
FastWAM's own image augmentation is only a resize (no colour jitter / crop), so nothing native is stacked.
Installed in the main process before the dataloaders fork; HB_FW_JOINTS sets the number of perturbed joints."""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.environ.get("HB_AUG_DIR", ""))
import hb_aug_v2 as A  # noqa: E402

NJ = int(os.environ.get("HB_FW_JOINTS", "29"))
_seen = {"img": False, "state": False}
_calls = {"pid": None, "n": 0}


def _worker_id():
    info = torch.utils.data.get_worker_info()
    return 0 if info is None else info.id + 1


# 2026-11-23 user (hb_training/REPRODUCIBILITY.md section 3): with HB_AUG_SEED set -- the training entry sets it
# for every run it starts -- the augmentation stream is derived from (training seed, torch.initial_seed() of this
# process / DataLoader worker, global rank, worker id), so a rerun with the same seed and layout draws the same
# augmentations. Processes started before this change have no HB_AUG_SEED and keep the old pid/urandom stream.
# One generator per call; the per-process call counter keeps successive draws in a worker distinct.
def _rng(idx=0):
    if os.environ.get("HB_AUG_SEED") is not None:
        pid = os.getpid()
        if _calls["pid"] != pid:
            _calls["pid"], _calls["n"] = pid, 0
        _calls["n"] += 1
        return np.random.default_rng([int(os.environ["HB_AUG_SEED"]), torch.initial_seed() % (2 ** 63),
                                      int(os.environ.get("RANK", "0")), _worker_id(), int(idx) & 0xFFFFFFFF, _calls["n"]])
    return np.random.default_rng([os.getpid(), int(idx) & 0xFFFFFFFF, time.time_ns() & 0xFFFFFFFF])


def _train(proc):
    v = getattr(proc, "is_train", False)
    return bool(v() if callable(v) else v)


def install():
    from fastwam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor

    pre0 = FastWAMProcessor.preprocess

    def preprocess(self, data):
        if _train(self) and isinstance(data.get("state"), dict) and "default" in data["state"]:
            st = data["state"]["default"]
            q = st[..., :NJ].detach().cpu().numpy()
            st = st.clone()
            st[..., :NJ] = torch.from_numpy(A.perturb_joints(q, _rng())).to(st.dtype)
            data["state"]["default"] = st
            if not _seen["state"]:
                _seen["state"] = True
                print(f"HB_FW_AUG proprio perturb_joints on state[..., :{NJ}] shape {tuple(st.shape)}", flush=True)
        return pre0(self, data)

    get0 = BaseLerobotDataset.__getitem__

    def getitem(self, idx):
        s = get0(self, idx)
        if self.processor is not None and _train(self.processor) and "pixel_values" in s:
            pv = s["pixel_values"]
            five = pv.ndim == 5
            x = (pv if five else pv[None]).permute(1, 0, 2, 3, 4).float()  # [T, V, C, H, W]
            hi = float(x.max())
            scale = 255.0 if hi > 1.5 else 1.0
            mask = torch.zeros(x.shape[0], dtype=torch.bool)
            mask[0] = True
            y, p = A.augment((x / scale).clamp(0, 1), _rng(idx), mask)
            y = (y * scale).permute(1, 0, 2, 3, 4)
            s["pixel_values"] = (y if five else y[0]).to(pv.dtype)
            if not _seen["img"]:
                _seen["img"] = True
                print(f"HB_FW_AUG visual {A.VERSION} on pixel_values {tuple(pv.shape)} dtype {pv.dtype} max {hi:.3f} "
                      f"(T={x.shape[0]}, obs frame 0) boost={p['boost']}", flush=True)
        return s

    FastWAMProcessor.preprocess = preprocess
    BaseLerobotDataset.__getitem__ = getitem
    print(f"HB_FW_AUG installed: {A.VERSION}, joints {NJ}", flush=True)
