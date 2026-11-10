"""GR00T N1.7 fine-tuning entry with Humanoid Benchmark augmentation v2.

Patches Gr00tN1d7Processor in training mode only:
- images: native random crop / colour jitter / rotation are bypassed (the deterministic eval resize+center-crop is
  used), then hb_aug_v2 is applied jointly to all cameras of the sample;
- proprio: the body-joint state groups (HB_BODY_STATE_KEYS) get the v2 bias + noise before normalization.
Then runs the pinned upstream gr00t/experiment/launch_finetune.py unchanged.
"""
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hb_offline_fix  # noqa: E402,F401  (tokenizer load under HF_HUB_OFFLINE=1, see module docstring)
import hb_aug_v2 as A  # noqa: E402
from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as P  # noqa: E402

Proc = P.Gr00tN1d7Processor
BODY_KEYS = [k for k in os.environ.get("HB_BODY_STATE_KEYS", "left_leg,right_leg,waist,left_arm,right_arm").split(",") if k]
LOG = os.environ.get("HB_AUG_LOG")
_state = {"rng": None, "count": 0}


def _rng():
    if _state["rng"] is None:
        _state["rng"] = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(8), "little")])
    return _state["rng"]


_orig_get, _orig_apply, _orig_call = Proc._get_vlm_inputs, Proc._apply_vlm_processing, Proc.__call__


def _get_vlm_inputs(self, image_keys, images, masks, image_transform, language):
    if getattr(self, "training", False):
        self._hb_views = len(image_keys)
        try:
            return _orig_get(self, image_keys, images, masks, self.eval_image_transform, language)
        finally:
            self._hb_views = None
    return _orig_get(self, image_keys, images, masks, image_transform, language)


def _apply_vlm_processing(self, images, language):
    views = getattr(self, "_hb_views", None)
    if views and isinstance(images, torch.Tensor):
        shape = images.shape
        x = images.reshape(-1, views, *shape[1:]).float().div(255.0)
        x, p = A.augment(x, _rng())
        images = (x * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).reshape(shape)
        _state["count"] += 1
        if LOG and _state["count"] in (1, 50):
            with open(f"{LOG}.{os.getpid()}", "a") as f:
                f.write(json.dumps({"count": _state["count"], "views": views, "hw": list(shape[-2:]),
                                    "boost": p["boost"], "version": A.VERSION}) + "\n")
    return _orig_apply(self, images, language)


def _call(self, messages):
    if getattr(self, "training", False) and BODY_KEYS:
        states = messages[0]["content"].states
        for key in BODY_KEYS:
            if key in states:
                states[key] = A.perturb_joints(states[key], _rng())
    return _orig_call(self, messages)


Proc._get_vlm_inputs = _get_vlm_inputs
Proc._apply_vlm_processing = _apply_vlm_processing
Proc.__call__ = _call

if os.environ.get("HB_PATCH_ONLY") != "1":
    entry = Path(os.environ["GROOT_SRC"]) / "gr00t/experiment/launch_finetune.py"
    sys.argv = [str(entry)] + sys.argv[1:]
    runpy.run_path(str(entry), run_name="__main__")
