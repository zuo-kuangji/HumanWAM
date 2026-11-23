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
_state = {"rng": None, "count": 0, "pid": None}


def _worker_id():
    info = torch.utils.data.get_worker_info()
    return 0 if info is None else info.id + 1


# 2026-11-23 user (hb_training/REPRODUCIBILITY.md section 3): with HB_AUG_SEED set -- the training entry sets it
# for every run it starts -- the augmentation stream is derived from (training seed, torch.initial_seed() of this
# process / DataLoader worker, global rank, worker id), so a rerun with the same seed and layout draws the same
# augmentations. Processes started before this change have no HB_AUG_SEED and keep the old pid/urandom stream.
def _rng():
    pid = os.getpid()
    if _state["rng"] is None or _state["pid"] != pid:
        if os.environ.get("HB_AUG_SEED") is not None:
            _state["rng"] = np.random.default_rng([int(os.environ["HB_AUG_SEED"]), torch.initial_seed() % (2 ** 63),
                                  int(os.environ.get("RANK", "0")), _worker_id()])
        else:
            _state["rng"] = np.random.default_rng([pid, int.from_bytes(os.urandom(8), "little")])
        _state["pid"] = pid
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

# 2026-11-18 user: joint47 experiment (hb_j47.py turns the world root rotation6D chunk into current-heading relative)
if os.environ.get("HB_J47_ROOT_KEY"):
    import hb_j47  # noqa: E402,F401

# 2026-11-15 user: in-training open-loop action MSE every 250 steps (hb_inloop_eval.py adds a TrainerCallback to
# Gr00tTrainer; HB_INLOOP_EVERY=0 turns it off). It never blocks training: any setup problem only disables it.
if os.environ.get("HB_PATCH_ONLY") != "1":
    try:
        import hb_inloop_eval  # noqa: E402,F401
    except Exception as _e:  # noqa: BLE001
        print(f"HB_INLOOP disabled: {_e!r}", flush=True)

if os.environ.get("HB_PATCH_ONLY") != "1":
    os.environ.setdefault("HB_AUG_SEED", "42")  # = the GR00T training seed (gr00t data_config seed 42); see _rng()
    print(f"HB_AUG_SEED={os.environ['HB_AUG_SEED']} (seeded augmentation stream, REPRODUCIBILITY.md section 3)", file=sys.stderr, flush=True)
    entry = Path(os.environ["GROOT_SRC"]) / "gr00t/experiment/launch_finetune.py"
    sys.argv = [str(entry)] + sys.argv[1:]
    try:
        runpy.run_path(str(entry), run_name="__main__")
    finally:  # batch probe: report this rank's peak reserved memory (groot_chain.py compares it with MAX_FRAC)
        _peak = os.environ.get("HB_PEAK_FILE")
        if _peak and torch.cuda.is_available():
            try:
                _dev = torch.cuda.current_device()
                with open(f"{_peak}-rank{os.environ.get('RANK', '0')}.json", "w") as _f:
                    json.dump({"rank": int(os.environ.get("RANK", "0")),
                               "peak_reserved_gb": torch.cuda.max_memory_reserved(_dev) / 1e9,
                               "total_gb": torch.cuda.get_device_properties(_dev).total_memory / 1e9}, _f)
            except Exception:  # noqa: BLE001 - never mask the training result
                pass
