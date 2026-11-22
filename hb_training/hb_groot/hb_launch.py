"""GR00T N1.7 fine-tuning entry with Humanoid Benchmark augmentation v2.

Patches Gr00tN1d7Processor in training mode only:
- images: native random crop / colour jitter / rotation are bypassed (the deterministic eval resize+center-crop is
  used), then hb_aug_v2 is applied jointly to all cameras of the sample;
- proprio: the body-joint state groups (HB_BODY_STATE_KEYS) get the v2 bias + noise before normalization.
Then runs the pinned upstream gr00t/experiment/launch_finetune.py unchanged.

HB_AUG_PROFILE=unified (or unified_v3) selects the current shared recipe;
unified_v2 retains the preceding 5% clean-image recipe;
unified_v1 explicitly retains its earlier 20% recipe. It uses
loader episode IDs and row-step anchors with epoch=0; native sharded scheduling
does not provide the trainer's epoch at this processor boundary.
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
_PROFILE = os.environ.get("HB_AUG_PROFILE", "hb_v2")
if _PROFILE not in {"hb_v2", "unified", "unified_v1", "unified_v2", "unified_v3"}:
    raise ValueError(f"Unknown HB_AUG_PROFILE: {_PROFILE!r}; choose hb_v2, unified, unified_v1, unified_v2 or unified_v3")
_UNIFIED = _PROFILE != "hb_v2"
if _UNIFIED:
    from humanwam.augmentation import SampleKey, UnifiedAugmentation

    _shared_aug = UnifiedAugmentation.from_hb_profile(_PROFILE, seed=int(os.environ.get("HB_AUG_SEED", "42")))

    # Native processing drops the episode ID when producing its modality-only
    # dataframe. Attach the real loader ID before extraction; no content hash or
    # worker-local counter substitutes for a sample identity.
    from gr00t.data.dataset import sharded_single_step_dataset as _steps
    from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader as _EpisodeLoader

    _orig_parquet = _EpisodeLoader._load_parquet_data
    _orig_extract = _steps.extract_step_data

    def _load_parquet_with_identity(self, episode_index):
        frame = _orig_parquet(self, episode_index)
        frame.attrs["hb_episode_index"] = int(episode_index)
        return frame

    def _extract_with_identity(episode_data, step_index, *args, **kwargs):
        if "hb_episode_index" not in episode_data.attrs:
            raise ValueError("Unified GR00T augmentation requires real loader episode identity")
        sample = SampleKey(episode=int(episode_data.attrs["hb_episode_index"]), frame=int(step_index), epoch=0)
        content = _orig_extract(episode_data, step_index, *args, **kwargs)
        content.metadata = {**getattr(content, "metadata", {}), "hb_unified_sample": sample}
        return content

    _EpisodeLoader._load_parquet_data = _load_parquet_with_identity
    _steps.extract_step_data = _extract_with_identity
    print(f"HB_AUG_PROFILE={_PROFILE} profile={_shared_aug.profile} seed={_shared_aug.seed}: GR00T uses real episode/step IDs; epoch=0 "
          "(native sharded processor has no training-epoch identity)", flush=True)


def _rng():
    if _state["rng"] is None:
        _state["rng"] = np.random.default_rng([os.getpid(), int.from_bytes(os.urandom(8), "little")])
    return _state["rng"]


def _training_argv(arguments, *, profile):
    """Set the v3 native parameter through the normal fine-tuning CLI.

    The upstream launcher forwards this one configuration value to both its
    processor and action head. Explicit CLI values win here; hb_launch_v's
    existing model overrides run after CLI parsing and can replace this default.
    """
    result = list(arguments)
    if profile != "humanwam_unified_augmentation_v3":
        return result
    explicit = {"--state_dropout_prob", "--state-dropout-prob"}
    if any(argument.split("=", 1)[0] in explicit for argument in result):
        return result
    print("HB unified v3: CLI default state_dropout_prob=0.05; variant model overrides may replace it", flush=True)
    return result + ["--state_dropout_prob", "0.05"]


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
        if _UNIFIED:
            sample = getattr(self, "_hb_unified_sample", None)
            if not isinstance(sample, SampleKey):
                raise ValueError("Unified GR00T augmentation requires sample metadata")
            x = _shared_aug.apply_rgb(x, sample=sample, observation_mask=self._hb_observation_mask)
            p = {"boost": None}
        else:
            x, p = A.augment(x, _rng())
        images = (x * 255.0 + 0.5).clamp(0, 255).to(torch.uint8).reshape(shape)
        _state["count"] += 1
        if LOG and _state["count"] in (1, 50):
            with open(f"{LOG}.{os.getpid()}", "a") as f:
                f.write(json.dumps({"count": _state["count"], "views": views, "hw": list(shape[-2:]),
                                    "boost": p["boost"], "version": _shared_aug.profile if _UNIFIED else A.VERSION,
                                    **({"epoch_source": "fixed_zero_native_unavailable", "epoch": 0, "seed": _shared_aug.seed, "profile": _shared_aug.profile} if _UNIFIED else {})}) + "\n")
    return _orig_apply(self, images, language)


def _call(self, messages):
    if _UNIFIED and getattr(self, "training", False):
        import copy

        content = messages[0]["content"]
        sample = getattr(content, "metadata", {}).get("hb_unified_sample")
        if not isinstance(sample, SampleKey):
            raise ValueError("Unified GR00T augmentation requires real episode/frame sample metadata")
        # Copy raw state before normalization, preserving actions and all other
        # state fields exactly. One combined body vector preserves joint order.
        content = copy.copy(content)
        content.states = dict(content.states)
        keys = [key for key in BODY_KEYS if key in content.states]
        if keys:
            pieces = [np.asarray(content.states[key]) for key in keys]
            widths = [piece.shape[-1] for piece in pieces]
            joints = _shared_aug.apply_state(np.concatenate(pieces, axis=-1), body_dims=sum(widths), sample=sample)
            offset = 0
            for key, width in zip(keys, widths):
                content.states[key] = joints[..., offset:offset + width].copy()
                offset += width
        # No augmentation metadata is passed through to the native processor or
        # model. Image processing reads scoped context on this processor only.
        content.metadata = {key: value for key, value in content.metadata.items() if key != "hb_unified_sample"}
        wrapped = [{**messages[0], "content": content}]
        config = self.modality_configs[content.embodiment.value]["video"]
        previous = (getattr(self, "_hb_unified_sample", None), getattr(self, "_hb_observation_mask", None))
        self._hb_unified_sample = sample
        self._hb_observation_mask = [index <= 0 for index in config.delta_indices]
        try:
            return _orig_call(self, wrapped)
        finally:
            self._hb_unified_sample, self._hb_observation_mask = previous
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
    entry = Path(os.environ["GROOT_SRC"]) / "gr00t/experiment/launch_finetune.py"
    arguments = sys.argv[1:]
    if _UNIFIED:
        arguments = _training_argv(arguments, profile=_shared_aug.profile)
    sys.argv = [str(entry)] + arguments
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
