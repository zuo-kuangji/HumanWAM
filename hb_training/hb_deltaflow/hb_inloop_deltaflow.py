"""Delta Flow (ImageWAM) in-training open-loop action MSE (2026-11-15 user: every 250 steps, like GR00T).

install(TrainerCls, cfg, run_root) is called by scripts/wholebody/train_deltaflow.py (every Delta-flow code tree)
right before runtime.run_training. It wraps the trainer's _checkpoint_save_flags, which the native loop calls once
after every optimizer update with the new global_step; at multiples of HB_INLOOP_EVERY (default 250) all 32 ranks
evaluate their shard of the benchmark eval cache with the live model (ZeRO-1: full weights on every rank) through
the post-hoc adapter's path (eval-action-mse/adapters/ad_deltaflow.py):
  * inputs from the run's own NON-augmented dataset class (first imagewam dataset class in the MRO of cfg.data.train
    without aug / Robust), split "validation" for heldout and "train" for train10: two 224 views, normalised state,
    normalised 16-step past-action history, cached Qwen instruction embedding = the training-time tensors;
    the dataset ground truth must equal the cache ground truth (max abs diff < 1e-3) or the eval is disabled;
  * model.infer_action_flux2 with its native defaults (1 action step, history + 0.5 noise), fixed seed 1234;
  * prediction denormalised with the dataset normaliser; first 16 steps and the benchmark action dims scored.
The dataset samples of this rank's queries are decoded once (first eval) and kept in memory.
Results: <run>/run/production/inloop-mse.jsonl (rank 0) + "HB_INLOOP" log lines. Training is untouched (modes + RNG
restored, no gradients); any failure disables later evals on all ranks.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

G = Path(__file__).resolve().parent.parent
EVAL = G / "eval-action-mse"
if str(EVAL) not in sys.path:
    sys.path.insert(0, str(EVAL))
import hb_eval_common as E  # noqa: E402
import hb_inloop_core as C  # noqa: E402

DF_SEED = 1234


def base_dataset(cfg, split):
    from hydra.utils import get_class
    from omegaconf import OmegaConf
    d = OmegaConf.to_container(cfg.data.train, resolve=True)
    aug_cls = get_class(d.pop("_target_"))
    d.pop("image_augmentation", None)
    d.pop("proprio_augmentation", None)
    base = next(k for k in aug_cls.__mro__ if k.__module__.startswith("imagewam.datasets")
                and "aug" not in k.__module__.lower() and "aug" not in k.__name__.lower()
                and "Robust" not in k.__name__ and "__getitem__" in vars(k))
    d["split"] = "validation" if split == "heldout" else "train"
    return base(**d), f"{base.__module__}.{base.__name__}"


class DFEval:
    def __init__(self, cfg, run_root, out_jsonl=None):
        self.cfg = cfg
        rec = json.loads((Path(run_root) / "data-verified.json").read_text())
        task = E.task_for_source(rec["dataset_dir"])
        if task is None:
            raise LookupError(f"no benchmark task for {rec['dataset_dir']}")
        split = C.pick_split(task)
        out = Path(out_jsonl) if out_jsonl else Path(cfg.output_dir) / "inloop-mse.jsonl"
        self.model = None
        self.il = C.InLoop("deltaflow", task, split, out, predict=self.predict, prepare=self.prepare, cams=[],
                           info={"denoising_steps": "native(1)", "uses_past_action_history": True,
                                 "seed_df": DF_SEED, "run": Path(run_root).name})
        # override the batch size: infer_action_flux2 is called per query
        self.max_gt_diff = 0.0

    def _denorm(self, x):
        n = self.ds.normalizer
        for name in ("backward", "inverse", "denormalize"):
            if hasattr(n, name):
                return np.asarray(getattr(n, name)(np.asarray(x, dtype=np.float64), "action"))
        raise AttributeError("normaliser has no inverse")

    def prepare(self, queries):
        self.ds, self.ds_name = base_dataset(self.cfg, self.il.split)
        starts = {ep: int(self.ds.ends[i - 1]) if i else 0 for i, ep in enumerate(self.ds.episode_ids)}
        keep = ("video", "proprio", "text_hidden_states", "text_attention_mask", "past_action")
        for q in queries:
            ep, t = q["episode"], q["t"]
            if ep not in starts:
                raise KeyError(f"episode {ep} not in the native {self.il.split} split of {self.ds_name}")
            s = self.ds[starts[ep] + t]
            assert int(s["episode_index"]) == ep and int(s["frame_index"]) == t
            gt = self._denorm(s["action"].float().numpy())[: int(q["valid"].sum())]
            ref = q["gt"][: len(gt)]
            self.max_gt_diff = max(self.max_gt_diff, float(np.abs(gt - ref).max()))
            if self.max_gt_diff > 1e-3:
                raise ValueError(f"native dataset GT differs from the benchmark GT by {self.max_gt_diff} (ep {ep} t {t})")
            q["df"] = {k: (s[k][:, 0].clone() if k == "video" else s[k].clone()) for k in keep}
        self.il.info["dataset_class"] = self.ds_name
        self.il.info["max_abs_gt_diff_vs_cache"] = self.max_gt_diff

    def predict(self, batch, seed):
        model = self.model
        out = []
        for q in batch:
            s = q["df"]
            o = model.infer_action_flux2(prompt=None, input_image=s["video"], action_horizon=16, proprio=s["proprio"][0],
                                         context=s["text_hidden_states"], context_mask=s["text_attention_mask"],
                                         action_init=s["past_action"], seed=DF_SEED)
            out.append(self._denorm(o["action"].float().cpu().numpy()))
        return np.stack(out)

    def evaluate(self, step, model):
        self.model = model
        try:
            self.il.evaluate(step, model)
        finally:
            self.model = None


class _Broken:
    def __init__(self, err):
        self.err = err
        self.il = C.InLoop("deltaflow", "b6", "train10", "/dev/null", predict=None, prepare=self._boom, cams=[])

    def _boom(self, _q):
        raise RuntimeError(self.err)

    def evaluate(self, step, model):
        self.il.evaluate(step, model)


def install(trainer_cls, cfg, run_root):
    if C.EVERY <= 0:
        return
    orig = trainer_cls._checkpoint_save_flags  # staticmethod -> plain function

    def flags(self, **kw):
        step = int(kw.get("global_step", 0))
        if step > 0 and step % C.EVERY == 0:
            ev = getattr(self, "_hb_inloop", None)
            if ev is None:
                try:
                    ev = DFEval(cfg, run_root)
                except Exception as e:  # noqa: BLE001
                    ev = _Broken(f"{type(e).__name__}: {e}")
                self._hb_inloop = ev
            if not ev.il.disabled:
                ev.evaluate(step, self.accelerator.unwrap_model(self.model))
        return orig(**kw)

    trainer_cls._checkpoint_save_flags = flags
    import os
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"HB_INLOOP deltaflow installed: every {C.EVERY} steps", flush=True)
