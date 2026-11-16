"""FastWAM in-training open-loop action MSE (2026-11-16 user: every 250 steps, like GR00T / Psi-Zero / Delta Flow).

install() is called by fastwam/tools/hb_fastwam_train.py right before native.run_training(cfg). It uses the trainer's
own eval slot: after Wan22Trainer.__init__ it sets eval_every = HB_INLOOP_EVERY (default 250) and a non-None
val_dataset marker, and replaces Wan22Trainer.evaluate (the native one generates a full video for one random val sample;
it is off in our runs, eval_every=0 on the command line). The native loop then calls evaluate() on every rank right after
the optimizer step of every 250th update, followed by accelerator.wait_for_everyone(); returning None logs nothing.
evaluate(): all ranks score their shard of the benchmark eval cache with the live model (accelerator.unwrap_model; ZeRO-1
keeps full bf16 weights on every rank) through EXACTLY the post-hoc adapter path (eval-action-mse/adapters/ad_fastwam.py
Inputs + infer: FastWAM.infer_action, the action-only deployment path; 10 denoising steps = config
eval_num_inference_steps; seed 42 per query; same cameras / proprio normalisation / T5 cache as training).
Results: <output_dir>/inloop-mse.jsonl (rank 0) + "HB_INLOOP" log lines. Training is untouched (module modes and every
RNG restored by hb_inloop_core, no gradients); any failure disables later evals on all ranks.
"""
import json
import os
import sys
from pathlib import Path

G = Path(__file__).resolve().parents[2]  # .../policies-20261109 (this file: $G/fastwam/tools/)
EVAL = G / "eval-action-mse"
for p in (str(EVAL), str(EVAL / "adapters")):
    if p not in sys.path:
        sys.path.insert(0, p)
import numpy as np  # noqa: E402
import hb_eval_common as E  # noqa: E402
import hb_inloop_core as C  # noqa: E402
import ad_fastwam as A  # noqa: E402


class _Marker:
    """non-None val_dataset so the native loop reaches evaluate(); never indexed"""

    def __len__(self):
        return 1


class FwEval:
    def __init__(self, trainer):
        cfg = trainer.cfg
        self.cfg = cfg
        ddir = Path(list(cfg.data.train.dataset_dirs)[0])
        iface = json.loads((ddir / "meta/hb_fastwam_interface.json").read_text())
        task = E.task_for_source(iface["source"])
        if task is None:
            raise LookupError(f"no benchmark task for {iface['source']}")
        split = C.pick_split(task)
        self.steps = int(cfg.eval_num_inference_steps)
        self.inp = None
        self.model = None
        out = Path(cfg.output_dir) / "inloop-mse.jsonl"
        self.il = C.InLoop("fastwam", task, split, out, predict=self.predict, prepare=self.prepare,
                           info={"denoising_steps": self.steps, "seed_per_query": 42, "dataset": ddir.name,
                                 "path": "FastWAM.infer_action (ad_fastwam)"})
        self.il.cams = None

    def prepare(self, queries):
        self.inp = A.Inputs(self.cfg, Path(self.cfg.output_dir) / "dataset_stats.json")
        miss = [c for c in self.inp.cams if c not in self.il.meta["cams"]]
        if miss:
            raise KeyError(f"cameras {miss} not in the eval cache {self.il.meta['cams']}")
        for q in {q["prompt"] for q in queries}:
            self.inp.context(q)  # fail early if an instruction has no T5 cache

    def predict(self, batch, seed):
        return np.stack([np.asarray(A.infer(self.model, self.inp, q, self.steps, 42), dtype=np.float64) for q in batch])

    def evaluate(self, step, model):
        self.model = model
        try:
            self.il.evaluate(step, model)
        finally:
            self.model = None


class _Broken:
    def __init__(self, err):
        self.err = err
        self.il = C.InLoop("fastwam", "b6", "train10", "/dev/null", predict=None, prepare=self._boom)

    def _boom(self, _q):
        raise RuntimeError(self.err)

    def evaluate(self, step, model):
        self.il.evaluate(step, model)


def install():
    if C.EVERY <= 0:
        return
    from fastwam import trainer as T
    WT = T.Wan22Trainer
    orig_init = WT.__init__

    def __init__(self, *a, **kw):
        orig_init(self, *a, **kw)
        # native eval off (eval_every 0 on our command line). NB runtime.build_datasets sets val_ds = train_ds when
        # there is no data.val, so val_dataset is never None here (11-16 smoke: the old "val_dataset is None" test
        # silently left the in-loop eval off).
        if int(self.eval_every) == 0:
            self.eval_every = C.EVERY
            if self.val_dataset is None:
                self.val_dataset = _Marker()
            self._hb_inloop_on = True
        else:  # a native eval is configured: leave it alone
            self._hb_inloop_on = False

    def evaluate(self):
        if not getattr(self, "_hb_inloop_on", False):
            return orig_eval(self)
        ev = getattr(self, "_hb_inloop", None)
        if ev is None:
            try:
                ev = FwEval(self)
            except Exception as e:  # noqa: BLE001
                ev = _Broken(f"{type(e).__name__}: {e}")
            self._hb_inloop = ev
        if not ev.il.disabled:
            ev.evaluate(int(self.global_step), self.accelerator.unwrap_model(self.model))
        return None

    orig_eval = WT.evaluate
    WT.__init__ = __init__
    WT.evaluate = evaluate
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"HB_INLOOP fastwam installed: every {C.EVERY} steps", flush=True)
