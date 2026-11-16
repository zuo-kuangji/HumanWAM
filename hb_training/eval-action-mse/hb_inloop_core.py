"""Shared machinery of the in-training open-loop action MSE (Psi-Zero, Delta Flow; GR00T has its own tested copy in
hb_groot/hb_inloop_eval.py with the same protocol).

InLoop(model_name, task, split, out_jsonl, prepare, predict):
  prepare(queries) -> optional per-model setup on this rank's queries (first eval only, inside the RNG fork)
  predict(batch, seed) -> np.ndarray (B, >=16, >=Da) in raw action units; the first 16 steps and the first Da dims
                          (Da = benchmark action dim of the layout) are scored.
InLoop.evaluate(step, model): all train/eval modes of model.modules() and every RNG state (torch CPU + this CUDA device,
numpy, python) are restored afterwards; partial sums are all-reduced once (error flag included, so a failure on one
rank cannot leave the others waiting); rank 0 appends one json line per eval. Any failure disables later evals.
Protocol and metrics = hb_eval_common (post-hoc eval): queries every 16 frames, nMSE with the training-frame std.
"""
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hb_eval_common as E  # noqa: E402

EVERY = int(os.environ.get("HB_INLOOP_EVERY", "250"))
BATCH = int(os.environ.get("HB_INLOOP_BATCH", "8"))
MAX_Q = int(os.environ.get("HB_INLOOP_MAX_QUERIES", "0"))
SEED = 1234


def dist_info():
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def pick_split(task):
    split = os.environ.get("HB_INLOOP_SPLIT")
    if not split:
        split = "heldout" if (E.split_dir(task, "heldout") / "meta.json").exists() else "train10"
    if not (E.split_dir(task, split) / "meta.json").exists():
        raise FileNotFoundError(f"eval cache {task}/{split} missing (build_cache.py {task} {split})")
    return split


class InLoop:
    def __init__(self, model_name, task, split, out_jsonl, predict, prepare=None, cams=None, info=None):
        self.model_name, self.task, self.split = model_name, task, split
        self.out = Path(out_jsonl)
        self.predict, self.prepare = predict, prepare
        self.cams = cams
        self.info = info or {}
        self.disabled = False
        self.ready = False
        self.done_steps = set()

    def say(self, *a):
        if dist_info()[0] == 0:
            print("HB_INLOOP", *a, flush=True)

    def _setup(self):
        rank, world = dist_info()
        meta = E.load_meta(self.task, self.split)
        self.meta, self.lay = meta, E.LAYOUTS[meta["layout"]]
        self.std = E.norm_std(E.load_stats(self.task))
        cams = self.cams or meta["cams"]
        miss = [c for c in cams if c not in meta["cams"]]
        if miss:
            raise KeyError(f"cameras {miss} not in the eval cache {meta['cams']}")
        allq = []
        for ep in meta["episodes"]:
            with np.load(E.split_dir(self.task, self.split) / f"ep_{ep:06d}.npz") as z:
                allq += [(ep, qi, int(t)) for qi, t in enumerate(z["query_t"])]
        if MAX_Q:
            allq = allq[:MAX_Q]
        self.n_total = len(allq)
        mine = [q for i, q in enumerate(allq) if i % world == rank]
        self.eps = list(meta["episodes"])
        by_ep = {}
        for ep, qi, t in mine:
            by_ep.setdefault(ep, []).append((qi, t))
        self.q = []
        for ep, lst in by_ep.items():
            with np.load(E.split_dir(self.task, self.split) / f"ep_{ep:06d}.npz") as z:
                state, action = z["state"], z["action"]
                prompt = meta["prompts"][str(int(z["task_index"]))]
                imgs = {c: z[f"img_{c}"] for c in cams}
                for qi, t in lst:
                    gt, valid = E.gt_chunk(action, t)
                    self.q.append(dict(episode=int(ep), t=int(t), T=len(action), state=state[t].copy(), prompt=prompt,
                                       images={c: imgs[c][qi].copy() for c in cams}, gt=gt, valid=valid,
                                       past_action=action[max(0, t - E.HORIZON):t].copy()))
        if self.prepare:
            self.prepare(self.q)
        self.ready = True
        self.say(f"{self.model_name} ready: task {self.task} split {self.split} queries {self.n_total} "
                 f"(rank {rank}/{world}: {len(self.q)}), every {EVERY} steps, batch {BATCH}, cams {cams}")

    def evaluate(self, step, model):
        if self.disabled or step in self.done_steps:
            return
        self.done_steps.add(step)
        rank, world = dist_info()
        t0 = time.time()
        err, msg = 0.0, ""
        acc, nq, infer = None, 0, 0.0
        modes = [(m, m.training) for m in model.modules()]
        dev = torch.cuda.current_device() if torch.cuda.is_available() else None
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[dev] if dev is not None else []):
                if not self.ready:
                    self._setup()
                D = self.lay["action_dim"]
                model.eval()
                acc = E.Accumulator(self.std, self.lay["groups"])
                for bi in range(0, len(self.q), BATCH):
                    batch = self.q[bi:bi + BATCH]
                    seed = SEED + 1000 * rank + bi
                    torch.default_generator.manual_seed(seed)
                    if dev is not None:
                        torch.cuda.manual_seed(seed)
                    s = time.time()
                    with torch.no_grad():
                        pred = np.asarray(self.predict(batch, seed), dtype=np.float64)
                    infer += time.time() - s
                    if pred.ndim != 3 or pred.shape[0] != len(batch) or pred.shape[1] < E.HORIZON or pred.shape[2] < D:
                        raise ValueError(f"prediction {pred.shape}, need ({len(batch)}, >={E.HORIZON}, >={D})")
                    for q, p in zip(batch, pred):
                        acc.add(q["episode"], p[:E.HORIZON, :D], q["gt"], q["valid"])
                        nq += 1
        except Exception as e:  # noqa: BLE001 - never stop training
            err, msg = 1.0, f"{type(e).__name__}: {e}"
            print(f"HB_INLOOP {self.model_name} rank {rank} step {step} FAILED: {msg}\n{traceback.format_exc()}",
                  file=sys.stderr, flush=True)
        finally:
            for m, tr in modes:
                m.training = tr
            np.random.set_state(np_state)
            random.setstate(py_state)
        D = self.lay["action_dim"] if self.ready else 0
        H = E.HORIZON
        eps = self.eps if self.ready else []
        nE = len(eps)
        vec = np.zeros(3 * D + 1 + 2 * H + 2 * nE + 3)
        if acc is not None and not err:
            o = 0
            vec[o:o + D] = acc.sse_raw; o += D
            vec[o:o + D] = acc.sae_raw; o += D
            vec[o:o + D] = acc.sse_n; o += D
            vec[o] = acc.rows; o += 1
            vec[o:o + H] = acc.step_sse_n; o += H
            vec[o:o + H] = acc.step_rows; o += H
            for j, ep in enumerate(eps):
                s_, c_ = acc.per_episode.get(int(ep), (0.0, 0))
                vec[o + j], vec[o + nE + j] = s_, c_
            o += 2 * nE
            vec[o], vec[o + 1] = nq, infer
        vec[-1] = err
        tot = self._allreduce(vec)
        if tot is None or tot[-1] > 0 or not self.ready:
            self.disabled = True
            nfail = None if tot is None else int(tot[-1])
            self._write({"step": step, "error": msg or "failed on another rank / collective failed",
                         "ranks_failed": nfail})
            self.say(f"{self.model_name} step {step}: eval failed ({nfail} rank(s)) -> in-training eval disabled: {msg}")
            return
        o = 0
        acc = E.Accumulator(self.std, self.lay["groups"])
        acc.sse_raw = tot[o:o + D]; o += D
        acc.sae_raw = tot[o:o + D]; o += D
        acc.sse_n = tot[o:o + D]; o += D
        acc.rows = int(round(tot[o])); o += 1
        acc.step_sse_n = tot[o:o + H]; o += H
        acc.step_rows = tot[o:o + H]; o += H
        acc.per_episode = {int(ep): [float(tot[o + j]), int(round(tot[o + nE + j]))]
                           for j, ep in enumerate(eps) if tot[o + nE + j] > 0}
        o += 2 * nE
        n_q, infer_sum = int(round(tot[o])), float(tot[o + 1])
        res = acc.result()
        res.pop("nmse_by_dim", None)
        rec = {"step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": self.model_name,
               "task": self.task, "split": self.split, "queries": n_q, "ranks": world,
               "wall_s": round(time.time() - t0, 1), "infer_s_per_query": round(infer_sum / max(n_q, 1), 4),
               "seed": SEED, **self.info, **res}
        self._write(rec)
        g = " ".join(f"{k} {v['nmse']:.4f}" for k, v in res["groups"].items())
        self.say(f"{self.model_name} step {step}: {self.task}/{self.split} nMSE {res['nmse']:.4f} ({g}) on {n_q} "
                 f"queries, {rec['wall_s']} s")

    @staticmethod
    def _allreduce(vec):
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()) or dist.get_world_size() == 1:
            return vec
        try:
            dev = "cuda" if dist.get_backend() == "nccl" else "cpu"
            n = torch.tensor([float(len(vec))], dtype=torch.float64, device=dev)
            mx, mn = n.clone(), n.clone()
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)
            dist.all_reduce(mn, op=dist.ReduceOp.MIN)
            if mx.item() != mn.item():
                return None
            t = torch.from_numpy(vec).to(device=dev, dtype=torch.float64)
            dist.all_reduce(t)
            return t.cpu().numpy()
        except Exception as e:  # noqa: BLE001
            print(f"HB_INLOOP all_reduce failed: {e!r}", file=sys.stderr, flush=True)
            return None

    def _write(self, rec):
        if dist_info()[0] != 0:
            return
        try:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            with open(self.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"HB_INLOOP write failed: {e!r}", flush=True)


class TempAttrs:
    """set attributes on an object for the duration of a block, then restore (or delete) them"""
    _MISSING = object()

    def __init__(self, obj, **kw):
        self.obj, self.kw, self.old = obj, kw, {}

    def __enter__(self):
        for k, v in self.kw.items():
            self.old[k] = self.obj.__dict__.get(k, self._MISSING)
            setattr(self.obj, k, v)
        return self.obj

    def __exit__(self, *exc):
        for k, v in self.old.items():
            if v is self._MISSING:
                try:
                    delattr(self.obj, k)
                except AttributeError:
                    pass
            else:
                setattr(self.obj, k, v)
        return False
