"""GR00T in-training open-loop action MSE (2026-11-15 user: every 250 training steps).

Imported by hb_groot/hb_launch.py (hb_launch_v.py runs hb_launch.py). It wraps Gr00tTrainer.__init__ so that every
trainer gets a TrainerCallback. At global_step % HB_INLOOP_EVERY == 0 (default 250; 0 = off) every rank evaluates its
shard of the benchmark eval cache (eval-action-mse/cache) with the weights being trained right now, exactly as deployed:
  * Gr00tPolicy pipeline: eval processor from <output_dir>/processor (the files every checkpoint ships),
    parse_observation_gr00t inputs, bf16 inputs, model.get_action under inference_mode, processor.decode_action;
  * same protocol and metrics as the post-hoc eval (hb_eval_common): queries every 16 frames, first 16 predicted steps
    vs the recorded actions, nMSE with the training-frame std, per group / per step;
  * split: "heldout" when the task has recorded episodes that are never trained on (B6, trolley, B7), else "train10".
Training is not changed: module train/eval modes are restored module by module, torch CPU + CUDA (this device), numpy
and python RNG states are restored after each eval; the flow-matching noise uses a fixed seed (rank, batch) so every
eval sees the same noise (smoother curve). ZeRO-2 keeps the full bf16 weights on every rank, so ranks run independently;
partial sums are all-reduced once. Results: <run>/logs/inloop-mse.jsonl (rank 0), one json line per eval, and an
"HB_INLOOP" log line. Failures never stop training: an exception on any rank disables further evals on all ranks
(error flag in the same all-reduce, so no rank waits on another).
Env: HB_INLOOP_EVERY (250), HB_INLOOP_BATCH (8 queries per forward), HB_INLOOP_MAX_QUERIES (0 = all; tests only),
     HB_INLOOP_SPLIT (override), HB_INLOOP_PROCESSOR_DIR (tests only).
"""
import json
import os
import random
import sys
import time
import traceback
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

EVERY = int(os.environ.get("HB_INLOOP_EVERY", "250"))
BATCH = int(os.environ.get("HB_INLOOP_BATCH", "8"))
MAX_Q = int(os.environ.get("HB_INLOOP_MAX_QUERIES", "0"))
SEED = 1234
G = Path(__file__).resolve().parent.parent
EVAL_DIR = G / "eval-action-mse"
for _p in (str(EVAL_DIR), str(EVAL_DIR / "adapters")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import hb_eval_common as E  # noqa: E402
import ad_groot as AG  # noqa: E402  (canonical state -> GR00T state groups, same mapping as the post-hoc eval)

RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))


def say(*a):
    if RANK == 0:
        print("HB_INLOOP", *a, flush=True)


def _argv(name):
    a = sys.argv
    for i, x in enumerate(a):
        if x == name and i + 1 < len(a):
            return a[i + 1]
        if x.startswith(name + "="):
            return x.split("=", 1)[1]
    return None


def resolve(dataset_path):
    """groot-data/<view> -> (benchmark task, split, view dir) or raises with the reason"""
    view = Path(dataset_path)
    rec = json.loads((view / "hb-dataset-receipt.json").read_text())
    task = E.task_for_source(rec["source"])
    if task is None:
        raise LookupError(f"no benchmark task for source {rec['source']}")
    split = os.environ.get("HB_INLOOP_SPLIT")
    if not split:
        split = "heldout" if (E.split_dir(task, "heldout") / "meta.json").exists() else "train10"
    if not (E.split_dir(task, split) / "meta.json").exists():
        raise FileNotFoundError(f"eval cache {task}/{split} missing (build_cache.py {task} {split})")
    return task, split, view


class InLoopMSE:
    """transformers TrainerCallback (duck-typed: the callback handler only calls the hooks that exist)"""

    def __init__(self, task, split, view, processor_dir=None):
        self.task, self.split, self.view = task, split, view
        self.processor_dir = processor_dir
        self.disabled = False
        self.ready = False
        self.results = []

    # ---------------------------------------------------------------- setup (first eval; needs the live model)
    def _setup(self, model, output_dir):
        from transformers import AutoProcessor
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy
        from gr00t.policy.policy import BasePolicy

        meta = E.load_meta(self.task, self.split)
        self.meta = meta
        self.lay = E.LAYOUTS[meta["layout"]]
        self.std = E.norm_std(E.load_stats(self.task))
        # the Gr00tPolicy of a checkpoint, built around the live model instead of loading weights from disk
        pol = Gr00tPolicy.__new__(Gr00tPolicy)
        BasePolicy.__init__(pol, strict=True)
        pdir = Path(self.processor_dir or os.environ.get("HB_INLOOP_PROCESSOR_DIR") or Path(output_dir) / "processor")
        proc = AutoProcessor.from_pretrained(str(pdir))
        proc.eval()
        proc.validate_strict_action_padding_mask(model.config)
        tag = EmbodimentTag.NEW_EMBODIMENT
        pol.model, pol.processor, pol.embodiment_tag = model, proc, tag
        pol.modality_configs = {k: v for k, v in proc.get_modality_configs()[tag.value].items() if k != "rl_info"}
        pol.collate_fn = proc.collator
        pol.language_key = pol.modality_configs["language"].modality_keys[0]
        self.policy = pol
        self.mod = pol.modality_configs
        self.state_map = AG.H2_STATE if meta["layout"].startswith("h2") else AG.G1_STATE
        self.video_keys = list(self.mod["video"].modality_keys)
        self.state_keys = list(self.mod["state"].modality_keys)
        self.action_keys = list(self.mod["action"].modality_keys)
        missing = [k for k in self.state_keys if k not in self.state_map] + \
                  [k for k in self.video_keys if k not in meta["cams"]]
        if missing:
            raise KeyError(f"model expects {missing}, not in the benchmark state / cache cameras {meta['cams']}")
        tj = self.view / "meta/tasks.jsonl"
        self.prompts = {str(r["task_index"]): r["task"] for r in E.jl(tj)} if tj.exists() else {}
        # all queries of the split in a fixed order; this rank keeps every WORLD-th one (images only for those)
        allq = []
        for ep in meta["episodes"]:
            with np.load(E.split_dir(self.task, self.split) / f"ep_{ep:06d}.npz") as z:
                allq += [(ep, qi, int(t)) for qi, t in enumerate(z["query_t"])]
        if MAX_Q:
            allq = allq[:MAX_Q]
        self.n_total = len(allq)
        mine = [q for i, q in enumerate(allq) if i % WORLD == RANK]
        self.eps = list(meta["episodes"])
        self.q = []  # (ep, t, T, state[t], prompt, {cam: img}, gt(16, Da), valid(16))
        by_ep = {}
        for ep, qi, t in mine:
            by_ep.setdefault(ep, []).append((qi, t))
        for ep, lst in by_ep.items():
            with np.load(E.split_dir(self.task, self.split) / f"ep_{ep:06d}.npz") as z:
                state, action = z["state"], z["action"]
                prompt = meta["prompts"][str(int(z["task_index"]))]
                imgs = {c: z[f"img_{c}"] for c in self.video_keys}
                for qi, t in lst:
                    gt, valid = E.gt_chunk(action, t)
                    self.q.append((ep, t, len(action), state[t].copy(), prompt,
                                   {c: imgs[c][qi].copy() for c in self.video_keys}, gt, valid))
        self.ready = True
        say(f"ready: task {self.task} split {self.split} queries {self.n_total} (rank {RANK}: {len(self.q)}), "
            f"every {EVERY} steps, batch {BATCH}, cams {self.video_keys}, processor {pdir}")

    # ---------------------------------------------------------------- one forward on a batch of queries
    def _obs(self, q):
        from gr00t.data.utils import parse_observation_gr00t
        ep, t, T, st, prompt, imgs, _, _ = q
        o = {}
        for k in self.state_keys:
            a, b = self.state_map[k]
            o[f"state.{k}"] = np.asarray(st[a:b], dtype=np.float32)[None]
        for k in self.video_keys:
            o[f"video.{k}"] = imgs[k][None]
        o[self.language_key()] = self.prompts.get("0", prompt) if len(self.prompts) == 1 else prompt
        return parse_observation_gr00t(o, self.mod)

    def language_key(self):
        return self.policy.language_key

    @staticmethod
    def _merge(obs_list):
        out = {}
        for m in obs_list[0]:
            out[m] = {}
            for k, v in obs_list[0][m].items():
                vals = [o[m][k] for o in obs_list]
                out[m][k] = np.concatenate(vals, axis=0) if isinstance(v, np.ndarray) else sum(vals, [])
        return out

    def _predict(self, batch, model):
        obs = self._merge([self._obs(q) for q in batch])
        dtype = next(model.parameters()).dtype
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if (dtype != torch.bfloat16 and torch.cuda.is_available()) \
            else nullcontext()
        with ctx:
            act, _ = self.policy.get_action(obs)
        return np.concatenate([np.asarray(act[k], dtype=np.float64) for k in self.action_keys], axis=-1)

    # ---------------------------------------------------------------- the eval
    def evaluate(self, step, model, output_dir):
        t0 = time.time()
        D = self.lay["action_dim"] if self.ready else 0
        err, msg = 0.0, ""
        acc = None
        nq = 0
        infer = 0.0
        modes = [(m, m.training) for m in model.modules()]
        dev = torch.cuda.current_device() if torch.cuda.is_available() else None
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            with torch.random.fork_rng(devices=[dev] if dev is not None else []):
                if not self.ready:
                    self._setup(model, output_dir)
                    D = self.lay["action_dim"]
                model.eval()
                acc = E.Accumulator(self.std, self.lay["groups"])
                for bi in range(0, len(self.q), BATCH):
                    batch = self.q[bi:bi + BATCH]
                    torch.default_generator.manual_seed(SEED + 1000 * RANK + bi)
                    if dev is not None:
                        torch.cuda.manual_seed(SEED + 1000 * RANK + bi)
                    s = time.time()
                    pred = self._predict(batch, model)
                    infer += time.time() - s
                    if pred.shape[0] != len(batch) or pred.shape[1] < E.HORIZON or pred.shape[2] != D:
                        raise ValueError(f"prediction {pred.shape}, need ({len(batch)}, >={E.HORIZON}, {D})")
                    for q, p in zip(batch, pred):
                        acc.add(q[0], p[:E.HORIZON], q[6], q[7])
                        nq += 1
        except Exception as e:  # noqa: BLE001 - never stop training
            err, msg = 1.0, f"{type(e).__name__}: {e}"
            print(f"HB_INLOOP rank {RANK} step {step} FAILED: {msg}\n{traceback.format_exc()}", file=sys.stderr,
                  flush=True)
        finally:
            for m, tr in modes:
                m.training = tr
            np.random.set_state(np_state)
            random.setstate(py_state)
        # ---- one all-reduce of all partial sums (+ error flag); D / eps are identical on every rank once ready
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
                s, c = acc.per_episode.get(int(ep), (0.0, 0))
                vec[o + j], vec[o + nE + j] = s, c
            o += 2 * nE
            vec[o], vec[o + 1] = nq, infer
        vec[-1] = err
        tot = self._allreduce(vec)
        if tot is None:  # ranks disagree on the vector length (setup failed somewhere) -> stop for good
            self.disabled = True
            self._write(output_dir, {"step": step, "error": "length mismatch across ranks / collective failed"})
            return
        if tot[-1] > 0 or not self.ready:
            self.disabled = True
            self._write(output_dir, {"step": step, "error": msg or "failed on another rank", "ranks_failed": tot[-1]})
            say(f"step {step}: eval failed on {int(tot[-1])} rank(s) -> in-training eval disabled ({msg})")
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
        rec = {"step": step, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "task": self.task,
               "split": self.split, "queries": n_q, "ranks": WORLD, "wall_s": round(time.time() - t0, 1),
               "infer_s_per_query": round(infer_sum / max(n_q, 1), 4),
               "denoising_steps": int(getattr(model.action_head, "num_inference_timesteps", -1))
               if hasattr(model, "action_head") else None, "seed": SEED, **res}
        self.results.append(rec)
        self._write(output_dir, rec)
        g = " ".join(f"{k} {v['nmse']:.4f}" for k, v in res["groups"].items())
        say(f"step {step}: {self.task}/{self.split} nMSE {res['nmse']:.4f} ({g}) on {n_q} queries, "
            f"{rec['wall_s']} s")

    def _allreduce(self, vec):
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()) or WORLD == 1:
            return vec
        try:
            dev = "cuda" if dist.get_backend() == "nccl" else "cpu"
            n = torch.tensor([float(len(vec))], dtype=torch.float64, device=dev)
            mx = n.clone()
            dist.all_reduce(mx, op=dist.ReduceOp.MAX)
            mn = n.clone()
            dist.all_reduce(mn, op=dist.ReduceOp.MIN)
            if mx.item() != mn.item():
                return None
            t = torch.from_numpy(vec).to(device=dev, dtype=torch.float64)
            dist.all_reduce(t)
            return t.cpu().numpy()
        except Exception as e:  # noqa: BLE001
            print(f"HB_INLOOP rank {RANK} all_reduce failed: {e!r}", file=sys.stderr, flush=True)
            return None

    def _write(self, output_dir, rec):
        if RANK != 0:
            return
        try:
            out = Path(output_dir).parent / "logs" / "inloop-mse.jsonl"
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"HB_INLOOP write failed: {e!r}", flush=True)

    # ---------------------------------------------------------------- TrainerCallback hooks
    def on_step_end(self, args, state, control, model=None, **kw):
        step = int(state.global_step)
        if self.disabled or EVERY <= 0 or step <= 0 or step % EVERY or model is None:
            return control
        self.evaluate(step, model, args.output_dir)
        return control


def _install():
    if EVERY <= 0:
        say("disabled (HB_INLOOP_EVERY=0)")
        return
    ds = _argv("--dataset_path") or _argv("--dataset-path")
    if not ds:
        say("disabled: no --dataset_path in argv")
        return
    try:
        task, split, view = resolve(ds)
    except Exception as e:  # noqa: BLE001
        say(f"disabled: {e}")
        return
    from transformers import TrainerCallback
    from gr00t.experiment.trainer import Gr00tTrainer

    cb_cls = type("HBInLoopMSE", (InLoopMSE, TrainerCallback), {})
    _orig_init = Gr00tTrainer.__init__

    def __init__(self, *a, **k):
        _orig_init(self, *a, **k)
        self.add_callback(cb_cls(task, split, view))

    Gr00tTrainer.__init__ = __init__
    say(f"installed: task {task} split {split} every {EVERY} steps (view {view.name})")


if os.environ.get("HB_INLOOP_NO_INSTALL") != "1":
    _install()
