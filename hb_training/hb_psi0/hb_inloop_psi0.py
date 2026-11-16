"""Psi-Zero in-training open-loop action MSE (2026-11-15 user: every 250 steps, like GR00T).

install() is called by hb_psi0/hb_train.py after hb_patches.apply(). It wraps FinetuneTrainer.step: after every
optimizer update whose number is a multiple of HB_INLOOP_EVERY (default 250) all ranks evaluate their shard of the
benchmark eval cache with the live (unwrapped DDP) model through the deployment path of the post-hoc adapter
(eval-action-mse/adapters/ad_psi0.py = psi/deploy serve no_aug path):
  * views: Psi0ModelTransform no_aug branch (resize + center crop of every training camera), PIL input;
  * state: the run's field transform normalize_state_func; instruction lower-cased; pooled CLIP embedding from the
    trainer's own pooled cache (the same tensors it saves as clip_pooled_cache.pt);
  * Psi0Model.predict_action (bf16 autocast inside) with the inference objects Psi0Model.from_pretrained attaches at
    deploy time (Qwen3-VL processor of model_name_or_path, a fresh flow/DDIM scheduler, action_horizon / action_dim,
    device), set only for the duration of the eval and then removed again; 10 denoising steps (adapter default);
  * prediction denormalised with field.denormalize; the first 16 steps and the benchmark action dims are scored
    (A14 v2 "h2p": all 82 dims = body 31 + hands 44 + the 7-D SONIC planner command, planner as its own group).
Results: <project_dir>/inloop-mse.jsonl (rank 0) + "HB_INLOOP" log lines. Training is untouched (modes + RNG restored,
no gradients); any failure disables later evals on all ranks. Not in probe mode (HB_PROBE_OUT).
"""
import json
import os
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


def _scheduler(mc):
    if mc.noise_scheduler == "ddpm":  # exactly as Psi0Model.from_pretrained
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        return DDIMScheduler(num_train_timesteps=mc.train_diffusion_steps, beta_start=0.0001, beta_end=0.02,
                             beta_schedule="squaredcos_cap_v2", clip_sample=True, set_alpha_to_one=True,
                             steps_offset=0, prediction_type="epsilon")
    if mc.noise_scheduler == "flow":
        from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        return FlowMatchEulerDiscreteScheduler(num_train_timesteps=mc.train_diffusion_steps)
    raise ValueError(f"unsupported noise scheduler {mc.noise_scheduler}")


class PsiEval:
    def __init__(self, trainer, out_jsonl=None):
        self.tr = trainer
        cfg = trainer.cfg
        self.mc = cfg.model
        self.field = cfg.data.transform.field
        self.mt = cfg.data.transform.model
        self.cams = [k.split("observation.images.", 1)[-1] for k in cfg.data.transform.repack.image_keys]
        repo = list(cfg.data.train_repo_ids)[0]
        rec = json.loads((Path(cfg.data.root_dir) / repo / "hb-view-receipt.json").read_text())
        task = E.task_for_source(rec["source"])
        if task is None:
            raise LookupError(f"no benchmark task for {rec['source']}")
        split = C.pick_split(task)
        self.steps = int(getattr(self.mc, "num_inference_steps", None) or 10)
        self.model = None
        out = Path(out_jsonl) if out_jsonl else Path(trainer.project_dir) / "inloop-mse.jsonl"
        self.il = C.InLoop("psi0", task, split, out, predict=self.predict, prepare=self.prepare, cams=self.cams,
                           info={"denoising_steps": self.steps, "view": repo, "chunk": int(self.mc.action_chunk_size)})

    def prepare(self, queries):
        from torchvision.transforms import v2
        from transformers import AutoProcessor
        self.tf = v2.Compose([self.mt.resize(), self.mt.center_crop()])
        self.proc = AutoProcessor.from_pretrained(self.mc.model_name_or_path)
        self.sched = _scheduler(self.mc)
        prompts = {q["prompt"].lower() for q in queries}
        if getattr(self.mc, "pooled_text_encoder", None) is not None:
            miss = [p for p in prompts if p not in self.tr._pooled_cache]
            if miss:
                raise KeyError(f"instructions not in the trainer's pooled CLIP cache: {miss[:2]}")

    def predict(self, batch, seed):
        from PIL import Image
        model = self.model
        dev = next(model.parameters()).device
        views = [[self.tf(Image.fromarray(q["images"][c])) for c in self.cams] for q in batch]
        st = np.stack([np.asarray(self.field.normalize_state_func(np.asarray(q["state"], dtype=np.float32)),
                                  dtype=np.float32) for q in batch])
        states = torch.from_numpy(st[:, None]).to(dev)
        instr = [q["prompt"].lower() for q in batch]
        kw = {}
        if getattr(self.mc, "pooled_text_encoder", None) is not None:
            kw["pooled_projections"] = torch.stack([self.tr._pooled_cache[i] for i in instr]).to(dev)
        with C.TempAttrs(model, vlm_processor=self.proc, noise_scheduler=self.sched,
                         action_horizon=int(self.mc.action_chunk_size), action_dim=int(self.mc.action_dim),
                         device=dev):
            a = model.predict_action(observations=views, states=states, traj2ds=None, instructions=instr,
                                     num_inference_steps=self.steps, **kw).float().cpu().numpy()
        return np.stack([np.asarray(self.field.denormalize(x), dtype=np.float64) for x in a])

    def evaluate(self, step, model):
        self.model = model
        try:
            self.il.evaluate(step, model)
        finally:
            self.model = None


class _Broken:
    """construction failed on this rank: still take part in the all-reduce so no rank waits"""

    def __init__(self, err):
        self.err = err
        # any cached task: _setup reaches prepare, which raises -> the error flag is all-reduced like a real failure
        self.il = C.InLoop("psi0", "b6", "train10", "/dev/null", predict=None, prepare=self._boom)

    def _boom(self, _q):
        raise RuntimeError(self.err)

    def evaluate(self, step, model):
        self.il.evaluate(step, model)


def install():
    if C.EVERY <= 0 or os.environ.get("HB_PROBE_OUT"):
        return
    from psi.trainers import finetune as F
    FT = F.FinetuneTrainer
    orig = FT.step

    def step(self, batch_input, global_step, local_step):
        out = orig(self, batch_input, global_step, local_step)
        try:
            sync = bool(out[0])
        except Exception:  # noqa: BLE001
            sync = False
        done = int(global_step) + 1
        if sync and done % C.EVERY == 0:
            ev = getattr(self, "_hb_inloop", None)
            if ev is None:
                try:
                    ev = PsiEval(self)
                except Exception as e:  # noqa: BLE001
                    ev = _Broken(f"{type(e).__name__}: {e}")
                self._hb_inloop = ev
            if not ev.il.disabled:
                ev.evaluate(done, self.unwrap_model())
        return out

    FT.step = step
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"HB_INLOOP psi0 installed: every {C.EVERY} steps", flush=True)
