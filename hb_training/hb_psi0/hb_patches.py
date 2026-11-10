"""Runtime patches on the pinned Psi0 trainer (no upstream file is modified).

1. FinetuneTrainer.init_models (src/psi/trainers/finetune.py:323)
   Native behaviour on an action-dim/horizon mismatch is to load ONLY `transformer_blocks.*` from
   action_header.safetensors. For our recipes (G1 A76/S44/H30, H2 a18 A75/S78/H30; native init is
   A80/S45/H30) we instead load every tensor whose shape still matches (time/text embedders, VLM view
   projection, state_pos/state_null, dec_pos, all 12 blocks, final adaLN) through the native
   `_reconcile_action_header_state_dict` (which already skips shape mismatches), and newly initialize
   (PyTorch default nn.Linear init from the constructor) only the in/out projection tensors.
   The set of newly initialized keys must EQUAL expected_new(Da, odim, Tp); anything else aborts the
   run (catches recipe drift). The list goes to the log and <run>/hb_init_report.json.
   No zero padding: model dims are the real data dims.
2. FinetuneTrainer.__init__ (finetune.py:64-65) builds loss_w = [1/Da]*Da in fp32 and asserts the sum
   is exactly 1.0. If that float32 sum is not exactly 1.0 for Da (76 or 75) on this device, the native
   init runs with action_dim=64 (1/64 exact) and loss_w is then rebuilt as the identical uniform 1/Da
   vector; ac_dim is restored. Nothing else in __init__ depends on action_dim.
5. Trainer.next_epoch -> hb_psi0_config.set_epoch(epoch) (fork-shared value read by DataLoader workers
   for the per-trajectory joint bias).
3. Trainer.log -> also appends numeric metrics to <run>/hb_metrics.jsonl (report_to is None here).
4. Probe mode (HB_PROBE_OUT set): checkpoint saving is a no-op and step times are recorded.
6. FinetuneTrainer.save_checkpoint: after each save, older ckpt_* keep their weights but their optimizer /
   scheduler / RNG / scaler files are deleted (resume state only for the latest checkpoint).
"""
from __future__ import annotations

import json
import math
import os
import time

import torch

# dimension-dependent tensors of ActionTransformerModel (src/psi/models/psi0.py:815-870, 632-634)
ALLOWED_NEW = {
    "action_proj_in.ac_proj.0.weight",   # Linear(Da, Da)
    "action_proj_in.ac_proj.0.bias",
    "action_proj_in.ac_proj.2.weight",   # Linear(Da, hidden)   (bias is hidden-sized -> loaded)
    "action_proj_out.linear.weight",     # Linear(hidden, Da)
    "action_proj_out.linear.bias",
    "obs_proj._obs_proc.1.weight",       # Linear(odim, hidden) (bias is hidden-sized -> loaded)
    "action_proj_in.dec_pos",            # (Tp, hidden): only if the horizon differs from 30
}
NATIVE_A, NATIVE_S, NATIVE_H = 80, 45, 30   # postpre.sonic1.1 action_header.safetensors (verified header)


def expected_new(action_dim: int, odim: int, horizon: int) -> set:
    new = set()
    if action_dim != NATIVE_A:
        new |= {"action_proj_in.ac_proj.0.weight", "action_proj_in.ac_proj.0.bias",
                "action_proj_in.ac_proj.2.weight", "action_proj_out.linear.weight", "action_proj_out.linear.bias"}
    if odim != NATIVE_S:
        new.add("obs_proj._obs_proc.1.weight")
    if horizon != NATIVE_H:
        new.add("action_proj_in.dec_pos")
    return new


PROBE: dict = {"times": []}


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def load_action_header_redim(header: torch.nn.Module, ckpt_dir: str, reconcile_fn, expected: set) -> dict:
    """Load action_header.safetensors into `header` keeping every shape-compatible tensor; the set of
    newly initialized keys must equal `expected` (= expected_new(Da, odim, Tp))."""
    from safetensors.torch import load_file

    path = os.path.join(ckpt_dir, "action_header.safetensors")
    sd = load_file(path)
    model_sd = header.state_dict()
    filtered = reconcile_fn(sd)  # native: same-shape keys (+ adaLN-zero -> continuous conversion)
    missing, unexpected = header.load_state_dict(filtered, strict=False)
    new_keys = sorted(set(model_sd) - set(filtered))
    report = {
        "checkpoint": path,
        "checkpoint_keys": len(sd), "model_keys": len(model_sd), "loaded_keys": len(filtered),
        "newly_initialized": [
            {"key": k, "model_shape": list(model_sd[k].shape),
             "checkpoint_shape": list(sd[k].shape) if k in sd else None,
             "init": "PyTorch default (module constructor), not zero-padded"}
            for k in new_keys],
        "checkpoint_keys_not_loaded": sorted(set(sd) - set(filtered)),
        "missing_after_load": sorted(missing), "unexpected": sorted(unexpected),
    }
    report["expected_new"] = sorted(expected)
    if set(new_keys) != set(expected) or not set(expected) <= ALLOWED_NEW:
        raise RuntimeError(f"recipe mismatch: newly initialized {new_keys} != expected {sorted(expected)}")
    after = header.state_dict()
    for k in list(filtered)[:: max(1, len(filtered) // 8)]:  # numeric spot check of loaded tensors
        if not torch.equal(after[k].detach().cpu().float(), filtered[k].float()):
            raise RuntimeError(f"loaded tensor {k} differs from checkpoint")
    report["new_param_numel"] = int(sum(model_sd[k].numel() for k in new_keys))
    return report


def apply() -> None:
    from psi.models.psi0 import Psi0Model
    from psi.trainers import finetune as F
    from psi.trainers.trainer import Trainer

    FT = F.FinetuneTrainer
    ow = F.overwatch

    # ---- 2. loss_w exact-sum guard -------------------------------------------------------------
    orig_init = FT.__init__

    def hb_init(self, cfg, device):
        D = cfg.model.action_dim
        dev = torch.device(device)
        exact = D == 7 or bool(torch.tensor([1.0 / D] * D, dtype=torch.float32).to(dev).sum() == 1.0)
        if exact:
            return orig_init(self, cfg, device)
        cfg.model.action_dim = 64
        try:
            orig_init(self, cfg, device)
        finally:
            cfg.model.action_dim = D
        self.ac_dim = D
        self.loss_w = torch.full((D,), 1.0 / D, dtype=torch.float32, device=self.device)
        ow.warning(f"[hb] fp32 sum of [1/{D}]*{D} != 1.0 on {dev}; native init ran with Da=64 for the assert, "
                   f"loss_w rebuilt as uniform 1/{D} (identical weights), ac_dim={D}")

    FT.__init__ = hb_init

    # ---- 1. re-dimensioned warm start ------------------------------------------------------------
    def hb_init_models(self):
        vlm_model = self.init_qwen3vl_models()
        self.model = Psi0Model(model_cfg=self.model_cfg, vlm_model=vlm_model)
        ckpt = self.model_cfg.pretrained_action_header_path
        if ckpt is None:
            raise RuntimeError("--model.pretrained-action-header-path is required (SONIC v1.1 init dir)")
        mc = self.model_cfg
        report = load_action_header_redim(self.model.action_header, ckpt, self._reconcile_action_header_state_dict,
                                          expected_new(mc.action_dim, mc.odim, mc.action_chunk_size))
        report.update(action_dim=self.model_cfg.action_dim, odim=self.model_cfg.odim,
                      horizon=self.model_cfg.action_chunk_size, vlm=self.model_cfg.model_name_or_path)
        ow.info(f"[hb] action header: loaded {report['loaded_keys']}/{report['model_keys']} tensors from {ckpt}")
        for e in report["newly_initialized"]:
            ow.warning(f"[hb] NEWLY INITIALIZED {e['key']}: model {e['model_shape']} vs checkpoint {e['checkpoint_shape']}")
        if _rank() == 0:
            os.makedirs(self.project_dir, exist_ok=True)
            with open(os.path.join(self.project_dir, "hb_init_report.json"), "w") as f:
                json.dump(report, f, indent=2)
        if self.train_cfg.data_parallel == "deepspeed":  # native finetune.py:358-361
            self.model.config = vlm_model.config
            self.model.config.hidden_size = vlm_model.config.text_config.hidden_size
        self.apply_vlm_trainability()
        n = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        ow.info(f"Model has {n:,} trainable parameters")

    FT.init_models = hb_init_models

    # ---- 3. metrics jsonl ---------------------------------------------------------------------
    orig_log = Trainer.log
    every = int(os.environ.get("HB_LOG_EVERY", "10"))

    def hb_log(self, metrics, start_time=None):
        orig_log(self, metrics, start_time)
        if not self.accelerator.is_main_process:
            return
        step = int(getattr(self, "global_step", 0))
        if step % every:
            return
        rec = {"step": step, "time": time.time()}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.item()
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                rec[k] = float(v)
        with open(os.path.join(self.project_dir, "hb_metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")

    Trainer.log = hb_log

    # ---- 5. epoch -> DataLoader workers (per-trajectory bias); train.py calls next_epoch(epoch) before
    #         iterating the (persistent-worker) dataloader for that epoch, also after a resume.
    import hb_psi0_config
    orig_next_epoch = Trainer.next_epoch

    def hb_next_epoch(self, epoch):
        hb_psi0_config.set_epoch(epoch)
        return orig_next_epoch(self, epoch)

    Trainer.next_epoch = hb_next_epoch

    # ---- 6. resume state only for the latest checkpoint (user rule 2026-11-10): after each save, older ckpt_* keep
    #         their model weights but lose accelerate's optimizer/scheduler/RNG/scaler state.
    orig_save = FT.save_checkpoint

    def hb_save_checkpoint(self, global_step):
        out = orig_save(self, global_step)
        if self.accelerator.is_main_process:
            import glob
            import re
            import shutil
            root = os.path.join(self.project_dir, "checkpoints")
            for d in glob.glob(os.path.join(root, "ckpt_*")):
                if os.path.basename(d) == f"ckpt_{global_step}":
                    continue
                for f in os.listdir(d):
                    if re.match(r"(optimizer|scheduler|random_states|scaler)", f):
                        p = os.path.join(d, f)
                        (shutil.rmtree if os.path.isdir(p) else os.remove)(p)
                        print(f"[hb] dropped resume state {p}", flush=True)
        self.accelerator.wait_for_everyone()
        return out

    FT.save_checkpoint = hb_save_checkpoint

    # ---- 4. probe ----------------------------------------------------------------------------
    if os.environ.get("HB_PROBE_OUT"):
        FT.save_checkpoint = lambda self, global_step: None
        orig_step = FT.training_step

        def timed_step(self, batch):
            out = orig_step(self, batch)
            torch.cuda.synchronize()
            PROBE["times"].append(time.perf_counter())
            return out

        FT.training_step = timed_step
