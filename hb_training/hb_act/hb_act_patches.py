"""Runtime patches on the pinned Psi0 ACT trainer (psi.trainers.act_g1.ActG1Trainer); no upstream file modified.
1. Trainer.next_epoch -> hb_psi0_config.set_epoch(epoch): per-trajectory joint bias in the DataLoader workers.
2. Trainer.log -> numeric metrics also to <run>/hb_metrics.jsonl.
3. ActG1Trainer.save_checkpoint: older ckpt_* keep the model weights, lose optimizer/scheduler/RNG/scaler state.
4. Probe mode (HB_PROBE_OUT): no checkpoints, step times recorded.
"""
import json
import math
import os
import time

import torch

PROBE: dict = {"times": []}


def apply() -> None:
    import hb_psi0_config
    from psi.trainers import act_g1 as A
    from psi.trainers.trainer import Trainer

    AT = A.ActG1Trainer
    orig_next_epoch = Trainer.next_epoch

    def hb_next_epoch(self, epoch):
        hb_psi0_config.set_epoch(epoch)
        return orig_next_epoch(self, epoch)

    Trainer.next_epoch = hb_next_epoch

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

    orig_save = AT.save_checkpoint

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

    AT.save_checkpoint = hb_save_checkpoint

    if os.environ.get("HB_PROBE_OUT"):
        AT.save_checkpoint = lambda self, global_step: None
        orig_step = AT.training_step

        def timed_step(self, batch):
            out = orig_step(self, batch)
            torch.cuda.synchronize()
            PROBE["times"].append(time.perf_counter())
            return out

        AT.training_step = timed_step
