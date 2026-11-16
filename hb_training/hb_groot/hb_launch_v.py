"""2026-11-15 user: GR00T variant launcher = hb_launch.py (HB aug v2 + A20 proprio + processor patch, then the pinned
upstream launch_finetune.py) plus, from environment variables, before the run starts:
  HB_GROOT_MODEL_OVERRIDES  JSON {"attn_dropout": 0.0, "diffusion_model_cfg.dropout": 0.0, "num_inference_timesteps": 16,
                            ...}: set on config.model (dotted keys go into nested dicts / attributes)
  HB_GROOT_LR_GROUPS        JSON {"language": 1e-6, "vision": 1e-5, "projector": 1e-4, "rest": 1e-4}: layer-wise learning
                            rates (Psi-Zero style) through Gr00tTrainer.create_optimizer:
                              language  = backbone ... language_model.* and lm_head
                              vision    = backbone ... visual.* except the vision->language mergers
                              projector = visual.merger.* and visual.deepstack_merger_list.*
                              rest      = action head (DiT, VL self-attention, VLLN, encoders / decoders)
                            weight decay as HF Trainer (no decay on biases / norms); cosine + warmup scale every group.
Prints HB_LAUNCH_V lines (overrides applied, parameter count and lr per group) on rank 0 for verification."""
import json
import os
import runpy
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
RANK0 = os.environ.get("RANK", "0") == "0"


def say(*a):
    if RANK0:
        print("HB_LAUNCH_V", *a, flush=True)


OV = json.loads(os.environ.get("HB_GROOT_MODEL_OVERRIDES", "{}") or "{}")
LRG = json.loads(os.environ.get("HB_GROOT_LR_GROUPS", "{}") or "{}")

import gr00t.experiment.experiment as E  # noqa: E402

_run = E.run


def _set(obj, dotted, value):
    parts = dotted.split(".")
    for p in parts[:-1]:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, dict):
        if last not in obj:
            raise KeyError(f"model override {dotted}: no key {last}")
        old = obj[last]
        obj[last] = value
    else:
        if not hasattr(obj, last):
            raise AttributeError(f"model override {dotted}: no attribute {last}")
        old = getattr(obj, last)
        setattr(obj, last, value)
    say(f"override model.{dotted}: {old!r} -> {value!r}")


def run(config):
    for k, v in OV.items():
        _set(config.model, k, v)
    return _run(config)


E.run = run

if LRG:
    from transformers import Trainer
    from gr00t.experiment.trainer import Gr00tTrainer

    def group_of(name):
        if ".language_model." in name or "lm_head" in name:
            return "language"
        if ".visual." in name:
            return "projector" if (".visual.merger." in name or ".visual.deepstack_merger_list." in name) else "vision"
        return "rest"

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        model = self.model
        decay = set(self.get_decay_parameter_names(model))
        groups, stats = {}, {}
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            g = group_of(n)
            lr = float(LRG.get(g, LRG.get("rest", self.args.learning_rate)))
            wd = self.args.weight_decay if n in decay else 0.0
            groups.setdefault((g, lr, wd), []).append(p)
            s = stats.setdefault(g, [lr, 0, 0])
            s[1] += 1
            s[2] += p.numel()
        opt_cls, opt_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args, model)
        opt_kwargs.pop("lr", None)
        self.optimizer = opt_cls([{"params": ps, "lr": lr, "weight_decay": wd} for (g, lr, wd), ps in groups.items()],
                                 lr=self.args.learning_rate, **opt_kwargs)
        for g, (lr, n_t, n_p) in sorted(stats.items()):
            say(f"lr group {g}: lr {lr:g}, {n_t} tensors, {n_p / 1e6:.1f} M params")
        return self.optimizer

    Gr00tTrainer.create_optimizer = create_optimizer
    say("layer-wise lr groups installed", json.dumps(LRG))

say("model overrides", json.dumps(OV))
runpy.run_path(str(H / "hb_launch.py"), run_name="__main__")
