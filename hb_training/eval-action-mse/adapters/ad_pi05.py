"""pi0.5 (OpenPI PyTorch) adapter: policy_config.create_trained_policy + Policy.infer, the deployment path.

ckpt = .../pi05-runs/<cfg>/<task>/run/ckpt/pi05_hb_<layout>_<task>/run/<step>  (model.safetensors + assets/<id>/norm_stats.json)
Inputs as HBInputs expects at inference: observation/<cam> raw 480x640 uint8 (ResizeImages does resize_with_pad 224),
observation/state raw 44 / 78, prompt. Output: Unnormalize + HBOutputs -> raw (16, 76 | 75).
opt cams=head,chest  -> evaluate a checkpoint trained with the older 2-camera H2 layout (third slot masked).
"""
import dataclasses
import inspect
import json
import sys

import numpy as np

import hb_eval_common as E


class Adapter:
    def __init__(self, ckpt, task, split, meta, device="cuda:0", steps=None, cams=None):
        sys.path.insert(0, str(E.G / "hb_groot"))
        sys.path.insert(0, str(E.G / "hb_pi05"))
        import hb_g1_data as D
        if cams:
            lay = D.TASKS[task].layout
            D.LAYOUTS[lay] = dataclasses.replace(D.LAYOUTS[lay], cameras=tuple(cams.split(",")))
        import hb_g1_config  # noqa: F401  registers pi05_hb_<layout>_<task>
        from openpi.policies import policy_config
        from openpi.training import config as _config
        self.cfg = _config.get_config(D.config_name(task))
        kw = {}
        if "pytorch_device" in inspect.signature(policy_config.create_trained_policy).parameters:
            kw["pytorch_device"] = device
        if steps:
            kw["sample_kwargs"] = {"num_steps": int(steps)}
        self.policy = policy_config.create_trained_policy(self.cfg, str(ckpt), **kw)
        self.cams = D.LAYOUTS[D.TASKS[task].layout].cameras
        self.steps = int(steps) if steps else "native(Policy default)"
        self.ckpt = str(ckpt)
        miss = [c for c in self.cams if c not in meta["cams"]]
        if miss:
            raise KeyError(f"cameras {miss} not in the cache {meta['cams']}")
        # training prompt = the view's meta/tasks.jsonl (hb_prepare copies it); fall back to the recording's
        view = D.task_dir(task)
        self.prompts = {}
        if (view / "meta.json").exists():
            m = json.loads((view / "meta.json").read_text())
            if isinstance(m.get("prompts"), dict):
                self.prompts = {str(k): v for k, v in m["prompts"].items()}
            elif isinstance(m.get("prompt"), str):
                self.prompts = {"0": m["prompt"]}

    def info(self):
        return {"model": "pi05", "ckpt": self.ckpt, "config": self.cfg.name, "denoising_steps": self.steps,
                "cams": list(self.cams), "max_token_len": self.cfg.model.max_token_len}

    def predict(self, obs):
        o = {f"observation/{c}": obs["images"][c] for c in self.cams}
        o["observation/state"] = np.asarray(obs["state"], dtype=np.float32)
        o["prompt"] = self.prompts.get("0", obs["prompt"]) if len(self.prompts) == 1 else obs["prompt"]
        return np.asarray(self.policy.infer(o)["actions"])  # (16, Da) raw
