"""Psi-Zero adapter (Psi0 framework). Mirrors psi/deploy/serve_psi0_sonic.py (no_aug path) except that ALL training
cameras are passed (the sonic server forwards only cam0; HB Psi-Zero is trained on cam0 + chest [+ wrist]).

ckpt = .../psi0-runs/<task>/<cfg>/train/finetune/<run>/checkpoints/ckpt_<step>
    or .../psi0-runs/<task>/<cfg>/preserved/ckpt_<step>   (run dir found next to it; a /tmp run dir with links is used)
Normalisation: the run's SonicActionStateTransform (bounds) from run_config.json; actions denormalised with it.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import hb_eval_common as E

PSI = E.G / "HumanWAM/third_party/Psi0"


def resolve_run(ckpt):
    """-> (run_dir with checkpoints/ckpt_<step>, step)."""
    ckpt = Path(ckpt)
    step = ckpt.name.split("_", 1)[1]
    if ckpt.parent.name == "checkpoints":
        return ckpt.parent.parent, step
    runs = sorted(p.parent for p in ckpt.parent.parent.glob("train/*/*/argv.txt"))
    if len(runs) != 1:
        raise FileNotFoundError(f"cannot find the run dir for {ckpt}: {runs}")
    src = runs[0]
    tmp = Path(tempfile.mkdtemp(prefix="hb_eval_psi_"))
    for f in src.iterdir():
        if f.name != "checkpoints":
            os.symlink(f, tmp / f.name)
    (tmp / "checkpoints").mkdir()
    os.symlink(ckpt, tmp / "checkpoints" / ckpt.name)
    return tmp, step


def load_launch(run_dir, cfg_module, cfg_name):
    for p in (str(PSI / "src"), str(E.G / "hb_psi0"), str(E.G / "hb_act"), str(E.G / "hb_groot")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HB_AUG_DIR", str(E.G / "hb_groot"))
    mod = __import__(cfg_module)
    sys.modules[f"psi.config.train.{cfg_module}"] = mod
    from psi.utils import parse_args_to_tyro_config
    config_ = parse_args_to_tyro_config(Path(run_dir) / "argv.txt")
    conf = json.loads((Path(run_dir) / "run_config.json").read_text())
    try:
        from psi.utils import apply_legacy_model_config_defaults
        conf = apply_legacy_model_config_defaults(conf)
    except ImportError:
        pass
    return config_.model_validate(conf)


def image_cams(lc):
    return [k.split("observation.images.", 1)[-1] for k in lc.data.transform.repack.image_keys]


class Adapter:
    def __init__(self, ckpt, task, split, meta, device="cuda:0", steps=None):
        import torch
        self.torch = torch
        self.run_dir, step = resolve_run(ckpt)
        lc = load_launch(self.run_dir, "hb_psi0_config", "psi0")
        from psi.utils import seed_everything
        seed_everything(lc.seed or 42)
        import psi.models.psi0 as P0
        from psi.models.psi0 import Psi0Model
        # from_pretrained reads the Qwen3-VL config + processor from the hub id QWEN3VL_VARIANT; offline, point it at the
        # run's own local VLM dir (model_name_or_path, the dir training initialised from)
        mnp = getattr(lc.model, "model_name_or_path", None)
        P0.QWEN3VL_VARIANT = mnp if mnp and (Path(mnp) / "preprocessor_config.json").exists() \
            else str(E.G / "models/qwen3-vl-2b-instruct")
        self.vlm_dir = P0.QWEN3VL_VARIANT
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = Psi0Model.from_pretrained(self.run_dir, step, lc, device=self.device)
        self.model.to(self.device).eval()
        self.maxmin = lc.data.transform.field
        mt = lc.data.transform.model
        from torchvision.transforms import v2
        self.tf = v2.Compose([mt.resize(), mt.center_crop()])  # no_aug branch of Psi0ModelTransform
        self.cams = image_cams(lc)
        miss = [c for c in self.cams if c not in meta["cams"]]
        if miss:
            raise KeyError(f"cameras {miss} not in the cache {meta['cams']}")
        mc = lc.model
        self.native_steps = getattr(mc, "num_inference_steps", None)
        self.steps = int(steps or self.native_steps or 10)
        self.pooled = None
        if getattr(mc, "pooled_text_encoder", None) is not None:
            rel = getattr(mc, "pooled_cache_path", None) or "clip_pooled_cache.pt"
            path = rel if os.path.isabs(rel) else str(Path(self.run_dir) / rel)
            self.pooled = {k: v for k, v in torch.load(path, map_location="cpu").items()}
        self.Tp = int(mc.action_chunk_size)
        self.ckpt = str(ckpt)

    def info(self):
        return {"model": "psi0", "ckpt": self.ckpt, "run_dir": str(self.run_dir), "vlm_dir": self.vlm_dir,
                "denoising_steps": self.steps,
                "native_steps_in_config": self.native_steps, "cams": self.cams, "chunk": self.Tp,
                "note": "first 16 of the predicted chunk are scored"}

    def predict(self, obs):
        from PIL import Image
        torch = self.torch
        views = [self.tf(Image.fromarray(obs["images"][c])) for c in self.cams]
        s = self.maxmin.normalize_state_func(np.asarray(obs["state"], dtype=np.float32))
        states = torch.from_numpy(np.asarray(s, dtype=np.float32)[None, None]).to(self.device)
        instr = obs["prompt"].lower()
        kw = {}
        if self.pooled is not None:
            if instr not in self.pooled:
                raise KeyError(f"instruction not in the run's pooled CLIP cache: {instr!r} (have {list(self.pooled)[:3]})")
            kw["pooled_projections"] = self.pooled[instr].to(self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.model.predict_action(observations=[views], states=states, traj2ds=None, instructions=[instr],
                                          num_inference_steps=self.steps, **kw)[0].float().cpu().numpy()
        return np.asarray(self.maxmin.denormalize(a))
