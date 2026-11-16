"""DiT4DiT adapter (Mondo-Robotics/DiT4DiT @ 66a6f3a, HB adaptation in hb_dit4dit/).

ckpt = .../dit4dit-runs/<chain>/<task>/run/checkpoints/<...>.pt ; the framework's own from_pretrained reads the run's
config + dataset_statistics.json next to it (read_mode_config), builds the model and loads the weights strictly.
Inputs exactly as hb_d4d_data builds frame 0 of a training sample (augmentation off):
  image  [3, 224, 224*V] in [0, 1]: every camera 480x640 -> 224x224 bilinear (align_corners=False, no antialias),
         cameras side by side in the layout order; only the conditioning frame (no future frames at inference);
  state  [1, S_enc]: hb_d4d_data.encode_state (native sin/cos of joint angles + min_max of the rest) of the raw 44/78;
  lang   the training instruction.
Output: normalized actions [16, A] -> inverse min_max with the run's hb_action_normalization statistics.
NOT smoke-tested yet (no DiT4DiT checkpoint existed when this was written, 2026-11-14).
"""
import json
import sys
from pathlib import Path

import numpy as np

import hb_eval_common as E


def _find_stats(ckpt):
    for d in [Path(ckpt).parent, *Path(ckpt).parents]:
        p = d / "dataset_statistics.json"
        if p.exists():
            return json.loads(p.read_text())
    raise FileNotFoundError(f"dataset_statistics.json not found above {ckpt}")


class Adapter:
    def __init__(self, ckpt, task, split, meta, device="cuda:0", steps=None):
        import torch
        self.torch = torch
        for p in (str(E.G / "hb_dit4dit"), str(E.G / "HumanWAM/third_party/DiT4DiT"), str(E.G / "hb_groot")):
            if p not in sys.path:
                sys.path.insert(0, p)
        import hb_d4d_data as D
        self.D = D
        self.layout = D.layout_of(task)
        st = _find_stats(ckpt)
        blk = next(iter(st.values()))
        self.amin = np.asarray(blk["hb_action_normalization"]["min"], np.float64)
        self.amax = np.asarray(blk["hb_action_normalization"]["max"], np.float64)
        self.smin = np.asarray(blk["hb_state_encoding"]["min"], np.float64)
        self.smax = np.asarray(blk["hb_state_encoding"]["max"], np.float64)
        self.cams = list(blk["hb_layout"]["cameras"])
        miss = [c for c in self.cams if c not in meta["cams"]]
        if miss:
            raise KeyError(f"cameras {miss} not in the cache {meta['cams']}")
        try:  # same model patches as training, if the HB adaptation defines any for model construction
            import hb_patches
            if hasattr(hb_patches, "apply"):
                hb_patches.apply()
        except ImportError:
            pass
        import DiT4DiT.model.framework.base_framework as BF
        cls = next(v for v in vars(BF).values()
                   if isinstance(v, type) and v.__module__ == BF.__name__ and hasattr(v, "from_pretrained"))
        self.model = cls.from_pretrained(str(ckpt))
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        if steps:
            am = self.model.action_model
            for name in ("num_inference_timesteps", "num_inference_steps"):
                if hasattr(am, name):
                    setattr(am, name, int(steps))
        self.steps = steps or "native"
        view = D.task_dir(task)
        self.prompts = {}
        try:
            m = D.load_meta(task)
            if isinstance(m.get("prompt"), str):
                self.prompts = {"0": m["prompt"]}
        except FileNotFoundError:
            pass
        self.view = str(view)
        self.ckpt = str(ckpt)

    def info(self):
        return {"model": "dit4dit", "ckpt": self.ckpt, "cams": self.cams, "denoising_steps": self.steps,
                "state_encoded_dim": int(self.layout.state_dim), "view": self.view}

    def _img(self, obs):
        torch = self.torch
        F = torch.nn.functional
        views = []
        for c in self.cams:
            x = torch.from_numpy(np.ascontiguousarray(obs["images"][c])).permute(2, 0, 1)[None].float() / 255.0
            views.append(F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)[0])
        return torch.cat(views, dim=-1).clamp(0, 1)  # [3, 224, 224*V]

    def predict(self, obs):
        s = self.D.encode_state(np.asarray(obs["state"], np.float64), self.layout, self.smin, self.smax)
        ex = {"image": [self._img(obs).to(self.device)], "lang": self.prompts.get("0", obs["prompt"]),
              "state": s[None].astype(np.float32)}
        out = self.model.predict_action([ex])["normalized_actions"][0]  # [16, A]
        return self.D.unminmax(np.asarray(out, np.float64), self.amin, self.amax)
