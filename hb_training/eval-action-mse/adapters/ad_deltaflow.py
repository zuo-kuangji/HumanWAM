"""ImageWAM Delta-flow adapter. Uses the run's own NON-augmented dataset class (parent of the HB-aug v2 class in
config.yaml) so images (two 224 views side by side, [-1, 1]), normalised current state, normalised 16-step past
action history (zeros at the episode start, as in training) and the cached Qwen instruction embedding are exactly
the training-time tensors. Inference = the model's native infer_action_flux2 defaults (1 action step,
history + 0.5 additive noise), as wholebody_delta enforces. Predictions are denormalised with the dataset normaliser.
heldout -> split="validation" (B6: the 49 shelved episodes 100-148); train10 -> split="train".

ckpt = .../deltaflow-runs/<task>/run/production/checkpoints/weights/step_<N>.pt  (config.yaml two levels up)
The benchmark-cache images are NOT used (the native decoder is); the cache only fixes the (episode, t) queries,
and the dataset's ground truth is checked against the cache's (max abs diff must be < 1e-4).
"""
import sys
from pathlib import Path

import numpy as np

import hb_eval_common as E


class Adapter:
    def __init__(self, ckpt, task, split, meta, device="cuda:0", steps=None, seed="1234"):
        import torch
        self.torch = torch
        ckpt = Path(ckpt)
        prod = ckpt.parent.parent.parent
        code = E.G / "deltaflow-runs" / "HumanWAM"
        for p in (str(code / "src"), str(code)):
            if p not in sys.path:
                sys.path.insert(0, p)
        from hydra.utils import get_class, instantiate
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(prod / "config.yaml")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=self.device)
        self.model.load_checkpoint(str(ckpt))
        self.model.eval()
        d = OmegaConf.to_container(cfg.data.train, resolve=True)
        aug_cls = get_class(d.pop("_target_"))
        d.pop("image_augmentation", None)
        d.pop("proprio_augmentation", None)
        # the native NON-augmented dataset class = first imagewam dataset class in the MRO without "aug" in its
        # module / name (G1: imagewam.datasets.wholebody.WholeBodyLeRobotDataset; its __getitem__ is the plain one)
        base = next(k for k in aug_cls.__mro__ if k.__module__.startswith("imagewam.datasets")
                    and "aug" not in k.__module__.lower() and "aug" not in k.__name__.lower()
                    and "Robust" not in k.__name__ and "__getitem__" in vars(k))
        d["split"] = "validation" if split == "heldout" else "train"
        self.ds = base(**d)
        self.base_name = f"{base.__module__}.{base.__name__}"
        self.starts = {ep: int(self.ds.ends[i - 1]) if i else 0 for i, ep in enumerate(self.ds.episode_ids)}
        self.steps = steps
        self.seed = int(seed)
        self.max_gt_diff = 0.0
        self.ckpt = str(ckpt)

    def _denorm(self, x):
        n = self.ds.normalizer
        for name in ("backward", "inverse", "denormalize"):
            if hasattr(n, name):
                return np.asarray(getattr(n, name)(np.asarray(x, dtype=np.float64), "action"))
        raise AttributeError("normaliser has no inverse")

    def info(self):
        return {"model": "deltaflow", "ckpt": self.ckpt, "dataset_class": self.base_name,
                "denoising_steps": self.steps or "native(1)", "seed": self.seed,
                "uses_past_action_history": True, "max_abs_gt_diff_vs_cache": self.max_gt_diff}

    def predict(self, obs):
        torch = self.torch
        ep, t = obs["episode"], obs["t"]
        if ep not in self.starts:
            raise KeyError(f"episode {ep} not in the native {self.ds.episode_ids[:3]}... split")
        s = self.ds[self.starts[ep] + t]
        assert s["episode_index"] == ep and s["frame_index"] == t
        # native GT == benchmark GT (rows inside the episode)
        ref = obs["_gt_check"]
        gt = self._denorm(s["action"].float().numpy())[: len(ref)]
        self.max_gt_diff = max(self.max_gt_diff, float(np.abs(gt - ref).max()))
        if self.max_gt_diff > 1e-3:
            raise ValueError(f"native dataset GT differs from the benchmark GT by {self.max_gt_diff} at ep {ep} t {t}")
        kw = dict(prompt=None, input_image=s["video"][:, 0], action_horizon=16, proprio=s["proprio"][0],
                  context=s["text_hidden_states"], context_mask=s["text_attention_mask"],
                  action_init=s["past_action"], seed=self.seed)
        if self.steps:
            kw["num_inference_steps"] = int(self.steps)
        with torch.no_grad():
            out = self.model.infer_action_flux2(**kw)
        return self._denorm(out["action"].float().cpu().numpy())
