"""FastWAM adapter (2026-11-16). Deployment path of FastWAM's own policy (experiments/robotwin/fastwam_policy/
deploy_policy.py, WorldActionRobotWinPolicy._infer_action_chunk: model.infer_action, the action-only path = first-frame
video KV cache + action-expert denoising, no future video) fed exactly like our training dataset (RobotVideoDataset +
FastWAMProcessor of the run's config.yaml, no HB augmentation):
  * cameras in shape_meta order (g1 ego_view + chest, h2 head + chest): uint8 [1,C,H,W] -> processor val transforms
    (ToTensor /255 + Resize 224x224), concatenated horizontally (224 x 448), dataset ResizeSmallestSide/CenterCrop
    (no-op at 224 x 448), Normalize(0.5, 0.5) -> [-1, 1];
  * state: processor.action_state_transform -> normalizer.forward (min/max of the run's
    dataset_stats.json), the proprio of the first frame;
  * text: the run's T5 cache for DEFAULT_PROMPT.format(task=instruction) with context[~mask] = 0 and mask = ones, as in
    training (the text encoder is not loaded in training either; deploy_policy encodes the same prompt with T5 online);
  * infer_action(action_horizon = num_frames - 1 = 16, num_inference_steps = config eval_num_inference_steps (10),
    text_cfg_scale 1, seed 42 = native trainer eval), denormalised with normalizer.normalizers["action"][key].backward.
ckpt = .../fastwam/runs/<task>/out/checkpoints/weights/step_<N>.pt (config.yaml + dataset_stats.json in .../out).
Weights: model built from the config (Wan2.2 + ActionDiT init), then the checkpoint's mot + proprio_encoder are
loaded STRICTLY (FastWAM.load_checkpoint uses strict=False for mot; a missing key here is an error).
"""
import os
import sys
from pathlib import Path

import numpy as np

import hb_eval_common as E

FW = E.G / "fastwam"


def setup_env():
    os.environ.setdefault("FASTWAM_SRC", str(FW / "FastWAM"))
    os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(FW / "models"))
    os.environ.setdefault("DIFFSYNTH_SKIP_DOWNLOAD", "true")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    src = str(Path(os.environ["FASTWAM_SRC"]) / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


class Inputs:
    """model-agnostic cache query -> FastWAM inputs (shared by the post-hoc adapter and the in-training eval)"""

    def __init__(self, cfg, stats_path):
        setup_env()
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from fastwam.datasets.dataset_utils import CenterCrop, Normalize, ResizeSmallestSideAspectPreserving
        from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        d = cfg.data.train
        self.DEFAULT_PROMPT = DEFAULT_PROMPT
        self.proc = instantiate(d.processor).eval()
        self.proc.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats_path)))
        self.cams = [m["key"] for m in self.proc.shape_meta["images"]]
        if d.concat_multi_camera != "horizontal":
            raise ValueError(f"concat_multi_camera {d.concat_multi_camera} not supported")
        vs = list(d.video_size)
        self.resize = ResizeSmallestSideAspectPreserving(args={"img_w": vs[1], "img_h": vs[0]})
        self.crop = CenterCrop(args={"img_w": vs[1], "img_h": vs[0]})
        self.norm = Normalize(args={"mean": 0.5, "std": 0.5})
        self.cache_dir = d.text_embedding_cache_dir
        self.context_len = int(d.context_len)
        self.horizon = int(d.num_frames) - 1
        self.skey = self.proc.shape_meta["state"][0]["key"]
        self.akey = self.proc.shape_meta["action"][0]["key"]
        if len(self.proc.shape_meta["state"]) != 1 or len(self.proc.shape_meta["action"]) != 1:
            raise ValueError("expected one merged state key and one merged action key")
        self._ctx = {}
        self.OmegaConf = OmegaConf

    def image(self, images):
        import torch
        per = []
        for c in self.cams:
            x = torch.from_numpy(np.ascontiguousarray(images[c])).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W] uint8
            for tr in self.proc.val_transforms:
                x = tr(x)
            per.append(x)
        v = torch.cat(per, dim=-1)  # [1, C, 224, 448]
        v = self.norm(self.crop(self.resize(v)))
        return v  # [1, 3, H, W] in [-1, 1], float32

    def proprio(self, state):
        import torch
        b = {"state": {self.skey: torch.as_tensor(np.asarray(state, dtype=np.float32)).unsqueeze(0)}}
        b = self.proc.action_state_transform(b)
        b = self.proc.normalizer.forward(b)  # = deploy_policy._normalize_state (one merged key: merger is identity)
        return b["state"][self.skey]  # [1, D]

    def context(self, instruction):
        import hashlib
        import torch
        prompt = self.DEFAULT_PROMPT.format(task=instruction)
        if prompt not in self._ctx:
            h = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            p = os.path.join(self.cache_dir, f"{h}.t5_len{self.context_len}.wan22ti2v5b.pt")
            if not os.path.exists(p):
                raise FileNotFoundError(f"no T5 cache for the instruction ({p}): {instruction[:80]!r}")
            obj = torch.load(p, map_location="cpu")
            ctx, mask = obj["context"], obj["mask"]
            ctx = ctx.clone()
            ctx[~mask.bool()] = 0.0
            self._ctx[prompt] = (ctx, torch.ones_like(mask.bool()))
        return self._ctx[prompt]

    def denorm(self, action):
        import torch
        a = action
        if a.ndim == 2:
            a = a.unsqueeze(0)
        n = self.proc.normalizer.normalizers["action"][self.akey]
        return n.backward(a.to(dtype=torch.float32, device="cpu")).numpy()[0]


def infer(model, inp, obs, steps, seed=42):
    import torch
    img = inp.image(obs["images"]).to(device=model.device, dtype=model.torch_dtype)
    ctx, mask = inp.context(obs["prompt"])
    with torch.no_grad():
        out = model.infer_action(prompt=None, input_image=img, action_horizon=inp.horizon, proprio=inp.proprio(obs["state"]),
                                 context=ctx, context_mask=mask, text_cfg_scale=1.0, num_inference_steps=steps, seed=seed,
                                 rand_device="cpu", tiled=False)
    return inp.denorm(out["action"])


class Adapter:
    def __init__(self, ckpt, task, split, meta, device="cuda:0", steps=None, seed="42"):
        setup_env()
        import torch
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        self.torch = torch
        ckpt = Path(ckpt)
        out = ckpt.parent.parent.parent
        self.cfg = OmegaConf.load(out / "config.yaml")
        self.inp = Inputs(self.cfg, out / "dataset_stats.json")
        miss = [c for c in self.inp.cams if c not in meta["cams"]]
        if miss:
            raise KeyError(f"cameras {miss} not in the cache {meta['cams']}")
        self.device = device if torch.cuda.is_available() else "cpu"
        mp = str(self.cfg.get("mixed_precision", "bf16")).lower()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}[mp]
        self.model = instantiate(self.cfg.model, model_dtype=dtype, device=self.device)
        payload = torch.load(str(ckpt), map_location="cpu")
        miss_k, unexp = self.model.mot.load_state_dict(payload["mot"], strict=False)
        if miss_k or unexp:
            raise RuntimeError(f"checkpoint mot keys: missing {list(miss_k)[:5]} unexpected {list(unexp)[:5]}")
        if self.model.proprio_encoder is not None:
            self.model.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
        self.ckpt_step = payload.get("step")
        del payload
        self.model = self.model.to(self.device).eval()
        self.native_steps = int(self.cfg.eval_num_inference_steps)
        self.steps = int(steps or self.native_steps)
        self.seed = int(seed)
        self.ckpt = str(ckpt)

    def info(self):
        return {"model": "fastwam", "ckpt": self.ckpt, "ckpt_step": self.ckpt_step, "denoising_steps": self.steps,
                "native_steps_in_config": self.native_steps, "cams": self.inp.cams, "chunk": self.inp.horizon,
                "seed": self.seed, "path": "FastWAM.infer_action (action-only, deploy_policy)"}

    def predict(self, obs):
        return infer(self.model, self.inp, obs, self.steps, self.seed)
