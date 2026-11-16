"""ACT adapter (Psi0 baselines/act = LeRobot ACT). Mirrors src/act/deploy/act_g1_serve_simple.py:
images -> ToImage/ToDtype -> native resize (256x480) + center crop 224 -> ImageNet normalise, all training cameras;
state bounds-normalised; ACTPolicy.predict_action (VAE latent = 0 at inference); actions denormalised.

ckpt = .../act-runs-*/<task>/train/act-g1/<run>/checkpoints/ckpt_<step>
"""
import numpy as np

import hb_eval_common as E
from ad_psi0 import image_cams, load_launch, resolve_run


class Adapter:
    def __init__(self, ckpt, task, split, meta, device="cuda:0", steps=None):
        import torch
        from torchvision.transforms import v2
        self.torch = torch
        self.run_dir, step = resolve_run(ckpt)
        lc = load_launch(self.run_dir, "hb_act_config", "act")
        from act.models.act import ACTConfig, ACTPolicy
        from safetensors.torch import load_file
        mc = lc.model
        cfg = ACTConfig(n_obs_steps=mc.n_obs_steps, chunk_size=mc.chunk_size, n_action_steps=mc.n_action_steps,
                        action_dim=mc.action_dim, state_dim=mc.state_dim, dim_model=mc.dim_model, n_heads=mc.n_heads,
                        dim_feedforward=mc.dim_feedforward, feedforward_activation=mc.feedforward_activation,
                        n_encoder_layers=mc.n_encoder_layers, n_decoder_layers=mc.n_decoder_layers,
                        pre_norm=mc.pre_norm, dropout=mc.dropout, use_vae=mc.use_vae, latent_dim=mc.latent_dim,
                        n_vae_encoder_layers=mc.n_vae_encoder_layers, kl_weight=mc.kl_weight,
                        temporal_ensemble_coeff=mc.temporal_ensemble_coeff)
        self.model = ACTPolicy(config=cfg, dataset_stats=None)
        self.model.load_state_dict(load_file(str(self.run_dir / "checkpoints" / f"ckpt_{step}" / "model.safetensors")))
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model.to(self.device).eval()
        self.maxmin = lc.data.transform.field
        mt = lc.data.transform.model
        self.tf = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True), mt.resize(), mt.center_crop(),
                              mt.normalize()])
        self.cams = image_cams(lc)
        miss = [c for c in self.cams if c not in meta["cams"]]
        if miss:
            raise KeyError(f"cameras {miss} not in the cache {meta['cams']}")
        self.n_action_steps = int(mc.n_action_steps)
        self.ckpt = str(ckpt)

    def info(self):
        return {"model": "act", "ckpt": self.ckpt, "run_dir": str(self.run_dir), "cams": self.cams,
                "chunk": self.n_action_steps, "denoising_steps": None}

    def predict(self, obs):
        from PIL import Image
        torch = self.torch
        x = torch.stack([self.tf(Image.fromarray(obs["images"][c])) for c in self.cams])[None].to(self.device)
        s = self.maxmin.normalize_state_func(np.asarray(obs["state"], dtype=np.float32))
        st = torch.from_numpy(np.asarray(s, dtype=np.float32)[None]).to(self.device)
        with torch.inference_mode():
            a = self.model.predict_action({"observation.images": x, "observation.state": st})
        a = a[0] if a.ndim == 3 else a
        a = a[: self.n_action_steps].float().cpu().numpy()
        return np.asarray(self.maxmin.denormalize(a))
