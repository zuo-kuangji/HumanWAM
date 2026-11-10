"""CPU checks of the HB ACT pipeline for one task (no GPU): args -> tyro -> DynamicLaunchConfig (same parse as
scripts/train.py), real Psi0 view samples (2 cameras 224x224, raw state 44/78 without velocity, chunk 16), HB v2
augmentation on / native ColorJitter off, and one ACTPolicy forward (ResNet18 ImageNet weights from TORCH_HOME).
Usage: <psi0 python> verify_cpu.py --task trolley [--samples 4]"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
sys.path.insert(0, str(H.parent / "hb_psi0"))
FAILS = []


def check(c, m):
    print(("  ok   " if c else "  FAIL ") + m, flush=True)
    if not c:
        FAILS.append(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="trolley")
    ap.add_argument("--samples", type=int, default=4)
    a = ap.parse_args()
    import tyro
    import hb_act_args as AA
    import hb_act_config as C
    sys.modules["psi.config.train.hb_act_config"] = C
    r = AA.HA.receipt(a.task)
    SD, AD = int(r["state_dim"]), int(r["action_dim"])
    args = AA.build_args(a.task, a.samples, 100, 50, 3, tempfile.mkdtemp(prefix="hb_act_verify_"), "verify", 2, resume=False)
    cfg = tyro.cli(C.DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=args)
    m, mt = cfg.model, cfg.data.transform.model
    print(f"[1] task {a.task}: robot {r.get('robot')} state {SD} action {AD} cams {r['image_keys']}")
    check((m.action_dim, m.state_dim, m.chunk_size, m.n_action_steps) == (AD, SD, 16, 16), f"model A/S/chunk {m.action_dim}/{m.state_dim}/{m.chunk_size}")
    check(SD in (44, 78), f"raw state {SD} = 44 (G1) / 78 (H2), no joint velocity")
    check(not mt.img_aug and mt.hb_aug, "native ColorJitter off, HB v2 aug on")
    check(tuple(mt.resize.size) == (256, 480) and tuple(mt.center_crop.size) == (224, 224), f"native geometry resize {mt.resize.size} crop {mt.center_crop.size}")
    check(m.use_vae and m.kl_weight == 10.0 and cfg.train.learning_rate == 1e-4, "VAE kl 10, lr 1e-4 (native)")
    ds = cfg.data(split="train")
    check(len(ds) == r["frames"], f"len {len(ds)} == frames {r['frames']}")
    items = [ds[int(i)] for i in np.linspace(0, len(ds) - 1, a.samples).astype(int)]
    s = items[0]
    print("  sample keys", sorted(s))
    im = torch.as_tensor(s["observation.images"])
    check(tuple(im.shape) == (len(r["image_keys"]), 3, 224, 224), f"images {tuple(im.shape)} = (n_cameras, 3, 224, 224)")
    check(-2.2 < float(im.min()) and float(im.max()) < 2.7, f"ImageNet-normalised range [{float(im.min()):.2f}, {float(im.max()):.2f}]")
    st, ac = np.asarray(s["observation.state"]), np.asarray(s["action"])
    check(st.shape == (1, SD) and ac.shape == (16, AD), f"state {st.shape} action {ac.shape}")
    check(np.abs(ac).max() <= 1.0 + 1e-4, f"actions bounds-normalised (max |a| {np.abs(ac).max():.3f})")
    check(not np.asarray(s["action_is_pad"]).any(), "no action padding")
    batch = torch.utils.data.default_collate(items)
    from act.models.act import ACTConfig, ACTPolicy
    ac_cfg = ACTConfig(chunk_size=m.chunk_size, n_action_steps=m.n_action_steps, action_dim=m.action_dim, state_dim=m.state_dim,
                       dim_model=m.dim_model, n_heads=m.n_heads, dim_feedforward=m.dim_feedforward,
                       feedforward_activation=m.feedforward_activation, n_encoder_layers=m.n_encoder_layers,
                       n_decoder_layers=m.n_decoder_layers, pre_norm=m.pre_norm, dropout=m.dropout, use_vae=m.use_vae,
                       latent_dim=m.latent_dim, n_vae_encoder_layers=m.n_vae_encoder_layers, kl_weight=m.kl_weight,
                       temporal_ensemble_coeff=m.temporal_ensemble_coeff)
    pol = ACTPolicy(config=ac_cfg, dataset_stats=None)
    if batch["observation.state"].dim() == 3:
        batch["observation.state"] = batch["observation.state"].squeeze(1)
    batch = {k: (v.float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in batch.items()}
    out = pol(batch)
    check(all(torch.isfinite(torch.as_tensor(v)).all() for v in out.values() if torch.is_tensor(v)), f"ACT forward {{k: float(v) for k, v in out.items() if torch.is_tensor(v)}}".replace("{k", "").replace("}", "") + " " + str({k: round(float(v), 4) for k, v in out.items() if torch.is_tensor(v) or isinstance(v, float)}))
    print(f"  ACT params {sum(p.numel() for p in pol.parameters()) / 1e6:.1f} M")
    print("VERIFY_" + ("OK" if not FAILS else f"FAILED ({len(FAILS)})"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
