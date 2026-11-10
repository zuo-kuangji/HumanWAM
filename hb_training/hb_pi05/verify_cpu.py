"""CPU-only checks of the HB pi0.5 pipeline for one task (no GPU needed). Tasks: b6 basket a3 a7 (g1), a18 (h2).

  python verify_cpu.py --task a18           # uses the prepared view $HB_PI05_DATA/a18 (hb_prepare.py a18)
  python verify_cpu.py --task a18 --mini    # first builds a 2-episode view in a temp dir (no full prep needed)
  python verify_cpu.py --task b6 --forward  # + full-size PI0Pytorch fp32 forward on CPU with $HB_PI05_INIT (~35 GB RAM)
Checks: raw sample vs parquet, action chunk/clamp, visual augmentation (images changed, letterbox black, labels
untouched), proprio augmentation (delay <= 2 control periods within the episode, per-trajectory bias, per-sample noise,
only the n_body joints touched, delay rate ~15%), norm stats dims, OpenPI transform chain (image slots/masks, state
tokens not truncated for this layout's max_token_len, actions padded to 76 with zeros for h2), Unnormalize + HBOutputs
back to the raw action dim, torch loader batch -> Observation, OpenPI image aug disabled, PI0Pytorch structure on the
meta device, init checkpoint header shapes.
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="b6")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--samples", type=int, default=200)
    a = ap.parse_args()
    if a.mini:
        os.environ["HB_PI05_DATA"] = tempfile.mkdtemp(prefix="hb_pi05_mini_")
    import torch

    import hb_g1_data as D
    import hb_prepare

    if a.mini:
        hb_prepare.prepare(a.task, max_episodes=2, check_expected=False, workers=4)
    import hb_g1_config as C  # noqa: E402
    import openpi.models.model as _model
    import openpi.training.config as _config
    import openpi.training.data_loader as _data
    import openpi.transforms as _transforms

    tok = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser() / "big_vision/paligemma_tokenizer.model"
    check(tok.exists(), f"tokenizer present at {tok}")
    rng = np.random.default_rng(0)
    L = D.layout_of(a.task)
    cam0, cam1 = (f"observation/{c}" for c in L.cameras)
    print(f"task {a.task}: layout {L.name} state {L.state_dim} actions {L.action_dim} cams {L.cameras} "
          f"n_body {L.n_body} max_token_len {L.max_token_len}")

    # 1. raw sample vs parquet
    ds = D.HBCachedDataset(a.task, augment=False)
    n = len(ds)
    s = ds[5]
    check(s[cam0].shape == (224, 224, 3) and s[cam0].dtype == np.uint8 and s[cam1].shape == (224, 224, 3),
          f"{L.cameras} uint8 224x224x3")
    check(s["observation/state"].shape == (L.state_dim,) and s["actions"].shape == (16, L.action_dim),
          f"state {s['observation/state'].shape} actions {s['actions'].shape}")
    check(isinstance(s["prompt"], str) and len(s["prompt"]) > 0, f"prompt {s['prompt']!r}")
    src = D.TASKS[a.task].root
    info = json.loads((src / "meta/info.json").read_text())
    ep0 = hb_prepare.read_episode(src, info, 0, L.name)
    check(np.allclose(s["observation/state"], ep0["state"][5], atol=1e-6), f"state[5] == parquet ({L.state_desc})")
    check(np.allclose(s["actions"][0], ep0["action"][5], atol=1e-6) and np.allclose(s["actions"][15], ep0["action"][20], atol=1e-6),
          f"actions rows t..t+15 == parquet ({L.action_desc})")
    ln = len(ep0["state"])
    tail = ds[ln - 3]
    check(np.allclose(tail["actions"][2:], ep0["action"][ln - 1], atol=1e-6), "chunk clamped to the episode's last row")
    q = ds.state[:, :L.n_body]
    check(np.abs(q).max() < 2 * math.pi, f"perturbed joints look like radians (max |q| {np.abs(q).max():.2f})")
    print(f"  fps {ds.fps} control_hz {ds.control_hz} -> delay rows for 1/2 periods: "
          f"{max(1, round(ds.fps / ds.control_hz))}/{max(1, round(2 * ds.fps / ds.control_hz))}")

    # 2. visual augmentation (+ labels untouched)
    dsa = D.HBCachedDataset(a.task, augment=True)
    top, bottom, _, _ = ds.box
    changed, pad_black, act_ok = 0, True, True
    for i in rng.integers(0, n, 16):
        r, g = ds[int(i)], dsa[int(i)]
        changed += int(not np.array_equal(r[cam0], g[cam0]))
        for k in (cam0, cam1):
            pad_black &= bool(g[k][:top].max(initial=0) == 0 and g[k][bottom:].max(initial=0) == 0)
        act_ok &= bool(np.array_equal(g["actions"], r["actions"]))
    check(changed >= 12, f"augmentation changes images ({changed}/16)")
    check(pad_black, f"letterbox rows stay black (content rows {top}:{bottom})")
    check(act_ok, "action labels untouched by augmentation")

    # 3. proprio augmentation (CLAUDE.md sec. 3): delay, per-trajectory bias, per-sample noise
    dsa._lazy_init()  # noqa: SLF001
    nb, max_d = L.n_body, max(1, round(2 * ds.fps / ds.control_hz))
    bound = math.radians(D.BIAS_DEG + 3 * D.NOISE_DEG) + 1e-5
    delays, ok = [], True
    for i in rng.integers(0, n, 400):
        i = int(i)
        st, d = dsa.proprio(i, epoch=0)
        j = i - d
        ok &= bool(0 <= d <= max_d and j >= dsa.ep_start[i] and np.array_equal(st[nb:], ds.state[j][nb:])
                   and np.abs(st[:nb] - ds.state[j][:nb]).max() <= bound)
        delays.append(d)
    rate = np.mean(np.asarray(delays) > 0)
    check(ok, f"delay <= {max_d} rows within the episode; only state[:{nb}] perturbed within 2 deg + 3 sigma")
    check(0.08 <= rate <= 0.23, f"delayed fraction {rate:.3f} ~ 0.15 (clamped delays at episode start count as 0)")
    ep = int(ds.episode[0])
    b1, b2 = D.trajectory_bias(42, 0, ep, nb), D.trajectory_bias(42, 0, ep, nb)
    check(np.array_equal(b1, b2) and np.abs(b1).max() <= math.radians(D.BIAS_DEG), "bias deterministic per (seed, pass, episode), |b| <= 2 deg")
    check(not np.array_equal(b1, D.trajectory_bias(42, 1, ep, nb)) and not np.array_equal(b1, D.trajectory_bias(42, 0, ep + 1, nb)),
          "bias redrawn per pass and per episode")
    # same episode, same pass, no delay: difference between two samples' offsets is noise only (<= 2 * 3 sigma)
    same = [k for k in range(n) if ds.episode[k] == ep][:50]
    offs = []
    for k in same:
        st, d = dsa.proprio(k, epoch=0)
        if d == 0:
            offs.append(st[:nb] - ds.state[k][:nb])
    spread = np.ptp(np.stack(offs), axis=0).max() if len(offs) > 1 else 0.0
    check(spread <= 2 * math.radians(3 * D.NOISE_DEG) + 1e-5, f"bias fixed within the episode (offset spread {math.degrees(spread):.2f} deg)")
    check(dsa[n + 5]["actions"].shape == (16, L.action_dim), "index + pass*N decodes to row 5 of pass 1")

    # 4. config + norm stats
    cfg = C.make_train_config(a.task, batch_size=2, num_workers=0, num_train_steps=10)
    check(_config.get_config(D.config_name(a.task)).name == cfg.name, f"config {cfg.name} registered")
    m = cfg.model
    check(m.pi05 and m.action_dim == 76 and m.action_horizon == 16 and m.discrete_state_input
          and m.max_token_len == L.max_token_len, f"model {m}")
    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    check(dc.use_quantile_norm, "pi05 uses quantile norm")
    ns = dc.norm_stats or {}
    check(set(ns) == {"state", "actions"} and np.asarray(ns["state"].q01).shape == (L.state_dim,)
          and np.asarray(ns["actions"].q99).shape == (L.action_dim,),
          f"norm stats state[{L.state_dim}] actions[{L.action_dim}] (raw dims, no pad dim) with q01/q99")

    # 5. OpenPI transform chain
    tds = _data.transform_dataset(dsa, dc)
    t = tds[7]
    check(set(t["image"]) == set(_model.IMAGE_KEYS), f"image slots {sorted(t['image'])}")
    check(t["image"]["base_0_rgb"].shape == (224, 224, 3) and t["image"]["base_0_rgb"].dtype == np.uint8, "base_0_rgb uint8 224")
    check(bool(t["image_mask"]["base_0_rgb"]) and bool(t["image_mask"]["left_wrist_0_rgb"])
          and not bool(t["image_mask"]["right_wrist_0_rgb"]), "masks base/left True, right False")
    sw = max(L.state_dim, 76)
    check(t["state"].shape == (sw,) and np.all(t["state"][L.state_dim:] == 0), f"state after pad ({sw},)")
    check(t["actions"].shape == (16, 76) and np.all(t["actions"][:, L.action_dim:] == 0),
          f"actions (16,76); dims {L.action_dim}: are zero padding")
    lens = [int(tds[int(i)]["tokenized_prompt_mask"].sum()) for i in rng.integers(0, n, a.samples)]
    check(max(lens) < m.max_token_len, f"prompt+state tokens max {max(lens)} < max_token_len {m.max_token_len} (mean {np.mean(lens):.0f})")
    from openpi.models.tokenizer import PaligemmaTokenizer

    sp = PaligemmaTokenizer(m.max_token_len)._tokenizer  # noqa: SLF001
    print("  decoded prompt:", sp.decode([int(x) for x, k in zip(t["tokenized_prompt"], t["tokenized_prompt_mask"]) if k]))
    out = _transforms.compose([_transforms.Unnormalize(dc.norm_stats, use_quantiles=True), *dc.data_transforms.outputs])(
        {"actions": np.asarray(t["actions"], np.float64), "state": np.asarray(t["state"])})
    check(out["actions"].shape == (16, L.action_dim), f"Unnormalize + HBOutputs -> (16,{L.action_dim})")
    raw_rows = ds.action[np.minimum(np.arange(7, 23), ds.ep_end[7] - 1)]
    check(np.allclose(out["actions"], raw_rows, atol=1e-3 * (1 + np.abs(raw_rows).max())), "normalize -> unnormalize round trip")

    # 6. loader + OpenPI aug disabled
    D.install_hooks(augment=True)
    loader = _data.create_data_loader(cfg, framework="pytorch", shuffle=True, num_batches=1)
    obs, act = next(iter(loader))
    check(tuple(obs.images["base_0_rgb"].shape) == (2, 3, 224, 224) and obs.images["base_0_rgb"].dtype == torch.float32,
          f"batch image {tuple(obs.images['base_0_rgb'].shape)} float [-1,1]")
    check(tuple(obs.state.shape) == (2, sw) and tuple(act.shape) == (2, 16, 76), f"batch state (2,{sw}) actions (2,16,76)")
    import openpi.models_pytorch.preprocessing_pytorch as pp

    torch.manual_seed(0)
    o1 = pp.preprocess_observation_pytorch(obs, train=True)
    o2 = pp.preprocess_observation_pytorch(obs, train=False)
    check(all(torch.equal(o1.images[k], o2.images[k]) for k in o1.images), "OpenPI train-time image aug disabled")

    # 7. model structure (meta device) + init checkpoint header
    import openpi.models_pytorch.pi0_pytorch as p0

    mcfg = C.model_config(L.name, dtype="float32", pytorch_compile_mode=None)
    with torch.device("meta"):
        net = p0.PI0Pytorch(mcfg)
    check(tuple(net.action_in_proj.weight.shape) == (1024, 76) and tuple(net.action_out_proj.weight.shape) == (76, 1024),
          "PI0Pytorch action_in/out_proj sized for 76")
    print(f"  params: {sum(p.numel() for p in net.parameters()) / 1e9:.3f} B")
    init = Path(os.environ.get("HB_PI05_INIT", "")) / "model.safetensors"
    if init.exists():
        from safetensors import safe_open

        sd = net.state_dict()
        groups = {}
        for name, p in net.named_parameters(remove_duplicate=False):
            groups.setdefault(id(p), []).append(name)
        alias = {k: g for g in groups.values() for k in g}
        with safe_open(str(init), "pt") as f:
            keys = set(f.keys())
            bad = [k for k in keys if k not in sd or tuple(f.get_slice(k).get_shape()) != tuple(sd[k].shape)]
        missing = [k for k in sd if k not in keys and not any(x in keys for x in alias.get(k, []))]
        check(not bad and not missing, f"init checkpoint matches model ({len(keys)} tensors; bad {bad[:5]} missing {missing[:5]})")
    else:
        print(f"  (skip) {init} not found")

    if a.forward:
        import safetensors.torch

        net = p0.PI0Pytorch(mcfg)
        safetensors.torch.load_model(net, str(init))
        net.eval()
        with torch.no_grad():
            loss = net(obs, act.to(torch.float32))
        check(tuple(loss.shape) == (2, 16, 76) and torch.isfinite(loss).all().item(), f"fp32 CPU forward loss {loss.mean().item():.4f}")

    print("VERIFY_" + ("OK" if not FAILS else f"FAILED ({len(FAILS)})"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
