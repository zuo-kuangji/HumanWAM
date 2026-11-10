"""CPU-only checks of the HB DiT4DiT pipeline for one task (no GPU needed). Tasks: b6 a3 a7 (g1), a18 (h2).

  python verify_cpu.py --task a18            # uses the prepared view $HB_D4D_DATA/a18 (dit4dit_prepare.py a18)
  python verify_cpu.py --task a18 --mini     # first builds a 2-episode view in a temp dir (no full prep needed)
  python verify_cpu.py --task b6 --forward   # + full DiT4DiT (Cosmos 2B + 7B text encoder) CPU forward, ~25 GB RAM, slow
Checks: sample vs parquet (state/action rows, chunk + video clamp), native sample schema/dtypes, state = raw 44 (G1) /
78 (H2) dims with min_max (no velocity, no sin/cos) and action min_max round trip, images = cached frames side by side, visual augmentation (images change, labels
untouched, sensor effects only on the observation frame), proprio augmentation (delay <= 2 control periods in-episode, per-trajectory
bias, per-sample noise, only n_body joints), rank-sharded per-pass batches + exact resume position, step plan + launch
files, upstream contract (AST of the pinned files we patch), action head forward with our dims, Cosmos pipeline dir.
"""
import argparse
import ast
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
FAILS = []


def check(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        FAILS.append(msg)


def ast_methods(path, cls):
    tree = ast.parse(Path(path).read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            return {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="b6")
    ap.add_argument("--mini", action="store_true")
    ap.add_argument("--forward", action="store_true")
    a = ap.parse_args()
    if a.mini:
        os.environ["HB_D4D_DATA"] = tempfile.mkdtemp(prefix="hb_d4d_mini_")
    import torch

    import dit4dit_prepare
    import hb_d4d_config as C
    import hb_d4d_data as D

    if a.mini:
        dit4dit_prepare.prepare(a.task, max_episodes=2, check_expected=False, workers=4)
    src = Path(os.environ.get("HB_D4D_SRC", D.G / "HumanWAM/third_party/DiT4DiT"))
    sys.path.insert(0, str(src))
    L = D.layout_of(a.task)
    rng = np.random.default_rng(0)
    print(f"task {a.task}: layout {L.name} raw state {L.raw_state_dim} -> encoded {L.state_dim}, actions {L.action_dim}, "
          f"cams {L.cameras}, n_body {L.n_body}, segments {L.segments}")

    # 1. view vs parquet
    ds = D.HBD4DDataset(a.task, augment=False)
    meta, n = ds.meta, len(ds)
    if not a.mini:
        check(meta["episodes"] == D.TASKS[a.task].episodes and meta["frames"] == D.TASKS[a.task].frames,
              f"episodes {meta['episodes']} frames {meta['frames']} as expected (B6 = first 100 episodes)")
    spec = D.TASKS[a.task]
    info = json.loads((spec.root / "meta/info.json").read_text())
    ep0 = dit4dit_prepare.read_episode(spec.root, info, 0, L.name)
    ln = len(ep0["state"])
    check(np.allclose(ds.state[5], ep0["state"][5], atol=1e-5), f"raw state row 5 == parquet ({L.state_desc})")
    check(np.allclose(ds.action[np.arange(5, 21)], ep0["action"][5:21], atol=1e-5), f"raw action rows == parquet ({L.action_desc})")
    check(np.array_equal(ds.action_rows(ln - 3), np.minimum(np.arange(ln - 3, ln + 13), ln - 1)), "action chunk clamped to episode end")
    check(np.array_equal(ds.video_rows(ln - 5), np.minimum(ln - 5 + np.arange(0, 17, 2), ln - 1)), "video rows t..t+16 step 2, clamped")
    q = np.abs(ds.state[:, :L.n_body])
    check(q.max() < 2 * math.pi, f"perturbed joints look like radians (max |q| {q.max():.2f})")
    print(f"  fps {ds.fps} control_hz {ds.control_hz} -> delay rows 1/2 periods: "
          f"{max(1, round(ds.fps / ds.control_hz))}/{max(1, round(2 * ds.fps / ds.control_hz))}; "
          f"non-body state range {meta['hand_range']}; constant dims {meta['constant_dims']}")

    # 2. native sample schema
    s = ds[5]
    check(set(s) == {"action", "image", "lang", "state", "action_mask", "state_mask"}, f"sample keys {sorted(s)}")
    check(isinstance(s["image"], list) and len(s["image"]) == len(D.VIDEO_OFFSETS) == 9, "9 frames (t..t+16 step 2)")
    im0 = s["image"][0]
    check(tuple(im0.shape) == (3, 224, 448) and im0.dtype == torch.float16 and 0 <= float(im0.min()) and float(im0.max()) <= 1,
          f"frame {tuple(im0.shape)} {im0.dtype} in [0,1]")
    check(s["state"].shape == (1, L.state_dim) and s["state"].dtype == np.float16, f"state {s['state'].shape} float16")
    check(s["action"].shape == (D.HORIZON, L.action_dim) and s["action"].dtype == np.float16, f"action {s['action'].shape} float16")
    check(s["action_mask"].all() and s["action_mask"].shape == s["action"].shape and s["state_mask"].all(), "masks all True")
    check(isinstance(s["lang"], str) and len(s["lang"]) > 0, f"lang {s['lang']!r}")

    # 3. encodings
    enc = D.encode_state(ds.state[5], L, ds.smin, ds.smax)
    nb = L.n_body
    want = D.STATE_DIMS[L.name]
    check(L.raw_state_dim == L.state_dim == enc.shape[-1] == ds.state.shape[1] == len(ds.smin) == want
          and s["state"].shape == (1, want),
          f"model state = raw {want} dims (G1 44 / H2 78: no velocity, no sin/cos)")
    mm = D.minmax(ds.state[5], ds.smin, ds.smax)
    check(np.allclose(enc, mm) and np.abs(mm).max() <= 1 + 1e-9, "state encoding: every dim -> native min_max [-1,1]")
    check(np.allclose(s["state"][0].astype(np.float64), enc, atol=2e-3), "sample state == encode_state(raw row)")
    act = s["action"].astype(np.float64)
    check(np.abs(act).max() <= 1.0 + 1e-3, f"normalised actions within [-1,1] (max {np.abs(act).max():.4f})")
    back = D.unminmax(act, ds.amin, ds.amax)
    raw = ds.action[ds.action_rows(5)]
    tol = 1e-3 * (ds.amax - ds.amin) + 1e-6
    const = ds.amax == ds.amin
    check(np.all((np.abs(back - raw) <= tol) | const), "action min_max round trip (fp16 precision)")

    # 4. images == cached frames, side by side
    f0 = np.load(f"{ds.root}/frames_{L.cameras[0]}.npy", mmap_mode="r")[5]
    f1 = np.load(f"{ds.root}/frames_{L.cameras[1]}.npy", mmap_mode="r")[5]
    left = (im0[:, :, :224].float() * 255).permute(1, 2, 0).numpy()
    right = (im0[:, :, 224:].float() * 255).permute(1, 2, 0).numpy()
    check(np.abs(left - f0).max() < 0.6 and np.abs(right - f1).max() < 0.6, f"frame 0 = [{L.cameras[0]} | {L.cameras[1]}] of row 5")
    check(f0.shape == (224, 224, 3) and f0.dtype == np.uint8, "cache uint8 224x224")

    # 5. visual augmentation
    dsa = D.HBD4DDataset(a.task, augment=True)
    changed, labels_ok, fut_changed = 0, True, 0
    for i in rng.integers(0, n, 12):
        r, g = ds[int(i)], dsa[int(i)]
        changed += int(not torch.equal(r["image"][0], g["image"][0]))
        fut_changed += int(not torch.equal(r["image"][4], g["image"][4]))
        labels_ok &= bool(np.array_equal(g["action"], r["action"]))
    check(changed >= 10 and fut_changed >= 8, f"augmentation changes obs ({changed}/12) and future frames ({fut_changed}/12)")
    check(labels_ok, "action labels untouched by augmentation")
    import hb_aug_v2 as aug

    x = torch.rand(9, 2, 3, 64, 64)
    m = torch.zeros(9, dtype=torch.bool)
    m[0] = True
    p = aug.sample(np.random.default_rng(1), 2, 64)
    p.update(noise=0.05, jpeg=None, gaussian_blur=0.0, motion_blur=None, sharpen=0.0, defocus=0.0)
    y = aug.apply(x, p, m, noise_seed=3)
    p0 = dict(p, noise=0.0)
    y0 = aug.apply(x, p0, m, noise_seed=3)
    check(not torch.equal(y[0], y0[0]) and torch.equal(y[1:], y0[1:]), "sensor noise only on the observation frame (obs_mask)")

    # 6. proprio augmentation
    dsa._lazy_init()  # noqa: SLF001
    max_d = max(1, round(2 * ds.fps / ds.control_hz))
    bound = math.radians(D.BIAS_DEG + 3 * D.NOISE_DEG) + 1e-6
    delays, ok = [], True
    for i in rng.integers(0, n, 400):
        i = int(i)
        st, d = dsa.proprio(i, epoch=0)
        j = i - d
        ok &= bool(0 <= d <= max_d and j >= dsa.ep_start[i] and np.allclose(st[nb:], ds.state[j][nb:])
                   and np.abs(st[:nb] - ds.state[j][:nb]).max() <= bound)
        delays.append(d)
    rate = float(np.mean(np.asarray(delays) > 0))
    check(ok, f"delay <= {max_d} rows within the episode; only state[:{nb}] perturbed within 2 deg + 3 sigma")
    check(0.08 <= rate <= 0.23, f"delayed fraction {rate:.3f} ~ 0.15")
    ep = int(ds.episode[0])
    b1 = D.trajectory_bias(42, 0, ep, nb)
    check(np.array_equal(b1, D.trajectory_bias(42, 0, ep, nb)) and np.abs(b1).max() <= math.radians(D.BIAS_DEG),
          "bias deterministic per (seed, pass, episode), |b| <= 2 deg")
    check(not np.array_equal(b1, D.trajectory_bias(42, 1, ep, nb)) and not np.array_equal(b1, D.trajectory_bias(42, 0, ep + 1, nb)),
          "bias redrawn per pass and per episode")
    offs = []
    for k in [k for k in range(n) if ds.episode[k] == ep][:60]:
        st, d = dsa.proprio(k, epoch=0)
        if d == 0:
            offs.append(st[:nb] - ds.state[k][:nb])
    spread = np.ptp(np.stack(offs), axis=0).max() if len(offs) > 1 else 0.0
    check(spread <= 2 * math.radians(3 * D.NOISE_DEG) + 1e-6, f"bias fixed within the episode (offset spread {math.degrees(spread):.2f} deg)")
    check(dsa[n + 5]["action"].shape == (D.HORIZON, L.action_dim), "index + pass*N decodes to row 5 of pass 1")

    # 7. loader: rank sharding + resume position
    B, W = 3, 4
    lds = [D.HBLoader(ds, per_gpu_batch=B, rank=r, world=W, num_workers=0, seed=7) for r in range(W)]
    allidx = [i for ldr in lds for bt in ldr.pass_batches(2) for i in bt]
    check(len(allidx) == len(set(allidx)) == lds[0].batches_per_pass * B * W and all(2 * n <= i < 3 * n for i in allidx),
          f"pass 2: ranks disjoint, {len(allidx)} distinct rows, encoded with pass offset")
    check(lds[0].pass_batches(0) != lds[0].pass_batches(1), "reshuffled every pass")
    k = lds[1].batches_per_pass + 2
    it = lds[1].stream(start_step=k)
    first = next(it)
    exp = lds[1].pass_batches(1)[2]
    check(len(first) == B and all(np.array_equal(first[j]["action"], ds[exp[j]]["action"]) for j in range(B)),
          "stream(start_step) resumes at the exact batch")

    # 8. plan + launch files
    pl = C.plan(a.task, 8, 32, frames=n)
    gb = 8 * 32
    check(pl["steps"] == math.ceil(20 * n / gb) and pl["save_steps"] == sorted({math.ceil(10 * n / gb), pl["steps"]}),
          f"plan: steps {pl['steps']}, saves {pl['save_steps']}, warmup {pl['warmup_steps']} (B=8, 32 GPUs)")
    tmpd = Path(tempfile.mkdtemp(prefix="hb_d4d_launch_"))
    C.write_launch_files(a.task, 8, 32, 4, tmpd)
    dsj = json.loads((tmpd / "ds_config.json").read_text())
    check(dsj["train_micro_batch_size_per_gpu"] == 8 and dsj["train_batch_size"] == 256 and dsj["zero_optimization"]["stage"] == 2
          and dsj["bf16"]["enabled"], "ds_config: ZeRO-2 bf16, explicit batch")
    yml = (tmpd / "accelerate.yaml").read_text()
    check("distributed_type: DEEPSPEED" in yml and str(tmpd / "ds_config.json") in yml, "accelerate.yaml -> our ds_config")
    if C.native_yaml().exists():
        cfg = C.make_cfg(pl, run_root=str(tmpd), cosmos_dir="/x", resume=False, probe=False)
        am, vd, tr = cfg.framework.action_model, cfg.datasets.vla_data, cfg.trainer
        check(am.action_dim == L.action_dim == vd.max_action_dim and am.state_dim == L.state_dim == vd.max_state_dim
              and am.future_action_window_size == 15 and am.past_action_window_size == 0,
              f"cfg dims A{am.action_dim} S{am.state_dim} H{am.future_action_window_size + 1}")
        check(list(vd.video_delta_indices) == list(range(17)) and vd.action_video_freq_ratio == 2 and vd.per_device_batch_size == 8,
              "cfg video 0..16 ratio 2, per-device batch")
        check(tr.freeze_modules == C.FREEZE and tr.learning_rate.action_model == 1e-4 and tr.learning_rate.backbone_interface == 1e-5
              and tr.repeated_diffusion_steps == 4 and tr.max_train_steps == pl["steps"] and tr.eval_interval > pl["steps"],
              "cfg trainer: frozen text_encoder+vae, lr 1e-5/1e-4, 4 repeats, steps, eval off")
        check(cfg.framework.cosmos25.extract_layer == 17 and cfg.framework.cosmos25.flow_matching.time_distribution == "uniform",
              "cfg cosmos25: layer 17, uniform flow-matching time")
    else:
        print(f"  (skip cfg) {C.native_yaml()} not found")

    # 9. upstream contract (AST, no import side effects)
    tr_py = src / "DiT4DiT/training/train.py"
    if tr_py.exists():
        txt = tr_py.read_text()
        check('DeepSpeedPlugin(hf_ds_config=' in txt and "from DiT4DiT.dataloader import build_dataloader" in txt
              and "from accelerate import Accelerator, DeepSpeedPlugin" in txt, "train.py builds DeepSpeedPlugin / imports build_dataloader by name")
        meth = ast_methods(tr_py, "VLATrainer")
        need = {"prepare_training", "_create_data_iterators", "_save_checkpoint", "_finalize_training", "_log_metrics",
                "_get_next_batch", "train", "_init_checkpointing"}
        check(need <= set(meth), f"VLATrainer has {sorted(need)}")
        tt = ast_methods(src / "DiT4DiT/training/trainer_utils/trainer_tools.py", "TrainerUtils")
        check("setup_distributed_training" in tt and "_get_latest_checkpoint" in tt, "TrainerUtils.setup_distributed_training present")
        ce = ast_methods(src / "DiT4DiT/model/modules/vlm/Cosmos25.py", "Cosmos25FeatureExtractor")
        gp = ce.get("_get_prompt_embeds")
        check(gp is not None and [x.arg for x in gp.args.kwonlyargs] == ["device", "dtype", "max_sequence_length"],
              "Cosmos25FeatureExtractor._get_prompt_embeds(prompt, *, device, dtype, max_sequence_length)")
        dtxt = (src / "DiT4DiT/dataloader/gr00t_lerobot/datasets.py").read_text()
        check("action=action," in dtxt and "image=all_images," in dtxt and "action_mask=action_mask," in dtxt
              and "state_mask=state_mask," in dtxt, "native sample dict keys match")
    else:
        print(f"  (skip contract) {tr_py} not found")

    # 10. action head with our dims (random backbone features)
    if C.native_yaml().exists():
        from DiT4DiT.model.modules.action_model.ActionDiT import FlowmatchingActionHead

        torch.manual_seed(0)
        head = FlowmatchingActionHead(full_config=cfg).float()
        batch = [ds[int(i)] for i in rng.integers(0, n, 2)]
        acts = torch.tensor(np.stack([b["action"] for b in batch]), dtype=torch.float32)
        st = torch.tensor(np.stack([b["state"] for b in batch]), dtype=torch.float32)
        mask = torch.from_numpy(np.stack([b["action_mask"] for b in batch]))
        vl = torch.randn(2, 64, int(cfg.framework.cosmos25.vl_hidden_dim))
        loss = head(vl, acts, mask, st)
        pred = head.predict_action(vl, st)
        check(bool(torch.isfinite(loss)) and tuple(pred.shape) == (2, D.HORIZON, L.action_dim),
              f"action head forward loss {float(loss):.3f}, predict {tuple(pred.shape)}")
        print(f"  action head params: {sum(p.numel() for p in head.parameters()) / 1e6:.1f} M")

    # 11. Cosmos pipeline dir
    cos = Path(os.environ.get("HB_COSMOS_DIR", D.G / "models/cosmos-predict2.5-2b-diffusers-post-trained"))
    if (cos / "model_index.json").exists():
        tc = json.loads((cos / "transformer/config.json").read_text())
        te = json.loads((cos / "text_encoder/config.json").read_text())
        te_txt = {**te, **(te.get("text_config") or {})}  # newer transformers nest the LM config
        vc = json.loads((cos / "vae/config.json").read_text())
        check(tc["in_channels"] == 17 and tc["num_layers"] == 28 > 17 and tc["num_attention_heads"] * tc["attention_head_dim"] == 2048,
              "transformer: 17 in-channels, 28 blocks (> extract_layer 17), hidden 2048 (= vl_hidden_dim)")
        check(tc.get("crossattn_proj_in_channels") == te_txt["num_hidden_layers"] * te_txt["hidden_size"] == 100352,
              "text encoder 28 x 3584 = crossattn_proj_in_channels 100352")
        check(vc.get("z_dim") == 16 and vc.get("temperal_downsample") == [False, True, True], "Wan2.1 VAE (z16, temporal x4)")
        prov = cos / "HB_PROVENANCE.json"
        print("  provenance:", prov.read_text()[:400] if prov.exists() else "(none: official copy?)")
    else:
        print(f"  (skip) {cos} not built yet (fetch_models.sh)")

    if a.forward:
        from omegaconf import OmegaConf

        from DiT4DiT.model.framework import build_framework

        cfg2 = C.make_cfg(pl, run_root=str(tmpd), cosmos_dir=str(cos), resume=False, probe=False)
        cfg2.trainer.repeated_diffusion_steps = 1
        model = build_framework(OmegaConf.create(OmegaConf.to_container(cfg2, resolve=True)))
        model = model.to(torch.bfloat16)  # as the DeepSpeed bf16 engine does (CPU has no cuda autocast to fp32)
        out = model.forward([ds[5]])
        check(all(bool(torch.isfinite(v)) for v in out.values()), f"full CPU forward: { {k: float(v) for k, v in out.items()} }")

    print("VERIFY_" + ("OK" if not FAILS else f"FAILED ({len(FAILS)})"))
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
