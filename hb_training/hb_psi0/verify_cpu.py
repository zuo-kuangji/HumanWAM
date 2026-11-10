"""CPU-only checks before any GPU time (no training, no VLM weights loaded).

Usage: <psi0 python> verify_cpu.py [--task b6|basket|a3|a7|a18] [--samples 3] [--no-header] [--no-data]
  1. hb_args -> tyro -> DynamicLaunchConfig (same parse as scripts/train.py) and its HB validators;
     dims / keys / cameras / body dims must match the task's robot layout (G1 A76/S44/29, H2 A75/S78/31)
  2. step plans (20 epochs, save at 10) for every prepared task
  3. synthetic repack/field/model transform: delay rate, joint bias fixed per (epoch, episode) + per-sample
     noise within 3 sigma, untouched non-joint dims, no-aug output identical to the native
     Psi0ModelTransform, HB v2 aug active
  4. real view through LeRobotDatasetWrapper (+ timestamp sync check) -> shapes / ranges / cameras / episode_index
  5. action header built by the native Psi0Model ctor (dummy VLM) + re-dimensioned load of the real
     action_header.safetensors: newly initialized keys must equal hb_patches.expected_new(A, S, 30);
     CPU forward -> (1, 30, A)
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

for _k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT", "LOCAL_WORLD_SIZE", "GROUP_RANK"):
    os.environ.pop(_k, None)  # DLC node shells export these; psi overwatch would think it is distributed
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
H = Path(__file__).resolve().parent
G = Path(os.environ.get("G", H.parent))
PSI = G / "HumanWAM/third_party/Psi0"
sys.path[:0] = [str(H), str(PSI / "src")]
os.environ.setdefault("HB_AUG_DIR", str(G / "hb_groot"))
os.environ.setdefault("HF_DATASETS_CACHE", f"/tmp/hb_psi0_hfds_{os.getuid()}")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv(PSI / ".env")
FAIL = []
EXPECT = {"g1": (76, 44, 29, "observation.images.ego_view"), "h2": (75, 78, 31, "observation.images.head")}


def check(cond, msg):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        FAIL.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="b6")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--no-data", action="store_true")
    a = ap.parse_args()

    import tyro
    import hb_args
    import hb_patches
    import hb_psi0_config as C
    from psi.config import transform as pt
    sys.modules["psi.config.train.hb_psi0_config"] = C

    print("[1] config")
    check(C.A.VERSION == C.HB_AUG_VERSION, f"hb_aug_v2 imported from {C._AUG_DIR} ({C.A.VERSION})")
    r = hb_args.receipt(a.task)
    robot = r.get("robot")
    check(robot in EXPECT, f"receipt robot '{robot}' (rebuild old views with prepare_data.py --force)")
    if robot not in EXPECT:
        return 1
    AD, SD, NB, CAM0 = EXPECT[robot]
    SK, AK = r["state_key"], r["action_key"]
    out = tempfile.mkdtemp(prefix="hb_psi0_verify_")
    args = hb_args.build_args(a.task, 16, 100, 50, 3, out, "verify", 2, resume=False)
    cfg = tyro.cli(C.DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=args)
    m, rp, fd = cfg.model, cfg.data.transform.repack, cfg.data.transform.field
    check((m.action_dim, m.odim, m.action_chunk_size) == (AD, SD, 30), f"model A/S/H = {m.action_dim}/{m.odim}/{m.action_chunk_size}")
    check(len(fd.action_min) == AD and len(fd.state_min) == SD, f"stats widths {AD}/{SD} (no padding)")
    check(rp.hb_body_dims == NB, f"joint-perturbed dims {rp.hb_body_dims}: {r['perturbed_joint_names']}")
    check(rp.image_keys == r["image_keys"] and rp.image_keys[0] == CAM0, f"cameras {rp.image_keys} (cam0 = {CAM0})")
    mt = cfg.data.transform.model
    check(not mt.img_aug and not mt.view_aug and mt.hb_aug, "native img/view aug off, HB v2 aug on")
    check(fd.state_noise_std == 0 and rp.state_temporal_jitter == 0, "native state noise/jitter off")
    check(m.num_blocks == 12 and m.qk_norm == "rms_norm" and m.state_null_token and m.combined_temb,
          "native SONIC v1.1 header recipe flags")
    s32 = torch.tensor([1.0 / AD] * AD, dtype=torch.float32).sum().item()
    print(f"  info fp32 sum(1/{AD} x {AD}) on CPU = {s32!r} (hb_patches handles != 1.0 on the GPU)")

    print("[2] step plans (global batch = per_gpu x 32)")
    for t in hb_args.TASKS:
        try:
            fr = hb_args.receipt(t)["frames"]
        except SystemExit:
            print(f"  {t}: no view yet")
            continue
        print(f"  {t}: frames={fr} " + " ".join(f"B{b}:{hb_args.plan(fr, b)['steps']}/{hb_args.plan(fr, b)['save_steps']}"
                                               for b in (32, 24, 16, 12, 8)))

    print("[3] synthetic transforms")
    from transformers import AutoProcessor
    post = hb_args.POST
    proc = AutoProcessor.from_pretrained(str(post)) if post.is_dir() else None
    check(proc is not None, f"Qwen3-VL processor from init dir {post}")
    rng = np.random.default_rng(0)
    smin, smax = np.array(fd.state_min), np.array(fd.state_max)
    amin, amax = np.array(fd.action_min), np.array(fd.action_max)
    win = np.stack([(smin + smax) / 2 + 0.01 * f for f in range(3)]).astype(np.float32)  # frames t-2,t-1,t (distinct)
    act = (amin + (amax - amin) * rng.random((30, AD))).astype(np.float32)
    imgs = {k: torch.rand(3, 480, 640) for k in rp.image_keys}
    base = {SK: torch.from_numpy(win), AK: torch.from_numpy(act), "task": "Verify Task",
            "episode_index": torch.tensor(5), **imgs}
    o = rp(dict(base), no_aug=True)
    check(np.array_equal(o["states"], win[-1:]) and np.array_equal(o["actions"], act), "no_aug: state = s_t, actions untouched")
    nb = slice(NB, SD)
    sig3 = 3 * math.radians(C.NOISE_DEG)
    delays, worst, noise_ok = [0, 0, 0], 0.0, True
    C.set_epoch(0)
    bias0 = C.traj_bias(rp.hb_bias_seed, 0, 5, NB)
    for _ in range(3000):
        o = rp(dict(base))
        k = [kk for kk in range(3) if np.array_equal(o["states"][0, nb], win[2 - kk, nb])]
        assert len(k) == 1, "non-joint dims changed"
        delays[k[0]] += 1
        d = o["states"][0, :NB] - win[2 - k[0], :NB]
        worst = max(worst, float(np.abs(d).max()))
        noise_ok &= bool(np.abs(d - bias0).max() <= sig3 + 1e-5)  # same bias every sample of (epoch 0, episode 5)
        assert np.array_equal(o["actions"], act)
    p = (delays[1] + delays[2]) / 3000
    check(0.12 < p < 0.18 and delays[1] > 0 and delays[2] > 0, f"delay rate {p:.3f} (spec 0.15), k=1:{delays[1]} k=2:{delays[2]}")
    bound = math.radians(C.BIAS_DEG) + sig3
    check(0 < worst <= bound + 1e-5, f"joint perturbation max {math.degrees(worst):.3f} deg <= {math.degrees(bound):.2f}")
    check(noise_ok and float(np.abs(bias0).max()) <= math.radians(C.BIAS_DEG),
          "bias fixed within (epoch, trajectory); per-sample noise within 3 sigma")
    check(not np.allclose(bias0, C.traj_bias(rp.hb_bias_seed, 0, 6, NB))
          and not np.allclose(bias0, C.traj_bias(rp.hb_bias_seed, 1, 5, NB)), "bias differs across episodes and epochs")
    keep_p, rp.hb_delay_prob = rp.hb_delay_prob, 0.0
    C.set_epoch(1)
    d1 = rp(dict(base))["states"][0, :NB] - win[2, :NB]
    C.set_epoch(0)
    rp.hb_delay_prob = keep_p
    check(float(np.abs(d1 - C.traj_bias(rp.hb_bias_seed, 1, 5, NB)).max()) <= sig3 + 1e-5,
          "set_epoch(1) switches to the epoch-1 bias")
    if proc is not None:
        native = pt.Psi0ModelTransform(resize=mt.resize, center_crop=mt.center_crop)
        d0 = fd(rp(dict(base), no_aug=True), no_aug=True)
        x_hb = mt(dict(d0), vlm_processor=proc, no_aug=True)
        x_nat = native(dict(d0), vlm_processor=proc, no_aug=True)
        check(torch.equal(x_hb["pixel_values"], x_nat["pixel_values"]) and torch.equal(x_hb["input_ids"], x_nat["input_ids"]),
              "no_aug output identical to native Psi0ModelTransform")
        x_aug = mt(dict(d0), vlm_processor=proc)
        check(x_aug["pixel_values"].shape == x_nat["pixel_values"].shape and not torch.equal(x_aug["pixel_values"], x_nat["pixel_values"]),
              "HB v2 aug changes pixels, same token layout")
        check(int(x_nat["image_grid_thw"].shape[0]) == len(rp.image_keys), f"{len(rp.image_keys)} images per sample to Qwen3-VL")
        boosts = {}
        x = torch.rand(1, len(rp.image_keys), 3, 270, 480)
        for _ in range(150):
            b = C.A.augment(x, C.hb_rng())[1]["boost"]
            boosts[b] = boosts.get(b, 0) + 1
        check(len(boosts) >= 12, f"1.5x boost spread over {len(boosts)}/15 effects")

    if not a.no_data and proc is not None:
        print(f"[4] real view {hb_args.DATA / a.task}")
        t0 = time.time()
        ds = cfg.data(split="train", transform_kwargs={"vlm_processor": proc})
        check(len(ds) == r["frames"], f"len {len(ds)} == receipt frames {r['frames']} (built in {time.time() - t0:.1f}s)")
        tasks_lc = {x.lower() for x in r["tasks"]}
        for i in np.linspace(0, len(ds) - 1, a.samples).astype(int):
            raw = ds.raw_dataset[int(i)]
            check(tuple(raw[SK].shape) == (3, SD) and tuple(raw[AK].shape) == (30, AD),
                  f"raw[{i}] windows {tuple(raw[SK].shape)} {tuple(raw[AK].shape)}")
            check("episode_index" in raw, f"raw[{i}] carries episode_index={int(np.asarray(raw.get('episode_index', -1)))}")
            check(all(tuple(raw[k].shape) == (3, 480, 640) for k in rp.image_keys), f"raw[{i}] frames 3x480x640")
            t1 = time.time()
            s = ds[int(i)]
            st, ac = np.asarray(s["states"]), np.asarray(s["actions"])
            check(st.shape == (1, SD) and ac.shape == (30, AD) and np.abs(st).max() <= 1 and np.abs(ac).max() <= 1
                  and bool(np.all(s["actions_mask"])), f"sample[{i}] states {st.shape} actions {ac.shape} in [-1,1] ({time.time() - t1:.2f}s)")
            check(s["instruction"] in tasks_lc and int(s["image_grid_thw"].shape[0]) == len(rp.image_keys),
                  f"sample[{i}] instruction '{s['instruction']}' / {int(s['image_grid_thw'].shape[0])} images")

    if not a.no_header:
        print("[5] action header re-dimensioning (CPU, ~6 GB RAM)")
        from psi.models.psi0 import Psi0Model
        from psi.trainers.finetune import FinetuneTrainer
        model = Psi0Model(model_cfg=m, vlm_model=torch.nn.Linear(1, 1))
        head = model.action_header
        fake = SimpleNamespace(model=SimpleNamespace(action_header=head))
        expected = hb_patches.expected_new(m.action_dim, m.odim, m.action_chunk_size)
        rep = hb_patches.load_action_header_redim(
            head, str(post), lambda sd: FinetuneTrainer._reconcile_action_header_state_dict(fake, sd), expected)
        new = {e["key"] for e in rep["newly_initialized"]}
        for e in rep["newly_initialized"]:
            print(f"  new  {e['key']}: {e['model_shape']} (ckpt {e['checkpoint_shape']})")
        check(new == expected and len(new) == 6, f"newly initialized == expected 6 keys; loaded {rep['loaded_keys']}/{rep['model_keys']}")
        check(set(rep["checkpoint_keys_not_loaded"]) <= new, f"checkpoint keys not loaded: {rep['checkpoint_keys_not_loaded']}")
        head.eval()
        with torch.no_grad():
            y = head(hidden_states=None, timestep=torch.tensor([500.0]), pooled_projections=torch.randn(1, 768),
                     joint_attention_kwargs=dict(action_hidden_embeds=torch.randn(1, 30, AD), views=torch.randn(1, 12, 8, 2048),
                                                 obs=torch.rand(1, 1, SD) * 2 - 1, traj2ds=None),
                     vlm_attn_mask=torch.ones(1, 8)).action
        check(tuple(y.shape) == (1, 30, AD) and bool(torch.isfinite(y).all()), f"header forward -> {tuple(y.shape)}")
        Path(out, "hb_init_report.json").write_text(json.dumps(rep, indent=2))

    print("VERIFY_OK" if not FAIL else f"VERIFY_FAIL ({len(FAIL)}): " + "; ".join(FAIL))
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
