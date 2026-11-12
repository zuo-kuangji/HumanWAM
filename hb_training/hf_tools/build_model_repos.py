"""Stage public model repos humanoid-benchmark/pi05 and humanoid-benchmark/psi0 (2026-11-10 runs, hb1110).
Layout: <repo>/<task-dir>/epoch-010|epoch-020/... ; big weight files are hard links (no copy), small files copied.
Resume state (optimizer, RNG) is NOT staged. Usage: python build_model_repos.py $P
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

P = Path(sys.argv[1])
G = P / "policies-20261109"
OUT = P / "hf-models-20261110"
TASKDIR = {"b6": "b6-pedal-bin", "a3": "a3-fridge-open", "a7": "a7-cabinet-close", "a18": "a18-pour-water"}
TASKNAME = {"b6": "B6 pedal-bin trash disposal (G1)", "a3": "A3 open fridge door (G1)",
            "a7": "A7 close cabinet door (G1)", "a18": "A18 pour water (H2)"}
AUG = "Humanoid Benchmark online augmentation spec v2 (2026-11-09), on from epoch 1; model-native augmentation off"
SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}")
BIG = 50 * 1024 * 1024


def rel(p):
    return str(p).replace(str(P), "$P")


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size >= BIG:
        os.link(src, dst)
    else:
        data = src.read_bytes()
        if SECRET.search(data):
            raise SystemExit(f"ABORT: token-like string in {src}")
        dst.write_bytes(data)


def files_of(d):
    return {str(p.relative_to(d)): p.stat().st_size for p in sorted(d.rglob("*")) if p.is_file()}


if OUT.exists():
    shutil.rmtree(OUT)
summary = {"pi05": {}, "psi0": {}}

# ---------------- pi05 ----------------
R = OUT / "pi05"
for t, td in TASKDIR.items():
    run = next((G / "pi05-runs/chain-hb1110" / t / "run/ckpt").glob("*/run"))
    plan = json.loads((run / "hb_plan.json").read_text())
    meta = json.loads((G / "pi05-data" / t / "meta.json").read_text())
    for ep, step in zip((10, 20), plan["save_steps"]):
        src = run / str(step)
        dst = R / td / f"epoch-{ep:03d}"
        put(src / "model.safetensors", dst / "model.safetensors")
        put(src / "metadata.pt", dst / "metadata.pt")
        for a in (src / "assets").rglob("*"):
            if a.is_file():
                put(a, dst / "assets" / a.relative_to(src / "assets"))
        put(run / "hb_plan.json", dst / "training_plan.json")
        asset_id = next((src / "assets").iterdir()).name
        proto = {
            "model": "pi0.5 (openpi, PyTorch weights)", "task": TASKNAME[t], "task_dir": td,
            "openpi_config": plan["config"], "layout": plan["layout"], "asset_id": asset_id,
            "epoch_completed": ep, "global_step": step, "total_epochs": plan["epochs"],
            "total_steps": plan["num_train_steps"], "global_batch": plan["global_batch"],
            "per_gpu_batch": plan["per_gpu_batch"], "world_size": plan["world_size"], "peak_lr": plan["peak_lr"],
            "warmup_steps": plan["warmup_steps"], "max_token_len": plan["max_token_len"],
            "init_weights": "lerobot/pi05_base converted to PyTorch, action/state projections resized to 76-D "
                            "(" + rel(Path(plan["weights"])) + ")",
            "augmentation": AUG if plan.get("augment") else "off", "aug_version": plan.get("aug_version"),
            "model_action_dim": 76, "action_horizon": 16,
            "episodes": meta["episodes"], "frames": meta["frames"], "fps": meta["fps"],
            "control_hz": meta.get("control_hz"), "prompt": meta["prompt"],
            "raw_state_dim": meta.get("raw_state_dim"), "state_layout": meta.get("state_layout"),
            "action_layout": meta.get("action_layout"), "cameras": meta.get("cameras"),
            "raw_image_hw": meta.get("raw_hw"), "content_box": meta.get("content_box"),
            "training_data": rel(Path(meta["source"])),
            "normalization": f"assets/{asset_id}/norm_stats.json (openpi quantile normalization)",
            "inference": "Register configs by importing code/hb_pi05/hb_g1_config.py (needs hb_g1_data.py on the path), "
                         "then openpi policy_config.create_trained_policy(get_config(openpi_config), <this folder>). "
                         "Clients send observation/<camera> raw RGB, observation/state and prompt; see HBInputs.",
            "resume_state_included": False, "files": None,
        }
        proto["files"] = files_of(dst)
        (dst / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n")
        summary["pi05"][f"{td}/epoch-{ep:03d}"] = {"step": step, "files": proto["files"]}
for f in sorted((G / "hb_pi05").glob("*.py")):
    put(f, R / "code/hb_pi05" / f.name)

# ---------------- psi0 ----------------
R = OUT / "psi0"
for t, td in TASKDIR.items():
    base = G / "psi0-runs" / t / "hb1110"
    dec = json.loads((base / "decision.json").read_text())
    fin = next((base / "train/finetune").iterdir())
    init = json.loads((fin / "hb_init_report.json").read_text())
    steps = sorted(int(p.name.split("_")[1]) for p in (fin / "checkpoints").glob("ckpt_*"))
    assert steps == [dec["save_steps"], dec["steps"]], (t, steps, dec["save_steps"], dec["steps"])
    for ep, step in zip((10, 20), steps):
        dst = R / td / f"epoch-{ep:03d}"
        put(fin / "checkpoints" / f"ckpt_{step}" / "model.safetensors", dst / "model.safetensors")
        for name in ("run_config.json", "dataset_statistics.json", "clip_pooled_cache.pt", "argv.txt",
                     "hb_init_report.json", "hb_metrics.jsonl"):
            put(fin / name, dst / name)
        proto = {
            "model": "Psi-Zero (Psi0 action expert, flow matching)", "task": TASKNAME[t], "task_dir": td,
            "run": fin.name, "robot": dec["robot"], "epoch_completed": ep, "global_step": step,
            "total_epochs": dec["epochs"], "total_steps": dec["steps"], "global_batch": dec["global_batch"],
            "per_gpu_batch": dec["per_gpu_batch"], "warmup_steps": dec["warmup_steps"], "frames": dec["frames"],
            "state_dim": dec["state_dim"], "action_dim": dec["action_dim"], "image_keys": dec["image_keys"],
            "init_checkpoint": rel(Path(init["checkpoint"])),
            "init_newly_initialized_keys": [k["key"] for k in init.get("newly_initialized", [])],
            "augmentation": AUG,
            "normalization": "dataset_statistics.json",
            "config": "run_config.json (full training config); argv.txt (launch arguments)",
            "text_cache": "clip_pooled_cache.pt",
            "inference": "Psi0 code (third_party/Psi0) plus the patches in code/hb_psi0 (hb_psi0_config.py, "
                         "hb_patches.py); load model.safetensors with run_config.json and dataset_statistics.json.",
            "resume_state_included": False, "files": None,
        }
        proto["files"] = files_of(dst)
        (dst / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n")
        summary["psi0"][f"{td}/epoch-{ep:03d}"] = {"step": step, "files": proto["files"]}
for f in sorted((G / "hb_psi0").glob("*.py")):
    put(f, R / "code/hb_psi0" / f.name)

# ---------------- READMEs ----------------
README = """---
tags:
- robotics
- humanoid
- imitation-learning
---
# {title}

{model} checkpoints fine-tuned on Humanoid Benchmark tasks (2026-11-10 training round).

| Folder | Task | Robot |
|---|---|---|
| `b6-pedal-bin/` | Pick up a chip barrel, step on the pedal bin, drop it in | Unitree G1 |
| `a3-fridge-open/` | Open the fridge door | Unitree G1 |
| `a7-cabinet-close/` | Close the cabinet door | Unitree G1 |
| `a18-pour-water/` | Pour water from a pitcher into a glass | H2 |

Each task folder has `epoch-010/` and `epoch-020/` (100 demonstrations per task, 20 epochs, saved at epoch 10 and 20).
Every epoch folder is self-contained: weights, normalization statistics, training config and `protocol.json`
(dimensions, input/output layout, step counts, file list). Optimizer / resume state is not included.
`code/` holds the adapter code used for training and inference.

Training used the Humanoid Benchmark online data augmentation spec v2 (photometric, geometric, sensor and lighting
augmentation plus proprioception noise/delay/drop), with the model's own image augmentation turned off.
"""
(OUT / "pi05/README.md").write_text(README.format(title="pi0.5 — Humanoid Benchmark", model="pi0.5 (openpi)"))
(OUT / "psi0/README.md").write_text(README.format(title="Psi-Zero — Humanoid Benchmark", model="Psi-Zero (Psi0)"))

tot = {}
for repo in ("pi05", "psi0"):
    tot[repo] = {"files": sum(1 for p in (OUT / repo).rglob("*") if p.is_file()),
                 "GB": round(sum(p.stat().st_size for p in (OUT / repo).rglob("*") if p.is_file()) / 1e9, 2)}
print(json.dumps(summary, indent=1))
print("TOTALS", json.dumps(tot))
print("STAGED", OUT)
