"""Stage public model repo humanoid-benchmark/groot (GR00T N1.7, groot-runs-bmax, per-GPU batch 320).
Layout: groot/<task-dir>/epoch-010|epoch-020/... ; big weight files are hard links, small files copied after a token scan.
DeepSpeed optimizer state (global_step*), RNG/scheduler state and zero_to_fp32.py are NOT staged.
Usage: python build_groot_repo.py $P
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

P = Path(sys.argv[1])
G = P / "policies-20261109"
RUNS = G / "groot-runs-bmax"
OUT = P / "hf-models-20261110-groot"
TASKDIR = {"b6": "b6-pedal-bin", "a3": "a3-fridge-open", "a7": "a7-cabinet-close", "a18": "a18-pour-water"}
TASKNAME = {"b6": "B6 pedal-bin trash disposal (G1)", "a3": "A3 open fridge door (G1)",
            "a7": "A7 close cabinet door (G1)", "a18": "A18 pour water (H2)"}
ROBOT = {"b6": "Unitree G1", "a3": "Unitree G1", "a7": "Unitree G1", "a18": "H2 (Sharpa hands)"}
CKPT_FILES = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json",
              "config.json", "embodiment_id.json", "processor_config.json", "statistics.json"]
SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}")
BIG = 50 * 1024 * 1024


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


def dims(groups):
    return {k: v["end"] - v["start"] for k, v in groups.items()}


if OUT.exists():
    shutil.rmtree(OUT)
R = OUT / "groot"
summary = {}
for t, td in TASKDIR.items():
    run = RUNS / t
    plan = json.loads((run / "production-complete.json").read_text())
    data = G / "groot-data" / t
    receipt = json.loads((data / "hb-dataset-receipt.json").read_text())
    modality = json.loads((data / "meta/modality.json").read_text())
    steps = [plan["save_steps"], plan["max_steps"]]
    for ep, step in zip((10, 20), steps):
        src = run / "production" / f"checkpoint-{step}"
        assert src.is_dir(), src
        dst = R / td / f"epoch-{ep:03d}"
        for name in CKPT_FILES:
            put(src / name, dst / name)
        for f in sorted((src / "experiment_cfg").iterdir()):
            put(f, dst / "experiment_cfg" / f.name)
        put(data / "meta/modality.json", dst / "dataset_meta/modality.json")
        put(data / "hb-dataset-receipt.json", dst / "dataset_meta/hb-dataset-receipt.json")
        sd, ad = dims(modality["state"]), dims(modality["action"])
        proto = {
            "model": "GR00T N1.7 (Gr00tN1d7), full fine-tune of the action head; VLM backbone frozen",
            "base_checkpoint": "GR00T N1.7 base (policies-20261109/models/groot-n17-base)",
            "vlm_backbone": "nvidia/Cosmos-Reason2-2B (gated on HF; needed at inference)",
            "task": TASKNAME[t], "task_dir": td, "robot": ROBOT[t], "embodiment_tag": "new_embodiment",
            "epoch_completed": ep, "global_step": step,
            "total_epochs": plan["epochs"], "total_steps": plan["max_steps"], "save_steps": plan["save_steps"],
            "per_gpu_batch": plan["per_gpu_batch"], "global_batch": plan["global_batch"], "num_gpus": 32,
            "lr": 1e-4, "lr_schedule": "cosine, warmup_ratio 0.05", "deepspeed_stage": 2,
            "episodes": receipt["episodes"], "frames": receipt["frames"], "fps": 30,
            "cameras": receipt["video_keys"],
            "state_groups": sd, "state_dim": sum(sd.values()),
            "action_groups": ad, "action_dim": sum(ad.values()),
            "action_chunk": 16, "model_action_horizon": 40, "state_encoding": "raw (apply_sincos_state_encoding false)",
            "normalization": "statistics.json (percentiles, use_percentiles true)",
            "augmentation": "Humanoid Benchmark online augmentation spec v2 (2026-11-09) via code/hb_groot/hb_launch.py; "
                            "GR00T native random crop / colour jitter / rotation bypassed in training "
                            "(config.json still lists the native defaults, they are not used)",
            "config": "experiment_cfg/conf.yaml (full training config); dataset_meta/modality.json (state/action/video keys)",
            "inference": "NVIDIA Isaac-GR00T (N1.7) policy loader on this folder with embodiment_tag new_embodiment and the "
                         "modality config in experiment_cfg/conf.yaml; import code/hb_groot/hb_offline_fix.py when "
                         "HF_HUB_OFFLINE=1. Augmentation is training-only.",
            "resume_state_included": False, "files": None,
        }
        proto["files"] = files_of(dst)
        (dst / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n")
        summary[f"{td}/epoch-{ep:03d}"] = {"step": step, "state_dim": proto["state_dim"],
                                           "action_dim": proto["action_dim"], "n_files": len(proto["files"])}
for f in sorted((G / "hb_groot").glob("*.py")):
    put(f, R / "code/hb_groot" / f.name)
for f in sorted((G / "hb_groot").glob("*.md")):
    put(f, R / "code/hb_groot" / f.name)

(R / "README.md").write_text("""---
tags:
- robotics
- humanoid
- imitation-learning
---
# GR00T N1.7 — Humanoid Benchmark

GR00T N1.7 checkpoints fine-tuned on Humanoid Benchmark tasks (2026-11-10 training round, per-GPU batch 320, 32 GPUs).

| Folder | Task | Robot |
|---|---|---|
| `b6-pedal-bin/` | Pick up a chip barrel, step on the pedal bin, drop it in | Unitree G1 |
| `a3-fridge-open/` | Open the fridge door | Unitree G1 |
| `a7-cabinet-close/` | Close the cabinet door | Unitree G1 |
| `a18-pour-water/` | Pour water from a pitcher into a glass | H2 |

Each task folder has `epoch-010/` and `epoch-020/` (100 demonstrations per task, 20 epochs, saved at epoch 10 and 20).
Every epoch folder is self-contained: sharded safetensors weights, `config.json`, processor config and normalization
statistics, the full training config (`experiment_cfg/`), the dataset modality layout (`dataset_meta/`) and
`protocol.json` (dimensions, step counts, file list). Only what evaluation needs is included (no optimizer / trainer / resume state).
The VLM backbone `nvidia/Cosmos-Reason2-2B` is loaded from its own (gated) repository.
`code/` holds the adapter code used for training.

Training used the Humanoid Benchmark online data augmentation spec v2 (photometric, geometric, sensor and lighting
augmentation plus proprioception noise/delay/drop), with GR00T's own image augmentation bypassed.
""")
tot = {"files": sum(1 for p in R.rglob("*") if p.is_file()),
       "GB": round(sum(p.stat().st_size for p in R.rglob("*") if p.is_file()) / 1e9, 2)}
print(json.dumps(summary, indent=1))
print("TOTALS", json.dumps(tot))
print("STAGED", OUT)
