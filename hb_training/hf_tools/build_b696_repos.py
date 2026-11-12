"""Stage the public model repos humanoid-benchmark/{groot,pi05} for the 2026-11-11 training round (global batch 696 =
87 per GPU x 8 GPUs, one node per run, 20 epochs, saved at epoch 10 and 20). Replaces the previous round in place
(the older checkpoints stay in the repo history; the revision is named in README.md).
Layout: <repo>/<task-dir>/epoch-010|epoch-020/... ; weights are hard links, small files copied after a token scan.
Only what deployment / evaluation needs: weights, model + processor config, normalization statistics, modality layout,
protocol.json, plus the training records kept for reproducibility (training plan, trainer state, training args, training
log). Resume state (optimizer / DeepSpeed shards / RNG / scheduler) is not uploaded.
Incremental: the staging folder keeps its .cache (upload_large_folder bookkeeping); everything else is rebuilt.
Usage: python build_b696_repos.py $P [groot] [pi05]
"""
import json
import os
import re
import shutil
import sys
from pathlib import Path

P = Path(sys.argv[1])
WHICH = sys.argv[2:] or ["groot", "pi05"]
G = P / "policies-20261109"
OUT = P / "hf-models-20261111-b696"
PREV = {"groot": "2e8f375584631cae0454911e58b5b7850ea9f678", "pi05": "b04edfb31a62fbcd7e85b7c3a85b6dfc8cd3587b"}
TASKS = ["b6", "a3", "a7", "a18", "trolley", "basket", "a17", "b7"]
TASKDIR = {"b6": "b6-pedal-bin", "a3": "a3-fridge-open", "a7": "a7-cabinet-close", "a18": "a18-pour-water",
           "trolley": "a10-trolley-push", "basket": "a20-basket-transfer", "b7": "b7-black-clothes-hamper", "a17": "a17-unscrew-bottle"}
TASKNAME = {"b6": "B6 pedal-bin trash disposal (G1)", "a3": "A3 open fridge door (G1)",
            "a7": "A7 close cabinet door (G1)", "a18": "A18 pour water (H2)", "trolley": "A10 trolley push (G1)",
            "basket": "A20 basket transfer (G1)", "b7": "B7 black clothes into the laundry hamper (G1)", "a17": "A17 unscrew bottle cap (H2)"}
DESC = {"b6": "Pick up a chip barrel, step on the pedal bin, drop it in", "a3": "Open the fridge door",
        "a7": "Close the cabinet door", "a18": "Pour water from a pitcher into a glass", "trolley": "Push a trolley",
        "basket": "Transfer a basket", "b7": "Take the black garment off the hanger and drop it into the hamper", "a17": "Unscrew the cap of a bottle"}
HFDATA = {"b6": "b6 v5t7 ready-start (first 100 episodes)", "a3": "a3-fridge-open-expert-v1",
          "a7": "a7-cabinet-close-expert-v1", "a18": "a18 H2 pour water", "trolley": "trolley-push-expert-v1 @ef9024a",
          "basket": "basket-transfer-expert-v1 @b0ff522 (current-dynamics-100-v21)",
          "a17": "a17-unscrew-bottle-expert-v1 @407e6e2",
          "b7": "b7-black-clothes-hamper-expert-v1 @1bb002a (first 100 of 107)"}
ROBOT = {t: ("H2 (Sharpa hands)" if t in ("a18", "a17") else "Unitree G1") for t in TASKS}
AUG = ("Humanoid Benchmark online augmentation spec v2 (2026-11-09), on from epoch 1; model-native image "
       "augmentation off")
SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}")
_TOK = Path("/root/.hb-secrets/hf_token")
TOKEN = _TOK.read_text().strip().encode() if _TOK.is_file() else None
BIG = 50 * 1024 * 1024


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size >= BIG:
        os.link(src, dst)
    else:
        data = src.read_bytes()
        if SECRET.search(data) or (TOKEN and TOKEN in data):
            raise SystemExit(f"ABORT: token-like string in {src}")
        dst.write_bytes(data)


def files_of(d):
    return {str(p.relative_to(d)): p.stat().st_size for p in sorted(d.rglob("*")) if p.is_file()}


def reset(repo_dir):
    """Delete everything except the upload cache."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    for p in repo_dir.iterdir():
        if p.name == ".cache":
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()


def dims(groups):
    return {k: v["end"] - v["start"] for k, v in groups.items()}


def meta_of(t):
    return json.loads((G / "pi05-data" / t / "meta.json").read_text())


def readme(title, model, done, extra):
    rows = "\n".join(f"| `{TASKDIR[t]}/` | {DESC[t]} | {ROBOT[t]} |" for t in done)
    return f"""---
tags:
- robotics
- humanoid
- imitation-learning
---
# {title}

{model} checkpoints fine-tuned on Humanoid Benchmark tasks (2026-11-11 training round: global batch 696 = 87 per GPU
x 8 GPUs, one node per run, 20 epochs).

| Folder | Task | Robot |
|---|---|---|
{rows}

Each task folder has `epoch-010/` and `epoch-020/` (100 demonstrations per task, 20 epochs, saved at epoch 10 and 20).
Every epoch folder is self-contained for deployment and evaluation: weights, model / processor configuration,
normalization statistics and `protocol.json` (input/output dimensions, cameras, step counts, file list).
Training records are included for reproducibility (`training_plan.json`, `train.log`, and for GR00T `trainer_state.json`,
`training_args.bin`, `wandb_config.json`); optimizer / scheduler / RNG / DeepSpeed resume state is not. `code/` holds the adapter code used for training and inference.

Model inputs are identical across all benchmark models: G1 state 44-D (29 body joint angles + 6 + 6 measured hand
values + 3 projected gravity), H2 state 78-D (75 joint angles + 3 projected gravity), no joint velocities.
Training used the Humanoid Benchmark online data augmentation spec v2 (photometric, geometric, sensor and lighting
augmentation plus proprioception noise/delay/drop), with the model's own image augmentation turned off.
{extra}
The previous training round of this repository is kept in the git history at revision `{{prev}}` (older batch setting).
"""


summary = {}
# ---------------- GR00T ----------------
if "groot" in WHICH:
    R = OUT / "groot"
    reset(R)
    done = []
    for t in TASKS:
        run = G / "groot-runs-b696" / t
        if not (run / "production-complete.json").is_file():
            continue
        plan = json.loads((run / "production-complete.json").read_text())
        assert plan["global_batch"] in (696, 512), plan  # 512 for runs started after 2026-11-11 20:27 (user)
        data = G / "groot-data" / t
        receipt = json.loads((data / "hb-dataset-receipt.json").read_text())
        modality = json.loads((data / "meta/modality.json").read_text())
        meta = meta_of(t)
        assert receipt["frames"] == plan["frames"] == meta["frames"], (t, receipt["frames"], plan["frames"])
        for ep, step in zip((10, 20), (plan["save_steps"], plan["max_steps"])):
            src = run / "production" / f"checkpoint-{step}"
            dst = R / TASKDIR[t] / f"epoch-{ep:03d}"
            for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                         "model.safetensors.index.json", "config.json", "embodiment_id.json",
                         "processor_config.json", "statistics.json"):
                put(src / name, dst / name)
            for name in ("trainer_state.json", "training_args.bin", "wandb_config.json"):  # training records
                if (src / name).is_file():
                    put(src / name, dst / name)
            put(run / "production-complete.json", dst / "training_plan.json")
            put(run / "logs/production.log", dst / "train.log")
            for f in sorted((src / "experiment_cfg").iterdir()):
                put(f, dst / "experiment_cfg" / f.name)
            put(data / "meta/modality.json", dst / "dataset_meta/modality.json")
            put(data / "hb-dataset-receipt.json", dst / "dataset_meta/hb-dataset-receipt.json")
            sd, ad = dims(modality["state"]), dims(modality["action"])
            proto = {
                "model": "GR00T N1.7 (Gr00tN1d7), full fine-tune of the action head; VLM backbone frozen",
                "base_checkpoint": "GR00T N1.7 base (policies-20261109/models/groot-n17-base)",
                "vlm_backbone": "nvidia/Cosmos-Reason2-2B (gated on HF; needed at inference)",
                "task": TASKNAME[t], "task_dir": TASKDIR[t], "training_data": HFDATA[t], "robot": ROBOT[t],
                "embodiment_tag": "new_embodiment", "epoch_completed": ep, "global_step": step,
                "total_epochs": plan["epochs"], "total_steps": plan["max_steps"], "save_steps": plan["save_steps"],
                "per_gpu_batch": plan["per_gpu_batch"], "num_gpus": plan["world_size"], "num_nodes": plan["world_size"] // 8,
                "global_batch": plan["global_batch"], "lr": 1e-4, "lr_schedule": "cosine, warmup_ratio 0.05",
                "weight_decay": 1e-5, "deepspeed_stage": 2,
                "episodes": receipt["episodes"], "frames": receipt["frames"], "fps": meta["fps"],
                "control_hz": meta.get("control_hz"), "cameras": receipt["video_keys"],
                "state_groups": sd, "state_dim": sum(sd.values()), "action_groups": ad, "action_dim": sum(ad.values()),
                "action_chunk": 16, "model_action_horizon": 40,
                "state_encoding": "raw (apply_sincos_state_encoding false, GR00T N1.7 default)",
                "normalization": "statistics.json (percentiles, use_percentiles true)",
                "augmentation": AUG + " (via code/hb_groot/hb_launch.py; config.json still lists GR00T's native "
                                      "augmentation defaults, they are bypassed in training)",
                "config": "experiment_cfg/conf.yaml (full model config); dataset_meta/modality.json (state/action/"
                          "video keys)",
                "inference": "NVIDIA Isaac-GR00T (N1.7) policy loader on this folder with embodiment_tag "
                             "new_embodiment and the modality config in experiment_cfg/conf.yaml; import "
                             "code/hb_groot/hb_offline_fix.py when HF_HUB_OFFLINE=1. Augmentation is training-only.",
                "training_records_included": True, "resume_state_included": False, "files": None,
            }
            assert proto["state_dim"] in (44, 78), (t, proto["state_dim"])
            proto["files"] = files_of(dst)
            (dst / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n")
            summary[f"groot/{TASKDIR[t]}/epoch-{ep:03d}"] = {"step": step, "state": proto["state_dim"],
                                                             "action": proto["action_dim"], "n": len(proto["files"])}
        done.append(t)
    for f in sorted((G / "hb_groot").glob("*.py")) + sorted((G / "hb_groot").glob("*.md")):
        put(f, R / "code/hb_groot" / f.name)
    if (P / "hf-repro-20261111/groot/VERSIONS.json").is_file():
        put(P / "hf-repro-20261111/groot/VERSIONS.json", R / "code/VERSIONS.json")
    (R / "README.md").write_text(readme("GR00T N1.7 — Humanoid Benchmark", "GR00T N1.7", done,
                                        "The VLM backbone `nvidia/Cosmos-Reason2-2B` is loaded from its own (gated) "
                                        "repository.\n").replace("{prev}", PREV["groot"]))

# ---------------- pi0.5 ----------------
if "pi05" in WHICH:
    R = OUT / "pi05"
    reset(R)
    done = []
    for t in TASKS:
        runs = list((G / "pi05-runs/b512" / t / "run/ckpt").glob("*/run")) or list((G / "pi05-runs/b696" / t / "run/ckpt").glob("*/run"))
        if len(runs) != 1 or not (runs[0] / "HB_DONE").exists():
            continue
        run = runs[0]
        plan = json.loads((run / "hb_plan.json").read_text())
        assert plan["global_batch"] in (696, 512), plan  # 512 for runs started after 2026-11-11 20:27 (user)
        meta = meta_of(t)
        for ep, step in zip((10, 20), plan["save_steps"]):
            src = run / str(step)
            dst = R / TASKDIR[t] / f"epoch-{ep:03d}"
            put(src / "model.safetensors", dst / "model.safetensors")
            put(src / "metadata.pt", dst / "metadata.pt")
            put(run / "hb_plan.json", dst / "training_plan.json")  # training records
            if (P / "hf-repro-20261111/pi05" / TASKDIR[t] / "train_config.json").is_file():
                put(P / "hf-repro-20261111/pi05" / TASKDIR[t] / "train_config.json", dst / "train_config.json")
            _lg = run.parent.parent.parent / "logs"
            put(next(x for x in (_lg / "train.log", _lg / "train-node-0.log") if x.is_file()), dst / "train.log")
            for a in (src / "assets").rglob("*"):
                if a.is_file():
                    put(a, dst / "assets" / a.relative_to(src / "assets"))
            asset_id = next((src / "assets").iterdir()).name
            proto = {
                "model": "pi0.5 (openpi, PyTorch weights)", "task": TASKNAME[t], "task_dir": TASKDIR[t],
                "training_data": HFDATA[t], "robot": ROBOT[t],
                "openpi_config": plan["config"], "layout": plan["layout"], "asset_id": asset_id,
                "epoch_completed": ep, "global_step": step, "total_epochs": plan["epochs"],
                "total_steps": plan["num_train_steps"], "save_steps": plan["save_steps"],
                "global_batch": plan["global_batch"], "per_gpu_batch": plan["per_gpu_batch"],
                "num_gpus": plan["world_size"], "num_nodes": plan["world_size"] // 8, "peak_lr": plan.get("peak_lr"),
                "warmup_steps": plan.get("warmup_steps"), "max_token_len": plan.get("max_token_len"),
                "init_weights": "lerobot/pi05_base converted to PyTorch, action/state projections resized to the "
                                "benchmark action dimension",
                "augmentation": AUG if plan.get("augment") else "off", "aug_version": plan.get("aug_version"),
                "action_horizon": 16, "episodes": meta["episodes"], "frames": meta["frames"], "fps": meta["fps"],
                "control_hz": meta.get("control_hz"), "prompt": meta.get("prompt"),
                "raw_state_dim": meta.get("raw_state_dim"), "state_layout": meta.get("state_layout"),
                "action_layout": meta.get("action_layout"), "cameras": meta.get("cameras"),
                "raw_image_hw": meta.get("raw_hw"), "content_box": meta.get("content_box"),
                "normalization": f"assets/{asset_id}/norm_stats.json (openpi quantile normalization)",
                "inference": "Register configs by importing code/hb_pi05/hb_g1_config.py (needs hb_g1_data.py on "
                             "the path), then openpi policy_config.create_trained_policy(get_config(openpi_config), "
                             "<this folder>). Clients send observation/<camera> raw RGB, observation/state and "
                             "prompt; see HBInputs.",
                "training_records_included": True, "resume_state_included": False, "files": None,
            }
            proto["files"] = files_of(dst)
            (dst / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n")
            summary[f"pi05/{TASKDIR[t]}/epoch-{ep:03d}"] = {"step": step, "n": len(proto["files"])}
        done.append(t)
    for f in sorted((G / "hb_pi05").glob("*.py")):
        put(f, R / "code/hb_pi05" / f.name)
    if (P / "hf-repro-20261111/pi05/VERSIONS.json").is_file():
        put(P / "hf-repro-20261111/pi05/VERSIONS.json", R / "code/VERSIONS.json")
    pending = [t for t in TASKS if t not in done]
    extra = (f"Still training, added when finished: {', '.join(TASKDIR[t] for t in pending)}.\n" if pending else "")
    (R / "README.md").write_text(readme("pi0.5 — Humanoid Benchmark", "pi0.5 (openpi)", done, extra)
                                 .replace("{prev}", PREV["pi05"]))

for repo in WHICH:
    d = OUT / repo
    fs = [p for p in d.rglob("*") if p.is_file() and ".cache" not in p.relative_to(d).parts]
    print("TOTALS", repo, len(fs), "files", round(sum(p.stat().st_size for p in fs) / 1e9, 2), "GB")
print(json.dumps(summary, indent=1))
print("STAGED", OUT)
