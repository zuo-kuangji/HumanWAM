"""Stage the inference sidecar files (config, normalization stats, Qwen text cache, protocol) for the four
ImageWAM Delta-Flow runs whose epoch-040 weights are in humanoid-benchmark/g1-basket-transfer-imagewam-deltaflow76.
Copies the ORIGINAL files the training config points to; computes nothing new. Usage: python build_sidecars.py $P
"""
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

P = Path(sys.argv[1])
OUT = P / "hf-sidecars-20261110"

RUNS = [
    dict(repo_dir="a3-fridge-open/epoch-040", run="a3-a7-20261109/a3", ckpt="step_004440.pt", step=4440, epochs=40,
         dataset="humanoid-benchmark/a3-fridge-open-expert-v1", revision="780fcf1e4bd7db947a8c2d8f7b92e61f24f71b45",
         robot="g1", extra_stats=["velocity-normalization.json"],
         state_layout="body29 joint angles + measured hands12 + projected gravity3 + body29 joint velocity (rad/s, normalized with velocity-normalization.json)",
         action_layout="motion_token64 + same-frame measured left6 + measured right6"),
    dict(repo_dir="a7-cabinet-close/epoch-040", run="a3-a7-20261109/a7", ckpt="step_004800.pt", step=4800, epochs=40,
         dataset="humanoid-benchmark/a7-cabinet-close-expert-v1", revision="1935a98e8f55d4d1d2d1615ddb97c2679713595c",
         robot="g1", extra_stats=["velocity-normalization.json"],
         state_layout="body29 joint angles + measured hands12 + projected gravity3 + body29 joint velocity (rad/s, normalized with velocity-normalization.json)",
         action_layout="motion_token64 + same-frame measured left6 + measured right6"),
    dict(repo_dir="h2-a18-pour-water/epoch-040", run="a18-h2-pour-from-scratch-20261108", ckpt="step_011400.pt", step=11400,
         epochs=40, dataset="humanoid-benchmark/a18-pour-water-expert-v1",
         revision="8e956a102c84f991d2745a8cc66155c9991b742d", robot="h2", extra_stats=[],
         state_layout="H2 schema a18-h2-sharpa-body31-hand44-state78-chunk16-v1 (see normalization.json)",
         action_layout="H2 schema a18-h2-sharpa-body31-hand44-state78-chunk16-v1 (see normalization.json)"),
    dict(repo_dir="epoch-040", run="g1-basket-transfer-ed77719-40epochs", ckpt="step_008520.pt", step=8520, epochs=40,
         dataset="humanoid-benchmark/basket-transfer-expert-v1", revision="ed77719266ca618e0ec2fbfb3cda7e6792fe3fb1",
         robot="g1", extra_stats=[],
         state_layout="body29 joint angles + measured hands12 + projected gravity3",
         action_layout="motion_token64 + same-frame measured left6 + measured right6"),
]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cfg_value(text, key):
    m = re.search(rf"^\s*{key}:\s*(.+?)\s*$", text, re.M)
    return m.group(1).strip().strip("'\"") if m else None


if OUT.exists():
    shutil.rmtree(OUT)
summary = []
for r in RUNS:
    src = P / r["run"]
    cfg_path = src / "run/production/config.yaml"
    cfg = cfg_path.read_text()
    stats_src = Path(cfg_value(cfg, "stats_path"))
    text_dir = Path(cfg_value(cfg, "qwen_text_cache_dir"))
    dataset_dir = Path(cfg_value(cfg, "dataset_dir"))
    dst = OUT / r["repo_dir"]
    (dst / "qwen-text").mkdir(parents=True)

    files = {"training_config.yaml": cfg_path, stats_src.name: stats_src}
    for name in r["extra_stats"]:
        files[name] = stats_src.parent / name
    text_files = sorted(text_dir.glob("*.pt"))
    assert len(text_files) == 1, (r["run"], text_files)
    files["qwen-text/" + text_files[0].name] = text_files[0]

    sidecars = {}
    for rel, s in files.items():
        shutil.copy2(s, dst / rel)
        sidecars[rel] = {"sha256": sha256(dst / rel), "bytes": (dst / rel).stat().st_size,
                         "original_nas_path": str(s).replace(str(P), "$P")}

    info = {}
    info_path = dataset_dir / "meta/info.json"
    if info_path.is_file():
        j = json.loads(info_path.read_text())
        info = {"total_episodes": j.get("total_episodes"), "total_frames": j.get("total_frames"), "fps": j.get("fps")}
    hf_receipt = src / "hf-verified.json"
    ckpt_receipt = json.loads(hf_receipt.read_text()) if hf_receipt.is_file() else None

    protocol = {
        "run": r["run"],
        "checkpoint": f"{r['repo_dir']}/{r['ckpt']}",
        "checkpoint_receipt": ckpt_receipt,
        "global_step": r["step"],
        "epoch_completed": r["epochs"],
        "config_max_steps": int(cfg_value(cfg, "max_steps")),
        "robot": r["robot"],
        "source_dataset": r["dataset"],
        "source_revision": r["revision"],
        "dataset_meta": info,
        "action_dim": int(cfg_value(cfg, "action_dim")),
        "state_dim": int(cfg_value(cfg, "action_state_dim")),
        "chunk": 16,
        "history": 16,
        "state_layout": r["state_layout"],
        "action_layout": r["action_layout"],
        "normalization_file": stats_src.name,
        "qwen_context_len": int(cfg_value(cfg, "qwen_context_len")),
        "qwen_text_cache": "qwen-text/" + text_files[0].name,
        "sidecar_note": ("Files copied unchanged from the paths the training config used (stats_path, "
                         "qwen_text_cache_dir); training_config.yaml is the production config.yaml verbatim, so its "
                         "absolute paths refer to the training NAS. Point stats_path / qwen_text_cache_dir (and "
                         "velocity normalization, if present) at the files in this folder. Do not substitute the B6 "
                         "config: model input dims differ per run."),
        "sidecars": sidecars,
        "staged_utc": "2026-11-10",
    }
    if r["robot"] == "h2":
        protocol["schedule_note"] = ("config_max_steps reflects the original 60-epoch plan; the target was changed to "
                                     "40 epochs and this checkpoint is step 11400 = epoch 40.")
    (dst / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summary.append({"repo_dir": r["repo_dir"], "files": sorted(sidecars) + ["protocol.json"],
                    "state_dim": protocol["state_dim"], "action_dim": protocol["action_dim"], "dataset_meta": info,
                    "config_max_steps": protocol["config_max_steps"]})

print(json.dumps(summary, indent=1))
print("STAGED", OUT)
