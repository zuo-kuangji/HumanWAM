"""Reproducibility fields for every published protocol.json (2026-11-23 user; hb_training/REPRODUCIBILITY.md section 8).

Used by hb_autoupload.py (new uploads) and hb_repro_backfill.py (already published folders). Adds a top-level "seed"
and a "reproducibility" block: seed source, augmentation random stream, proprio-bias / in-loop seeds, layout,
environment versions, hardware, upstream code, protocol document + commit. Never changes an existing field.
"""
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE.parent                                   # .../preflight-20261104T1648Z
BAI = P / "HumanWAM-bai"
PROTOCOL_URL = "https://github.com/zuo-kuangji/HumanWAM/blob/bai/hb_training/REPRODUCIBILITY.md"
SEEDED_FROM = "2026-11-23 21:05 CST (13:05 UTC)"
HARDWARE = "Alibaba PAI DLC, 4 nodes x 8 NVIDIA A800-SXM4-80GB, driver 570.133.20, CUDA 12.8"
INLOOP_SEED = 1234

# HF repo -> (training seed, where it comes from, env name, upstream code, augmentation stream seeded in every run)
MODELS = {
    "groot": (42, "upstream Isaac-GR00T default (gr00t/configs/data/data_config.py seed=42 -> set_seed; HF TrainingArguments.seed=42)",
              "groot-py312", "NVIDIA Isaac-GR00T @ 7aa0bca", False),
    "psi0": (292285, "upstream Psi0 official finetune scripts (--seed=292285)", "psi0", "Psi0 @ cda3fdf", False),
    "act": (292285, "LeRobot ACT of the Psi0 repository, same launch arguments as Psi-Zero (--seed=292285)", "psi0",
            "Psi0 @ cda3fdf", False),
    "fastwam": (42, "upstream FastWAM configs/train.yaml (seed: 42)", "fastwam",
                "github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4", False),
    "pi05": (42, "openpi TrainConfig.seed=42", "openpi", "openpi @ 215abfb", True),
    "dit4dit": (42, "hb_dit4dit/hb_d4d_config.py seed=42", "dit4dit", "DiT4DiT @ 66a6f3a", True),
    "deltaflow": (42, "deltaflow-runs/<task>/run/production.yaml seed=42", None, "HumanWAM wholebody Delta Flow tree", True),
}
SEEDED_TEXT = {
    "pi05": "seeded: numpy Generator from (torch.initial_seed() of the DataLoader worker, global rank); joint bias from (seed, pass, episode)",
    "dit4dit": "seeded: HB_AUG_SEED=42, data order permutation from (seed, pass)",
    "deltaflow": "seeded: image_policy seed 20261110, proprio_policy seed 20261106, mixed with rank / worker",
}


def _env(name):
    if not name:
        return None
    f = BAI / "hb_training" / "envs" / f"{name}.txt"
    out = {"name": name, "package_snapshot": f"hb_training/envs/{name}.txt"}
    try:
        txt = f.read_text()
        m = re.search(r"^# env \S+ python (\S+)", txt, re.M)
        t = re.search(r"^torch==(\S+)", txt, re.M | re.I)
        out.update(python=m.group(1) if m else None, torch=t.group(1) if t else None)
    except OSError:
        pass
    return out


def _commit():
    try:
        return subprocess.run(["git", "-C", str(BAI), "rev-parse", "--short=12", "HEAD"], capture_output=True,
                              text=True, timeout=30).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def find_logs(paths, depth=5):
    """Training logs next to the run that produced the uploaded files (<run>/logs/*.log)."""
    seen, logs = set(), []
    for p in paths:
        for parent in list(Path(p).parents)[:depth]:
            d = parent / "logs"
            if d in seen or not d.is_dir():
                continue
            seen.add(d)
            logs += sorted(x for x in d.glob("*.log") if x.is_file())
    return logs


def seed_in_logs(logs, cap=8 << 20):
    for f in logs:
        try:
            with open(f, "rb") as fh:
                data = fh.read(cap)
        except OSError:
            continue
        m = re.search(rb"HB_AUG_SEED=(\d+)", data)
        if m:
            return int(m.group(1))
    return None


def fields(repo, proto, *, run_logs=(), seed_found=None, backfill=None):
    """Return {"seed": ..., "reproducibility": {...}} for model repo `repo` (groot, psi0, fastwam, ...)."""
    if repo not in MODELS:
        return {}
    seed, src, env, upstream, always_seeded = MODELS[repo]
    aug_seed = seed_in_logs(run_logs) if run_logs else None
    if always_seeded:
        aug = SEEDED_TEXT[repo]
    elif aug_seed is not None:
        aug = (f"seeded: HB_AUG_SEED={aug_seed} (= training seed); numpy Generator from (seed, torch.initial_seed() of the "
               f"process / DataLoader worker, global rank, worker id) - reproducible for the same seed and layout")
    else:
        aug = ("not seeded: per-sample augmentation choices were drawn from a pid / os.urandom / time based generator "
               "(training started before the " + SEEDED_FROM + " fix); the augmentation distribution is exactly spec v2, "
               "but the individual draws are not bit-reproducible")
    used = seed_found if seed_found is not None else seed
    r = {
        "protocol": PROTOCOL_URL,
        "protocol_commit": _commit(),
        "training_seed": used,
        "training_seed_source": src,
        "training_seed_verified_from_run_files": seed_found is not None,
        "augmentation_random_stream": aug,
        "inloop_mse_seed": INLOOP_SEED,
        "layout": {k: proto[k] for k in ("num_nodes", "num_gpus", "world_size", "per_gpu_batch", "global_batch",
                                         "gradient_accumulation_steps") if k in proto} or None,
        "environment": _env(env),
        "hardware": HARDWARE,
        "upstream_code": upstream,
        "adapter_code": "code/ in this folder (snapshot of the HumanWAM hb_training adapters used for this run)",
        "determinism": ("statistical reproducibility: same data commit, config, seed and layout give the same curves within "
                        "run-to-run noise; GPU kernels (bf16, NCCL reductions) are not bitwise deterministic"),
        "exceptions": "see the exception registry, REPRODUCIBILITY.md section 7",
    }
    if repo in ("psi0", "act"):
        r["proprio_bias_seed"] = 292285
    if seed_found is not None and seed_found != seed:
        r["seed_note"] = f"run files record seed {seed_found}, model default is {seed}"
    if backfill:
        r["backfilled"] = backfill
    return {"seed": used, "reproducibility": r}
