"""Fill the reproducibility gaps in the public model repos (user 2026-11-11):
  pi05     : <task>/epoch-*/train_config.json (fully resolved openpi TrainConfig of that run)       [phase pi05cfg]
  psi0     : <task>/epoch-*/training_plan.json, train.log, envs.txt, train_args.txt (b6/a3/a7/a18 hb1110 runs)
  psi0, act: code/hb_aug/hb_aug_v2.py (the shared augmentation implementation); act also code/hb_psi0/*.py
  all repos: code/VERSIONS.json (upstream commits, local modifications, key package versions, adapter code commit)
Usage: python repro_supplement.py $P pi05cfg   (run with the openpi env, after `source hb_pi05/hb_env.sh`)
       python repro_supplement.py $P upload    (psi0 env; token from /root/.hb-secrets/hf_token)
"""
import dataclasses
import enum
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from pathlib import Path

P = Path(sys.argv[1]); PHASE = sys.argv[2]
G = P / "policies-20261109"
OUT = P / "hf-repro-20261111"
TASKDIR = {"b6": "b6-pedal-bin", "a3": "a3-fridge-open", "a7": "a7-cabinet-close", "a18": "a18-pour-water",
           "trolley": "a10-trolley-push", "basket": "a20-basket-transfer", "a17": "a17-unscrew-bottle",
           "b7": "b7-black-clothes-hamper"}


def conv(o, depth=0):
    if depth > 14:
        return repr(o)
    if dataclasses.is_dataclass(o) and not isinstance(o, type):
        return {"__type__": type(o).__name__, **{f.name: conv(getattr(o, f.name), depth + 1) for f in dataclasses.fields(o)}}
    if isinstance(o, (list, tuple)):
        return [conv(x, depth + 1) for x in o]
    if isinstance(o, dict):
        return {str(k): conv(v, depth + 1) for k, v in o.items()}
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, enum.Enum):
        return str(o)
    if isinstance(o, pathlib.Path):
        return str(o)
    return repr(o)


if PHASE == "pi05cfg":
    sys.path.insert(0, str(G / "hb_pi05"))
    import hb_g1_config as C
    for t, td in TASKDIR.items():
        for base in ("b512", "b696"):
            runs = list((G / "pi05-runs" / base / t / "run/ckpt").glob("*/run"))
            if runs and (runs[0] / "HB_DONE").exists():
                break
        else:
            continue
        run = runs[0]
        plan = json.loads((run / "hb_plan.json").read_text())
        cfg = C.make_train_config(t, exp_name=plan["exp_name"], batch_size=plan["global_batch"],
                                  num_train_steps=plan["num_train_steps"], num_workers=8,
                                  checkpoint_base_dir=str(run.parent.parent), resume=False,
                                  pytorch_weight_path="default", log_interval=10)
        d = {"note": "Fully resolved openpi TrainConfig of this run, rebuilt with code/hb_pi05/hb_g1_config.py "
                     "make_train_config() from training_plan.json (same arguments as hb_train.py). Absolute paths "
                     "refer to the training NAS.", "training_plan": plan, "train_config": conv(cfg)}
        dst = OUT / "pi05" / td / "train_config.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(d, indent=1) + "\n")
        print("CFG", t, base, dst.stat().st_size)
    sys.exit(0)

# ---------------------------------------------------------------- upload phase
from huggingface_hub import CommitOperationAdd, HfApi  # noqa: E402

TOK = Path("/root/.hb-secrets/hf_token").read_text().strip()
api = HfApi(token=TOK)


def git(path):
    def r(*a):
        try:
            return subprocess.run(["git", "-C", str(path), *a], capture_output=True, text=True, timeout=60).stdout.strip()
        except Exception:
            return ""
    mods = [l for l in r("status", "--porcelain").splitlines() if "__pycache__" not in l]
    return {"url": r("remote", "get-url", "origin"), "commit": r("rev-parse", "HEAD"), "local_modifications": mods}


def pkgs(py, names):
    code = ("import json, importlib.metadata as m, sys\nout = {}\nfor n in sys.argv[1:]:\n    try: out[n] = m.version(n)\n"
            "    except Exception: pass\nprint(json.dumps(out))")
    try:
        return json.loads(subprocess.run([str(py), "-c", code, *names], capture_output=True, text=True, timeout=120).stdout)
    except Exception as e:
        return {"error": repr(e)}


TP = G / "HumanWAM/third_party"
COMMON_PKGS = ["torch", "torchvision", "numpy", "transformers", "accelerate", "deepspeed", "diffusers", "safetensors",
               "lerobot", "datasets", "flash-attn", "peft", "jax", "flax", "openpi", "gr00t", "av", "torchcodec"]
adapter = {"repo": "https://github.com/zuo-kuangji/HumanWAM", "branch": "bai", **git(P / "HumanWAM-bai"),
           "note": "code/ in this repository is the exact copy of the adapter code used for these runs"}
VER = {
    "groot": {"upstream": {"Isaac-GR00T": git(TP / "Isaac-GR00T"), "HumanWAM": git(G / "HumanWAM")},
              "python_packages": pkgs(G / "envs/groot-py312/bin/python", COMMON_PKGS)},
    "pi05": {"upstream": {"openpi": git(TP / "openpi")}, "python_packages": pkgs(G / "envs/openpi/bin/python", COMMON_PKGS)},
    "psi0": {"upstream": {"Psi0": git(TP / "Psi0")}, "python_packages": pkgs(G / "envs/psi0/bin/python", COMMON_PKGS)},
    "act": {"upstream": {"Psi0 (baselines/act, src/act)": git(TP / "Psi0")},
            "python_packages": pkgs(G / "envs/psi0/bin/python", COMMON_PKGS)},
    "deltaflow": {"upstream": {"HumanWAM": {"url": "https://github.com/zuo-kuangji/HumanWAM",
                                            "branch": "wholebody-deltaflow-training-20261027",
                                            "base_commit": "9cb38e1075ad1ac7379f56fec6daadc532508b5f",
                                            "note": "legacy runs: see each original_protocol.json; benchmark-round runs "
                                                    "add their own code/ and records when uploaded"}}},
}
gen = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def addop(ops, path, data):
    if isinstance(data, Path):
        b = data.read_bytes()
    else:
        b = data.encode() if isinstance(data, str) else data
    if TOK.encode() in b:
        raise SystemExit(f"ABORT token in {path}")
    ops[path] = b


def git_sha1(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


plan_ops = {}
for repo in ("groot", "pi05", "psi0", "act", "deltaflow"):
    ops = {}
    addop(ops, "code/VERSIONS.json", json.dumps({"generated_utc": gen, "adapter_code": adapter, **VER[repo]}, indent=1) + "\n")
    tree = {f.path for f in api.list_repo_tree(f"humanoid-benchmark/{repo}", recursive=True) if hasattr(f, "size")}
    if repo in ("psi0", "act"):
        addop(ops, "code/hb_aug/hb_aug_v2.py", G / "hb_groot/hb_aug_v2.py")
    if repo == "act":
        for f in sorted((G / "hb_psi0").glob("*.py")):
            addop(ops, f"code/hb_psi0/{f.name}", f)
    if repo == "pi05":
        for td in TASKDIR.values():
            src = OUT / "pi05" / td / "train_config.json"
            if not src.is_file():
                continue
            for ep in sorted({p.split("/")[1] for p in tree if p.startswith(td + "/epoch-")}):
                addop(ops, f"{td}/{ep}/train_config.json", src)
    if repo == "psi0":
        for t in ("b6", "a3", "a7", "a18"):
            root = G / "psi0-runs" / t / "hb1110"
            run = next((c.parent for c in root.glob("train/*/*/checkpoints")), None)
            files = {"training_plan.json": root / "decision.json", "train.log": root / "logs/node-0.log",
                     "train_args.txt": root / "train_args.txt"}
            if run is not None and (run / "envs.txt").is_file():
                files["envs.txt"] = run / "envs.txt"
            for ep in ("epoch-010", "epoch-020"):
                if f"{TASKDIR[t]}/{ep}/model.safetensors" not in tree:
                    continue
                for n, src in files.items():
                    if src.is_file():
                        addop(ops, f"{TASKDIR[t]}/{ep}/{n}", src)
    plan_ops[repo] = ops
    print("PLAN", repo, len(ops), sorted(ops)[:6], "...")

if "--apply" not in sys.argv:
    sys.exit(0)
for repo, ops in plan_ops.items():
    rid = f"humanoid-benchmark/{repo}"
    c = api.create_commit(rid, operations=[CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in ops.items()],
                          commit_message="Add reproducibility records: VERSIONS.json, training plans/logs/configs, "
                                         "augmentation code")
    tree = {f.path: f for f in api.list_repo_tree(rid, recursive=True, expand=True) if hasattr(f, "size")}
    bad = []
    for k, v in ops.items():
        r = tree.get(k)
        if r is None or r.size != len(v) or (r.lfs is None and r.blob_id != git_sha1(v)) or \
                (r.lfs is not None and r.lfs.sha256 != hashlib.sha256(v).hexdigest()):
            bad.append(k)
    print("COMMIT", repo, c.oid, "files", len(ops), "bad", bad, "private", api.model_info(rid).private)
print("REPRO_DONE")
