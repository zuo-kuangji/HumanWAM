"""Sync the live Humanoid Benchmark adapter code into HumanWAM (branch bai) and push it.

    python hb_sync_bai.py "<commit message>"        (or with no message: generic sync message)

Source = the code actually used for training on the NAS ($G/hb_*, relay / pool / node-control scripts, task pools,
relay queue, top-level preparation scripts, HF upload tools); destination = $P/HumanWAM-bai/hb_training/.
Only code / config / small text files are copied (no logs, data, checkpoints, backups, caches). Files removed from the
live tree are removed from hb_training too (within the synced directories). Aborts if anything looks like a secret.
Commits as the repository's configured user (B1ue-Jay) and pushes origin bai with the repository's deploy key.
"""
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

P = Path("/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z")
G = P / "policies-20261109"
R = P / "HumanWAM-bai"
T = R / "hb_training"
CODE_EXT = (".py", ".sh", ".md", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg")
SKIP = ("__pycache__", ".bak*", "*.bak*", "*.log", "*.tmp", "*.pyc", "probe-*.env")
MAX_BYTES = 2 * 1024 * 1024
DIRS = ["hb_groot", "hb_pi05", "hb_psi0", "hb_act", "hb_deltaflow", "hb_dit4dit", "hb_b256", "hb_b512", "hb_b7"]
TOP = ["hb_relay.sh", "hb_nodectl.sh", "hb_pool.sh", "hb_pool2.sh", "hb_pool_b696.sh", "hb_sync_bai.py"]
OPS_GLOBS = ["*_2026111*.sh", "dl_*.py", "deltaflow_prep_and_queue.sh", "groot_env.sh", "groot_prep_run.sh",
             "download_*.py", "fetch_public.py"]
TOOLS = P / "hf-sidecar-tools-20261110"
SECRET = re.compile(rb"hf_[A-Za-z0-9]{30,}|ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|"
                    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}")


def skip(name):
    return any(fnmatch.fnmatch(name, p) for p in SKIP)


def wanted(p):
    return p.is_file() and p.suffix in CODE_EXT and not skip(p.name) and p.stat().st_size <= MAX_BYTES


def sync_dir(src, dst, protect=()):
    keep = set()
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if not skip(d) and not d.startswith(".")]
        for f in files:
            s = Path(root) / f
            if wanted(s):
                rel = s.relative_to(src)
                keep.add(rel)
                d = dst / rel
                d.parent.mkdir(parents=True, exist_ok=True)
                if not d.exists() or d.read_bytes() != s.read_bytes():
                    shutil.copy2(s, d)
    if dst.exists():
        for d in list(dst.rglob("*")):
            if (d.is_file() and d.relative_to(dst) not in keep and d.suffix in CODE_EXT
                    and not str(d.relative_to(dst)).startswith(protect)):
                d.unlink()


def copy(src, dst):
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or dst.read_bytes() != src.read_bytes():
            shutil.copy2(src, dst)


def git(*a, check=True):
    return subprocess.run(["git", "-C", str(R), *a], check=check, capture_output=True, text=True)


def main():
    msg = sys.argv[1] if len(sys.argv) > 1 else f"hb_training: sync live adapter code ({time.strftime('%Y-%m-%d %H:%M')})"
    if git("branch", "--show-current").stdout.strip() != "bai":
        sys.exit("HumanWAM-bai is not on branch bai")
    git("fetch", "-q", "origin", "bai")
    git("merge", "--ff-only", "-q", "origin/bai")
    for d in DIRS:
        if (G / d).is_dir():
            sync_dir(G / d, T / d, protect=("code/",) if d == "hb_deltaflow" else ())
    # Delta Flow: snapshot of the code trees the runs actually import (built by prepare_deltaflow*.py)
    for sub, tree in (("g1", G / "deltaflow-runs/HumanWAM"), ("h2", G / "deltaflow-runs/HumanWAM-h2")):
        rec = tree / "deltaflow-code.json"
        if not rec.is_file():
            continue
        dst = T / "hb_deltaflow/code" / sub
        rels = set(json.loads(rec.read_text())["sha256"]) | {"deltaflow-code.json"}
        rels |= {str(x.relative_to(dst)) for x in dst.rglob("*") if x.is_file()} if dst.exists() else set()
        for rel in sorted(rels):
            copy(tree / rel, dst / rel)
    for f in TOP:
        copy(G / f, T / f)
    copy(G / "relay" / "queue.txt", T / "relay_queue.txt")
    for jobs in sorted(G.glob("pools/*/jobs.txt")):
        copy(jobs, T / "pools" / jobs.parent.name / "jobs.txt")
    ops = sorted({p for g in OPS_GLOBS for p in G.glob(g) if wanted(p)})
    for p in ops:
        copy(p, T / "ops" / p.name)
    for p in sorted(TOOLS.glob("*.py")):
        if wanted(p):
            copy(p, T / "hf_tools" / p.name)
    bad = [str(p.relative_to(R)) for p in T.rglob("*") if p.is_file() and SECRET.search(p.read_bytes())]
    if bad:
        sys.exit(f"ABORT: secret-like content in {bad}")
    git("add", "-A", "-f", "hb_training")  # -f: task-pool jobs.txt files match the repo-wide *.txt ignore rule
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("NOTHING_TO_COMMIT")
        return
    stat = git("diff", "--cached", "--shortstat").stdout.strip()
    git("commit", "-q", "-m", msg, "-m", "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>")
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    git("push", "-q", "origin", "bai")
    remote = git("rev-parse", "--short", "origin/bai").stdout.strip()
    print("PUSHED", head, "origin/bai", remote, "|", stat)


if __name__ == "__main__":
    main()
