"""Per-node GR00T relay: per-layout batch-size probe (largest that fits), then 20-epoch production per task.

Run on every node (nohup), node rank from the DLC env RANK:  python groot_chain.py [task ...]
Default tasks: b6 a3 a7 a18 (basket postponed). A phase succeeds only if all 4 node launchers exit 0.

Batch (user rule: as large as memory allows, 2026-11-10): descending CANDIDATES, 4 steps each on all 32 GPUs; a
candidate fits when all launchers exit 0 AND the peak reserved memory of every rank (written by hb_launch.py to
logs/peak-<phase>-rank<r>.json) is <= MAX_FRAC of the GPU. Node 0 alone judges each probe and writes a verdict file
the other nodes wait for, so all nodes take the same decision. One decision per robot layout (g1 / h2).
Outputs: $G/groot-runs-bmax/ (the earlier per-GPU-64 runs stay in $G/groot-runs/).
"""
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
G = H.parent
RUNS = G / os.environ.get("HB_GROOT_RUNS_NAME", "groot-runs-bmax")
NODE = int(os.environ["RANK"])
TASKS = sys.argv[1:] or ["b6", "a3", "a7", "a18"]  # basket postponed (user plan 2026-11-10)
CANDIDATES = [320, 288, 256, 224, 192, 160, 128, 96, 64, 48, 32]
MAX_FRAC = 0.92
EPOCHS = 20
PORT_BASE = int(os.environ.get("HB_GROOT_PORT_BASE", "32500"))  # probe PORT_BASE+i, production PORT_BASE+50+k
RUNS.mkdir(parents=True, exist_ok=True)
os.environ["HB_GROOT_RUNS"] = str(RUNS)  # read by groot_node.sh
lock = (RUNS / f"chain-node-{NODE}.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} node{NODE} {msg}"
    print(line, flush=True)
    with (RUNS / f"chain-node-{NODE}.log").open("a") as f:
        f.write(line + "\n")


def run_phase(task, phase, bs, steps, save, port):
    logs = RUNS / task / "logs"
    exits = [logs / f"{phase}-node-{r}.exit" for r in range(4)]
    if all(p.is_file() for p in exits):
        return all(p.read_text().strip() == "0" for p in exits)
    subprocess.run(["bash", str(H / "groot_node.sh"), task, phase, str(bs), str(steps), str(save), str(port)])
    deadline = time.monotonic() + 600
    while not all(p.is_file() for p in exits):
        if time.monotonic() > deadline:
            log(f"{task}/{phase}: timed out waiting for peer exit files")
            return False
        time.sleep(5)
    ok = all(p.read_text().strip() == "0" for p in exits)
    log(f"{task}/{phase}: bs={bs} steps={steps} -> {'OK' if ok else 'FAIL'}")
    return ok


def peak_frac(task, phase):
    """Max over ranks of peak reserved / total; None when no rank reported."""
    fracs = []
    for f in (RUNS / task / "logs").glob(f"peak-{phase}-rank*.json"):
        try:
            d = json.loads(f.read_text())
            fracs.append(d["peak_reserved_gb"] / d["total_gb"])
        except (ValueError, KeyError, OSError):
            continue
    return (max(fracs), len(fracs)) if fracs else (None, 0)


def probe_verdict(task, cand, port):
    phase = f"probe-bs{cand}"
    verdict = RUNS / task / "logs" / f"{phase}.verdict.json"
    ok = run_phase(task, phase, cand, 4, 1000000, port)
    if NODE == 0 and not verdict.is_file():
        time.sleep(10)  # let peak files from the other nodes become visible on NAS
        frac, n = peak_frac(task, phase) if ok else (None, 0)
        fits = bool(ok and frac is not None and frac <= MAX_FRAC)
        verdict.write_text(json.dumps({"per_gpu_batch": cand, "ran_ok": ok, "peak_frac": frac, "ranks_reported": n,
                                       "max_frac": MAX_FRAC, "fits": fits}) + "\n")
    deadline = time.monotonic() + 600
    while not verdict.is_file():
        if time.monotonic() > deadline:
            log(f"{task}/{phase}: no verdict from node 0")
            return False
        time.sleep(5)
    v = json.loads(verdict.read_text())
    log(f"{task}/{phase}: verdict {v}")
    return v["fits"]


def prune_resume_state(out):
    """Keep weights of every checkpoint but resume state (DeepSpeed optimizer shards, scheduler, RNG) only in the latest."""
    ckpts = sorted((p for p in out.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
                   key=lambda p: int(p.name.split("-")[-1]))
    for c in ckpts[:-1]:
        for p in list(c.glob("global_step*")) + list(c.glob("optimizer*.pt")) + list(c.glob("scheduler.pt")) \
                + list(c.glob("rng_state*.pth")) + list(c.glob("latest")) + list(c.glob("zero_to_fp32.py")):
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink(missing_ok=True)
        log(f"{c.name}: resume state dropped (weights kept)")


def batch_for(task):
    layout = "h2" if task in ("a17", "a18") else "g1"
    decision = RUNS / f"batch-decision-{layout}.json"
    if decision.is_file():
        return json.loads(decision.read_text())["per_gpu_batch"]
    for i, cand in enumerate(CANDIDATES):
        if probe_verdict(task, cand, PORT_BASE + i + (20 if layout == "h2" else 0)):
            if NODE == 0:
                decision.write_text(json.dumps({"per_gpu_batch": cand, "global_batch": cand * 32, "probe_task": task,
                                                "layout": layout, "candidates": CANDIDATES,
                                                "max_frac": MAX_FRAC}) + "\n")
            return cand
        time.sleep(20)
    log(f"{layout}: no batch size fit; stopping")
    sys.exit(1)


for k, task in enumerate(TASKS):
    bs = batch_for(task)
    log(f"{task}: per-GPU batch {bs} (global {bs * 32})")
    receipt = json.loads((G / "groot-data" / task / "hb-dataset-receipt.json").read_text())
    steps = math.ceil(EPOCHS * receipt["frames"] / (bs * 32))
    save = math.ceil(steps / 2)
    if not run_phase(task, "production", bs, steps, save, PORT_BASE + 50 + k):
        log(f"{task}: production failed; stopping chain")
        sys.exit(1)
    if NODE == 0:
        prune_resume_state(RUNS / task / "production")
        (RUNS / task / "production-complete.json").write_text(json.dumps(
            {"task": task, "frames": receipt["frames"], "epochs": EPOCHS, "per_gpu_batch": bs,
             "global_batch": bs * 32, "max_steps": steps, "save_steps": save}) + "\n")
log("all tasks done")
