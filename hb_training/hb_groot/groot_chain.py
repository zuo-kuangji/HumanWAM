"""Per-node GR00T relay: batch-size probe once, then 20-epoch production for each task in order.

Run on every node (nohup), node rank from the DLC env RANK:  python groot_chain.py [task ...]
Default tasks: b6 basket a3 a7 a18. All nodes make the same decisions from the shared exit files on NAS, so no
extra coordination is needed: a phase succeeds only if all 4 node launchers exit 0.
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
RUNS = G / "groot-runs"
NODE = int(os.environ["RANK"])
TASKS = sys.argv[1:] or ["b6", "a3", "a7", "a18"]  # basket postponed (user plan 2026-11-10)
CANDIDATES = [64, 48, 40, 32, 24, 16, 12, 8]
EPOCHS = 20
PORT_BASE = int(os.environ.get("HB_GROOT_PORT_BASE", "29800"))  # probe PORT_BASE+i, production PORT_BASE+50+k
RUNS.mkdir(parents=True, exist_ok=True)
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


decision = RUNS / "batch-decision.json"
if decision.is_file():
    bs = json.loads(decision.read_text())["per_gpu_batch"]
else:
    bs = None
    for i, cand in enumerate(CANDIDATES):
        if run_phase(TASKS[0], f"probe-bs{cand}", cand, 4, 1000000, PORT_BASE + i):
            bs = cand
            break
        time.sleep(20)
    if bs is None:
        log("no batch size fit; stopping")
        sys.exit(1)
    if NODE == 0:
        decision.write_text(json.dumps({"per_gpu_batch": bs, "global_batch": bs * 32, "probe_task": TASKS[0],
                                        "candidates": CANDIDATES}) + "\n")
log(f"per-GPU batch {bs} (global {bs * 32})")
for k, task in enumerate(TASKS):
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
