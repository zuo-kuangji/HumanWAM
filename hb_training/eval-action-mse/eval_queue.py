"""Gap-filling evaluation queue: runs pending jobs on THIS node's GPUs only while no training owns the node.

Start (any node, stdlib only; one instance per node, a second one exits at once):
  setsid nohup python3 $G/eval-action-mse/eval_queue.py > $G/eval-action-mse/logs/queue-$(hostname).log 2>&1 &
Jobs: $G/eval-action-mse/jobs.txt, one per line:  <id> <model> <task> <split> <ckpt> [extra eval_run.py args]

When is the node free?  (v2, 2026-11-15)
  1. Relay state of this node ($G/relay/node-<n>.log + item-<k>-done-node-* + queue.txt), read every poll:
     * item K running             -> busy, unless the item script itself contains "eval_queue.py" (an idle-slot item
                                     such as hb_groot/groot_solo_*.sh: nodes without work in that item may evaluate);
     * item K ended, other nodes still on K (barrier wait)            -> may be free;
     * item K ended on all 4 nodes and item K+1 exists in queue.txt  -> busy (the relay is about to claim the GPUs);
     * item K ended on all nodes, queue exhausted                     -> may be free.
  2. GPU check BY GPU INDEX (nvidia-smi inside the DLC container reports host-namespace PIDs, so PIDs can never be
     matched to our own processes - v1 killed its own evals every poll): a GPU without an eval of ours that has any
     compute process, or a GPU with our eval and >1 compute process, means someone else is using the node.
  3. No training launcher / trainer process in "ps" (TRAIN_RE; the permanent relay / nodectl loops are not training).
  1-3 must hold for --free-min minutes before evals start. When any fails while evals run, our evals are killed at
  once (SIGTERM, then SIGKILL after 20 s) and their jobs go back to pending; a signal-killed eval never counts as a
  failure (rc -15/-9/143/137). One eval per GPU; a job is claimed by an atomic mkdir results/claims/<id>.
  done = results/<id>.json; 2 real failures -> results/<id>.failed (log kept in logs/<id>.log).
  $ROOT/PAUSE: launch nothing new;  $ROOT/STOP: kill our evals and exit.
Python reads this file once at start, so replacing it on NAS never breaks a running queue.
HB_EVAL_IGNORE_RELAY=1 (used by older idle-slot wrappers) skips the relay-state rule; GPU/ps checks stay.
"""
import argparse
import fcntl
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE
G = ROOT.parent
RELAY = G / "relay"
RES = ROOT / "results"
CLAIMS = RES / "claims"
LOGS = ROOT / "logs"
HOST = socket.gethostname()
IGNORE_RELAY = bool(os.environ.get("HB_EVAL_IGNORE_RELAY"))
TRAIN_RE = re.compile(r"(torchrun|deepspeed|accelerate(\s|-)launch|accelerate\.commands\.launch|launch_finetune|"
                      r"hb_launch\.py|scripts/train\.py|hb_train\.py|train_deltaflow\.py|_chain|_node\.sh|_task\.sh|"
                      r"hb_pool|_pool\w*\.sh|_single\w*\.sh|_quad\w*\.sh|_pair\w*\.sh|groot_probe|deltaflow_supervise|"
                      r"deltaflow_attempt|"
                      # any other process of a model environment (pool runners, prep, probes): the node is spoken for
                      r"envs/(groot-py312|openpi|psi0|dit4dit)/bin/python|/venv/bin/python)")
# permanent per-node loops that never mean "training is using the GPUs" (DLC entry scripts, relay/nodectl daemons,
# log tees); relay items are judged by the relay-state rule instead
IGNORE_BASE = (r"continuation-\d+-30d/(retry\d+_)?launch_node\.sh|^tee |^sleep |hb_relay\.sh|hb_nodectl\w*\.sh|"
               r"hb_sync_bai\.py|hb_autoupload\.py|fmdit_upload\.py")  # 11-16: master-0 hourly git-sync loop (runs the psi0 env python) is not training
FREE_KILL_RC = {-15, -9, 143, 137}
NODE_OF_HOST = (("master-0", 0), ("worker-0", 1), ("worker-1", 2), ("worker-2", 3))


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def node_index():
    if os.environ.get("RANK", "").isdigit():
        return int(os.environ["RANK"])
    return next((n for suf, n in NODE_OF_HOST if HOST.endswith(suf)), None)


def gpus():
    """-> {index: (uuid, mem_used_mib)}"""
    out = {}
    for line in sh("nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits").splitlines():
        i, u, m = [x.strip() for x in line.split(",")]
        out[int(i)] = (u, int(m))
    return out


def apps_per_gpu(g):
    """-> {gpu index: number of compute processes} (PIDs are host-namespace, unusable)"""
    uuid2idx = {u: i for i, (u, _) in g.items()}
    n = {i: 0 for i in g}
    for line in sh("nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) == 2 and p[1] in uuid2idx:
            n[uuid2idx[p[1]]] += 1
    return n


def queue_items():
    try:
        return [l.strip() for l in (RELAY / "queue.txt").read_text().splitlines()
                if l.strip() and not l.strip().startswith("#")]
    except OSError:
        return []


_EV_RE = re.compile(rb" item (\d+) (start|end): (\S+) (\S+)")
_relay_cache = {"off": 0, "ev": None, "size": -1}  # the relay log also carries training output (tens of MB):
                                                   # scan it once, then only the bytes appended since the last poll


def last_relay_event(log):
    size = log.stat().st_size
    c = _relay_cache
    if size < c["size"]:  # rotated / truncated
        c.update(off=0, ev=None)
    c["size"] = size
    with open(log, "rb") as f:
        f.seek(c["off"])
        while True:
            buf = f.read(8 << 20)
            if not buf:
                break
            cut = buf.rfind(b"\n") + 1  # a partial last line (relay still writing) is read again next poll
            if cut == 0:
                if len(buf) < (8 << 20):
                    break
                cut = len(buf)
            for m in _EV_RE.finditer(buf[:cut]):
                c["ev"] = tuple(x.decode("utf-8", "replace") for x in m.groups())
            c["off"] += cut
            f.seek(c["off"])
    return c["ev"]


def relay_state(node):
    """-> (state, detail, item_script_or_None); state in free / busy / idle-slot / none"""
    if node is None:
        return "none", "unknown node", None
    try:
        ev = last_relay_event(RELAY / f"node-{node}.log")
    except OSError:
        return "none", "no relay log", None
    if not ev:
        return "none", "no relay item yet", None
    k, kind, what, arg = int(ev[0]), ev[1], ev[2], ev[3]
    if kind == "start":
        script = arg if what == "cmd" else None
        try:
            idle_slot = script is not None and "eval_queue.py" in Path(script).read_text()
        except OSError:
            idle_slot = False
        if idle_slot:
            return "idle-slot", f"item {k} running ({Path(script).name}, idle-slot item)", script
        return "busy", f"item {k} running ({what} {Path(arg).name})", None
    if all((RELAY / f"item-{k}-done-node-{n}").exists() for n in range(4)):
        if len(queue_items()) > k + 1:
            return "busy", f"item {k} done on all nodes, item {k + 1} about to start", None
        return "free", f"item {k} done on all nodes, relay queue exhausted", None
    return "free", f"item {k} done here, barrier wait for other nodes", None


def descendants(pids):
    allp = set(pids)
    tree = {}
    for line in sh("ps -eo pid=,ppid=").splitlines():
        a = line.split()
        if len(a) == 2:
            tree.setdefault(int(a[1]), []).append(int(a[0]))
    stack = list(pids)
    while stack:
        for c in tree.get(stack.pop(), []):
            if c not in allp:
                allp.add(c)
                stack.append(c)
    return allp


def training_procs(ours, extra_ignore=None):
    ign = IGNORE_BASE + (r"|groot_solo_\w*\.sh|groot_resume_\w*\.sh" if IGNORE_RELAY else "")
    if extra_ignore:
        ign += "|" + re.escape(extra_ignore)
    ign_re = re.compile("(" + ign + ")")
    hits = []
    for line in sh("ps -eo pid=,args=").splitlines():
        line = line.strip()
        if not line:
            continue
        pid, _, args = line.partition(" ")
        if int(pid) in ours or "eval_queue.py" in args or "eval_run.py" in args or "run_eval.sh" in args \
                or "build_cache.py" in args or ign_re.search(args):
            continue
        if TRAIN_RE.search(args):
            hits.append(f"{pid} {args[:120]}")
    return hits


def load_jobs(path):
    jobs = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            p = line.split()
            jobs.append({"id": p[0], "model": p[1], "task": p[2], "split": p[3], "ckpt": p[4], "extra": p[5:]})
    return jobs


def pid_alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def release_stale_claims():
    for c in CLAIMS.glob("*"):
        try:
            info = json.loads((c / "owner.json").read_text())
        except (OSError, ValueError):
            try:  # another node may remove the claim between glob and stat
                old = time.time() - c.stat().st_mtime > 600
            except OSError:
                continue
            if old:
                shutil.rmtree(c, ignore_errors=True)
            continue
        if info.get("host") == HOST and not pid_alive(int(info.get("pid", -1))):
            shutil.rmtree(c, ignore_errors=True)


def claim(job):
    d = CLAIMS / job["id"]
    try:
        d.mkdir(parents=True)
    except FileExistsError:
        return False
    except OSError:
        return False
    (d / "owner.json").write_text(json.dumps({"host": HOST, "pid": 0, "t": time.time()}))
    return True


def kill(procs, reason):
    for gpu, (p, job) in list(procs.items()):
        print(f"{time.strftime('%F %T')} kill {job['id']} on gpu {gpu}: {reason}", flush=True)
        try:
            os.killpg(p.pid, signal.SIGTERM)
        except OSError:
            pass
    t = time.time()
    while time.time() - t < 20 and any(p.poll() is None for p, _ in procs.values()):
        time.sleep(1)
    for p, job in procs.values():
        if p.poll() is None:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except OSError:
                pass
        shutil.rmtree(CLAIMS / job["id"], ignore_errors=True)
    procs.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=str(ROOT / "jobs.txt"))
    ap.add_argument("--poll", type=int, default=15)
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--free-min", type=float, default=2.0,
                    help="all free conditions must have held this many minutes before evals start")
    ap.add_argument("--once", action="store_true", help="print the node state and the next jobs, launch nothing")
    a = ap.parse_args()
    for d in (RES, CLAIMS, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    if not a.once:
        lock = open("/tmp/hb_eval_queue.lock", "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"{time.strftime('%F %T')} another eval_queue already runs on {HOST} -> exit", flush=True)
            return
    node = node_index()
    procs = {}  # gpu -> (Popen, job)
    free_streak = 0
    last_reason = None
    print(f"{time.strftime('%F %T')} eval_queue v2 on {HOST} node {node} pid {os.getpid()} jobs {a.jobs} "
          f"ignore_relay={IGNORE_RELAY}", flush=True)
    while True:
        if (ROOT / "STOP").exists():
            kill(procs, "STOP file")
            (RES / f"RUNNING-{HOST}.json").unlink(missing_ok=True)
            print("STOP -> exit", flush=True)
            return
        # reap finished evals
        for gpu, (p, job) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[gpu]
            ok = rc == 0 and (RES / f"{job['id']}.json").exists()
            print(f"{time.strftime('%F %T')} {job['id']} gpu {gpu} rc {rc} "
                  f"{'OK' if ok else 'KILLED (requeued)' if rc in FREE_KILL_RC else 'FAILED'}", flush=True)
            if not ok and rc not in FREE_KILL_RC:
                fp = RES / f"{job['id']}.fails"
                n = int(fp.read_text()) + 1 if fp.exists() else 1
                fp.write_text(str(n))
                if n >= 2:
                    (RES / f"{job['id']}.failed").write_text(f"rc {rc} on {HOST}, see logs/{job['id']}.log\n")
            shutil.rmtree(CLAIMS / job["id"], ignore_errors=True)
        ours = descendants([p.pid for p, _ in procs.values()])
        g = gpus()
        napps = apps_per_gpu(g)
        foreign_gpus = [i for i in g if napps[i] > (1 if i in procs else 0)]
        if IGNORE_RELAY:
            rstate, rdetail, rscript = "idle-slot", "HB_EVAL_IGNORE_RELAY", None
        else:
            rstate, rdetail, rscript = relay_state(node)
        trainers = training_procs(ours, rscript)
        reasons = []
        if rstate == "busy":
            reasons.append(f"relay: {rdetail}")
        if foreign_gpus:
            reasons.append(f"other GPU processes on gpus {foreign_gpus}")
        if trainers:
            reasons.append(f"training processes {trainers[:2]}")
        busy = bool(reasons)
        if busy and procs:
            kill(procs, "; ".join(reasons))
        free_streak = 0 if busy else free_streak + 1
        state_line = "; ".join(reasons) if busy else f"free ({rdetail})"
        if state_line != last_reason and not a.once:
            print(f"{time.strftime('%F %T')} state: {state_line}", flush=True)
            last_reason = state_line
        jobs = load_jobs(a.jobs)
        pending = [j for j in jobs if not (RES / f"{j['id']}.json").exists() and not (RES / f"{j['id']}.failed").exists()
                   and not (CLAIMS / j["id"]).exists()]
        if a.once:
            print(json.dumps({"host": HOST, "node": node, "busy": busy, "reasons": reasons, "relay": [rstate, rdetail],
                              "gpu_apps": napps, "gpu_mem": {i: m for i, (_, m) in g.items()}, "jobs": len(jobs),
                              "done": sum((RES / f"{j['id']}.json").exists() for j in jobs),
                              "pending": [j["id"] for j in pending][:10]}, indent=1))
            return
        if not pending and not procs:
            if all((RES / f"{j['id']}.json").exists() or (RES / f"{j['id']}.failed").exists() for j in jobs):
                print(f"{time.strftime('%F %T')} all {len(jobs)} jobs done/failed -> exit", flush=True)
                (RES / f"RUNNING-{HOST}.json").unlink(missing_ok=True)
                return
        release_stale_claims()
        status = RES / f"RUNNING-{HOST}.json"  # visible marker for the training scheduler: these GPUs yield at once
        if procs:
            status.write_text(json.dumps({"host": HOST, "queue_pid": os.getpid(), "note": "open-loop action-MSE evals; "
                                          "they are killed automatically within one poll when training starts here "
                                          "(or touch eval-action-mse/PAUSE / STOP)",
                                          "gpus": {g: j["id"] for g, (_, j) in procs.items()}}, indent=1))
        elif status.exists():
            status.unlink()
        need = max(2, int(a.free_min * 60 / a.poll + 0.999))
        if free_streak >= need and not (ROOT / "PAUSE").exists():
            for gpu in sorted(g)[: a.max_gpus]:
                if gpu in procs or g[gpu][1] > 2000 or napps[gpu] or not pending:
                    continue
                job = pending.pop(0)
                if not claim(job):
                    continue
                log = open(LOGS / f"{job['id']}.log", "a")
                log.write(f"\n=== {time.strftime('%F %T')} {HOST} gpu {gpu}\n")
                log.flush()
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                cmd = ["bash", str(ROOT / "run_eval.sh"), job["model"], job["task"], job["split"], job["ckpt"],
                       str(RES / f"{job['id']}.json"), "--device", "cuda:0", *job["extra"]]
                p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
                log.close()
                (CLAIMS / job["id"] / "owner.json").write_text(json.dumps({"host": HOST, "pid": p.pid, "gpu": gpu,
                                                                           "t": time.time()}))
                procs[gpu] = (p, job)
                print(f"{time.strftime('%F %T')} start {job['id']} on gpu {gpu} pid {p.pid}", flush=True)
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
