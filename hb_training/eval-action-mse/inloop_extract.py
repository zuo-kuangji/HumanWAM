"""Collect every in-training MSE record (inloop-mse.jsonl) together with the same run's training loss -> one JSON.
    python3 inloop_extract.py out.json        (stdlib only)
GR00T loss: node-0 production log (HF dict lines, step = last tqdm position); Psi-Zero: hb_metrics.jsonl train/loss;
Delta Flow: "[train] ... step=N/M loss=X" lines of the node-0 logs; FastWAM: the same in fastwam/runs/<task>/logs/node-0.log.
"""
import glob
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent
OUT = Path(sys.argv[1])


def thin(hist, n=400):
    hist = sorted(hist)
    if not hist:
        return []
    k = max(1, len(hist) // n)
    pts = hist[::k]
    if pts[-1] != hist[-1]:
        pts.append(hist[-1])
    return [[int(a), round(float(b), 5)] for a, b in pts]


def read_jsonl(p):
    out = []
    for line in open(p, errors="replace"):
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def baselines(task, split):
    try:
        m = json.loads((HERE / "cache" / task / split / "meta.json").read_text())
        return {k: {"nmse": v["nmse"], "groups": {g: x["nmse"] for g, x in v["groups"].items()}}
                for k, v in m.get("baselines", {}).items()}
    except Exception:
        return {}


PROG = re.compile(r"(\d+)/(\d+) \[")
DICT = re.compile(r"\{'loss': ([0-9.eE+-]+)")


def groot_loss(logdir):
    f = Path(logdir) / "production.log"
    if not f.is_file():
        return [], None
    step, total, h = 0, None, {}
    for chunk in re.split(r"[\r\n]", f.read_text(errors="replace")):
        for m in PROG.finditer(chunk):
            step, total = int(m.group(1)), int(m.group(2))
        m = DICT.search(chunk)
        if m and step:
            h[step] = float(m.group(1))
    return list(h.items()), total


DFRE = re.compile(r"\[train\] epoch=\d+ step=(\d+)/(\d+) loss=([0-9.eE+-]+)")


def df_loss(R):
    logs = sorted(Path(R).glob("run/retry/**/node-0.log")) + sorted(Path(R).glob("run/node-logs/*node-0*.log"))
    h, total = {}, None
    for f in logs:
        if f.stat().st_size > 200 * 1024 * 1024:
            continue
        for m in DFRE.finditer(f.read_text(errors="replace")):
            h[int(m.group(1))] = float(m.group(3))
            total = int(m.group(2))
    return list(h.items()), total


FWRE = re.compile(r"step=(\d+)/(\d+)\s+(?:\S+\.py:\d+\s+)?loss=([0-9.eE+-]+)")


def fw_loss(R):
    """FastWAM (2026-11-16): rich-logged "[train] epoch=E step=N/M loss=X ..." lines of logs/node-0.log"""
    f = Path(R) / "logs/node-0.log"
    h, total = {}, None
    if f.is_file() and f.stat().st_size < 200 * 1024 * 1024:
        for m in FWRE.finditer(f.read_text(errors="replace")):
            h[int(m.group(1))] = float(m.group(3))
            total = int(m.group(2))
    return list(h.items()), total


def psi_loss(project_dir):
    f = Path(project_dir) / "hb_metrics.jsonl"
    h = {}
    if f.is_file():
        for d in read_jsonl(f):
            if "train/loss" in d:
                h[int(d["step"])] = float(d["train/loss"])
    return list(h.items())


runs = []


def add(model, run, jl, loss, total, files):
    recs = read_jsonl(jl)
    ok = [r for r in recs if "nmse" in r]
    errs = [r for r in recs if "error" in r]
    if not recs:
        return
    task = ok[0]["task"] if ok else None
    split = ok[0]["split"] if ok else None
    mse = [{"step": r["step"], "nmse": round(r["nmse"], 5),
            "groups": {g: round(v["nmse"], 5) for g, v in r["groups"].items()},
            "by_step": [round(x, 5) for x in r.get("nmse_by_step", [])],
            "sem": r.get("nmse_episode_sem"), "queries": r["queries"], "wall_s": r.get("wall_s")} for r in ok]
    ages = [Path(p).stat().st_mtime for p in files if Path(p).exists()]
    runs.append(dict(model=model, run=run, task=task, split=split, total_steps=total, mse=mse,
                     errors=[e.get("error") for e in errs], loss=thin(loss), baselines=baselines(task, split) if task else {},
                     updated_min=round((time.time() - max(ages)) / 60, 1) if ages else None))


for jl in sorted(glob.glob(str(G / "groot-runs-*/*/logs/inloop-mse.jsonl"))):
    logdir = Path(jl).parent
    loss, total = groot_loss(logdir)
    root = logdir.parent
    add("groot", f"{root.parent.name}/{root.name}", jl, loss, total, [jl, logdir / "production.log"])
for jl in sorted(glob.glob(str(G / "psi0-runs/*/*/train/*/*/inloop-mse.jsonl"))):
    pd = Path(jl).parent
    cfg = pd.parents[2]
    dec = cfg / "decision.json"
    total = json.loads(dec.read_text()).get("steps") if dec.is_file() else None
    add("psi0", f"{cfg.parent.name}/{cfg.name}", jl, psi_loss(pd), total, [jl, pd / "hb_metrics.jsonl"])
for jl in sorted(glob.glob(str(G / "deltaflow-runs/*/run/production/inloop-mse.jsonl"))):
    R = Path(jl).parents[2]
    loss, total = df_loss(R)
    add("deltaflow", R.name, jl, loss, total, [jl])

for jl in sorted(glob.glob(str(G / "fastwam/runs/*/out/inloop-mse.jsonl"))):
    R = Path(jl).parents[1]
    loss, total = fw_loss(R)
    add("fastwam", R.name, jl, loss, total, [jl, R / "logs/node-0.log"])

OUT.write_text(json.dumps({"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runs": runs}))
for r in runs:
    print(r["model"], r["run"], r["task"], r["split"], "mse pts", len(r["mse"]), "loss pts", len(r["loss"]),
          "last", r["mse"][-1]["step"] if r["mse"] else None, r["mse"][-1]["nmse"] if r["mse"] else None,
          "errors", r["errors"][:1])
