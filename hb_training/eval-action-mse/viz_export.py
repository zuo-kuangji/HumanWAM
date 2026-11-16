"""Export every post-hoc result (results/*.json) + training loss of the same run + the in-loop JSON -> one compact JSON
for the comparison page.   python3 viz_export.py <inloop_extract.json> <out.json>   (stdlib only)"""
import glob
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent
INL, OUT = Path(sys.argv[1]), Path(sys.argv[2])
KEEP = {"groot", "psi0", "deltaflow", "fastwam"}

PROG = re.compile(r"(\d+)/(\d+) \[")
DICT = re.compile(r"\{'loss': ([0-9.eE+-]+)")
DFRE = re.compile(r"\[train\] epoch=\d+ step=(\d+)/(\d+) loss=([0-9.eE+-]+)")
FWRE = re.compile(r"step=(\d+)/(\d+)\s+(?:\S+\.py:\d+\s+)?loss=([0-9.eE+-]+)")


def thin(hist, n=300):
    hist = sorted(hist)
    if not hist:
        return []
    k = max(1, len(hist) // n)
    pts = hist[::k]
    if pts[-1] != hist[-1]:
        pts.append(hist[-1])
    return [[int(a), round(float(b), 5)] for a, b in pts]


def groot_loss(logdir):
    f = Path(logdir) / "production.log"
    if not f.is_file() or f.stat().st_size > 300 * 1024 * 1024:
        return []
    step, h = 0, {}
    for chunk in re.split(r"[\r\n]", f.read_text(errors="replace")):
        for m in PROG.finditer(chunk):
            step = int(m.group(1))
        m = DICT.search(chunk)
        if m and step:
            h[step] = float(m.group(1))
    return list(h.items())


def regex_loss(files, rx):
    h = {}
    for f in files:
        f = Path(f)
        if f.is_file() and f.stat().st_size < 300 * 1024 * 1024:
            for m in rx.finditer(f.read_text(errors="replace")):
                h[int(m.group(1))] = float(m.group(3))
    return list(h.items())


def psi_loss(pd):
    h = {}
    f = Path(pd) / "hb_metrics.jsonl"
    if f.is_file():
        for line in f.read_text(errors="replace").splitlines():
            try:
                d = json.loads(line)
            except Exception:
                continue
            if "train/loss" in d:
                h[int(d["step"])] = float(d["train/loss"])
    return list(h.items())


def find_loss(model, ckpt):
    p = Path(ckpt)
    try:
        if model == "groot":
            for a in [p] + list(p.parents)[:4]:
                if (a / "logs/production.log").is_file():
                    return groot_loss(a / "logs"), str(a)
        elif model == "psi0":
            for a in [p] + list(p.parents)[:6]:
                if (a / "hb_metrics.jsonl").is_file():
                    return psi_loss(a), str(a)
        elif model == "deltaflow":
            for a in [p] + list(p.parents)[:6]:
                if a.parent.name == "deltaflow-runs":
                    fs = sorted(a.glob("run/retry/**/node-0.log")) + sorted(a.glob("run/node-logs/*node-0*.log"))
                    return regex_loss(fs, DFRE), str(a)
        elif model == "fastwam":
            for a in [p] + list(p.parents)[:6]:
                if (a / "logs/node-0.log").is_file():
                    return regex_loss([a / "logs/node-0.log"], FWRE), str(a)
    except Exception as e:  # noqa: BLE001
        return [], f"error {e}"
    return [], None


post = []
losses = {}
for f in sorted(glob.glob(str(HERE / "results/*.json"))):
    name = Path(f).name[:-5]
    parts = name.split(".")
    model = parts[0]
    if model not in KEEP or len(parts) < 5:
        continue
    try:
        d = json.loads(Path(f).read_text())
    except Exception:
        continue
    if "nmse" not in d:
        continue
    step = int(parts[-2][1:]) if parts[-2].startswith("s") and parts[-2][1:].isdigit() else None
    run = ".".join(parts[:-2])  # model.variant.task
    ck = d.get("ckpt", "")
    if run not in losses:
        loss, root = find_loss(model, ck)
        losses[run] = {"root": root, "loss": thin(loss)}
    post.append({"id": name, "run": run, "model": model, "variant": ".".join(parts[1:-3]), "task": d.get("task", parts[-3]),
                 "split": d.get("split", parts[-1]), "step": step, "nmse": round(d["nmse"], 5),
                 "sem": d.get("nmse_episode_sem"), "groups": {g: round(v["nmse"], 5) for g, v in d.get("groups", {}).items()},
                 "by_step": [round(x, 5) for x in d.get("nmse_by_step", [])],
                 "baselines": {k: round(v["nmse"], 5) for k, v in d.get("baselines", {}).items()},
                 "queries": d.get("queries"), "s_per_query": d.get("infer_s_per_query"), "wall_s": d.get("wall_s"),
                 "finished": d.get("finished_utc")})

inl = json.loads(INL.read_text())
jobs = [l.split()[0] for l in (HERE / "jobs.txt").read_text().splitlines() if l.strip() and not l.startswith("#")]
done = {p["id"] for p in post}
OUT.write_text(json.dumps({"generated_utc": inl.get("generated_utc"), "post": post, "losses": losses,
                           "inloop": inl["runs"], "jobs_total": len(jobs),
                           "jobs_done": sum(1 for j in jobs if j in done),
                           "jobs_missing": [j for j in jobs if j not in done]}, separators=(",", ":")))
print("post", len(post), "runs", len(losses), "with loss", sum(1 for v in losses.values() if v["loss"]),
      "inloop", len(inl["runs"]), "jobs", len(jobs), "missing", len([j for j in jobs if j not in done]))
for r, v in losses.items():
    if not v["loss"]:
        print("NO LOSS", r, v["root"])
