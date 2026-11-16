"""Cross-model comparison on one task (2026-11-16): every post-hoc result (results/*.json) and every in-training record
(inloop_extract output) of <task>, per model / run / step, with the cache baselines. stdlib only.
  python3 compare_task.py <task> <inloop_extract.json> [out.json]"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
task, ilp = sys.argv[1], Path(sys.argv[2])
rows = []
for f in sorted((HERE / "results").glob("*.json")):
    try:
        r = json.loads(f.read_text())
    except Exception:
        continue
    if r.get("task") != task or "nmse" not in r:
        continue
    step = int(f.stem.split(".s")[-1].split(".")[0]) if ".s" in f.stem else None
    rows.append(dict(src="post-hoc", model=r["model"], run=f.stem.split(".")[1], split=r["split"], step=step,
                     nmse=round(r["nmse"], 4), groups={g: round(v["nmse"], 4) for g, v in r["groups"].items()},
                     queries=r["queries"], s_per_query=r.get("infer_s_per_query"), id=f.stem))
base = {}
for run in json.loads(ilp.read_text())["runs"]:
    if run["task"] != task:
        continue
    base = base or run.get("baselines", {})
    for m in run["mse"]:
        rows.append(dict(src="in-training", model=run["model"], run=run["run"], split=run["split"], step=m["step"],
                         nmse=round(m["nmse"], 4), groups={g: round(v, 4) for g, v in m["groups"].items()}, queries=m["queries"]))
if not base:
    try:
        base = json.loads((HERE / "cache" / task / "train10" / "meta.json").read_text()).get("baselines", {})
    except Exception:
        pass
rows.sort(key=lambda r: (r["model"], r["run"], r["split"], r["step"] or 0, r["src"]))
print(f"task {task}: baselines", {k: round(v["nmse"], 4) for k, v in base.items()})
for r in rows:
    if r["src"] == "post-hoc" or r["step"] in (2000, 4000) or r["step"] == max(x["step"] for x in rows if x["run"] == r["run"]):
        print(f'{r["model"]:9s} {r["run"]:28s} {r["split"]:8s} step {r["step"]!s:>5} {r["src"]:11s} nMSE {r["nmse"]:.4f}  {r["groups"]}')
if len(sys.argv) > 3:
    Path(sys.argv[3]).write_text(json.dumps({"task": task, "baselines": base, "rows": rows}, indent=1))
