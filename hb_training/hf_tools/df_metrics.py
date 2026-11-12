"""Parse ImageWAM / Delta-flow trainer logs (rich console format) into a loss table.

A record starts at a line containing "epoch=E step=S/T" and continues over the following indented lines; every
key=value number on those lines belongs to the record (loss, loss components, lr, speed ...). Later files override
earlier ones for the same step (resumed attempts).
    python df_metrics.py <out_dir> <log> [<log> ...]   ->  <out_dir>/metrics.csv, <out_dir>/loss_summary.json
"""
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

STEP = re.compile(r"epoch=(\d+)\s+step=(\d+)/(\d+)")
KV = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)=(-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)\b")
NEW_RECORD = re.compile(r"^\S|\]\s+(INFO|WARNING|ERROR|DEBUG)\b")


def parse(paths):
    rows = {}
    for path in paths:
        cur = None
        for line in Path(path).read_text(errors="replace").replace("\r", "\n").splitlines():
            m = STEP.search(line)
            if m:
                cur = {"epoch": int(m.group(1)), "step": int(m.group(2)), "total_steps": int(m.group(3))}
                rows[cur["step"]] = cur
                tail = line[m.end():]
            elif cur is not None and line.strip() and not NEW_RECORD.search(line):
                tail = line
            else:
                cur = None
                continue
            for k, v in KV.findall(tail):
                if k not in ("epoch", "step"):
                    cur[k] = float(v)
    return [rows[s] for s in sorted(rows)]


def write(out_dir, rows, sources):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["step", "epoch", "total_steps"] + sorted({k for r in rows for k in r} - {"step", "epoch", "total_steps"})
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    per = defaultdict(list)
    for r in rows:
        if "loss" in r:
            per[r["epoch"]].append(r["loss"])
    summary = dict(
        note="Parsed from the node-0 trainer log(s); loss = total training loss as printed by the trainer every "
             "logged step (other columns of metrics.csv: loss components, lr, speed).",
        source_logs=[str(Path(s).name) for s in sources], logged_steps=len(rows),
        last_step=rows[-1]["step"] if rows else None, total_steps=rows[-1]["total_steps"] if rows else None,
        first_loss=next((r["loss"] for r in rows if "loss" in r), None),
        last_loss=next((r["loss"] for r in reversed(rows) if "loss" in r), None),
        loss_mean_per_epoch={str(e): round(sum(v) / len(v), 5) for e, v in sorted(per.items())},
        loss_min_per_epoch={str(e): round(min(v), 5) for e, v in sorted(per.items())})
    (out / "loss_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    logs = sys.argv[2:]
    s = write(sys.argv[1], parse(logs), logs)
    print(json.dumps({k: s[k] for k in ("logged_steps", "last_step", "total_steps", "first_loss", "last_loss")}))
