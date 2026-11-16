"""Scan the training run dirs and write jobs.txt for eval_queue.py (existing results are kept; ids are stable).

Usage: python3 make_jobs.py [--all-steps] [--models groot,pi05,psi0,act,deltaflow,dit4dit] [--out jobs.txt]
Per checkpoint: one job per cached split of its task ("heldout" first where recordings were never trained on, then
"train10"). Default = the FINAL checkpoint of every run (GR00T: every saved step, they are the 2000..4000 sweep).
Skipped on purpose: Delta Flow variants with another input protocol (*.velocity-*, *.delay-*, *.2cam-*, failed runs),
GR00T *.fail-* dirs, probes. Order: B6 held-out of every model first, then the rest grouped by task.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_eval_common as E  # noqa: E402

G = E.G
H2_3CAM_SINCE = time.mktime((2026, 11, 13, 13, 30, 0, 0, 0, 0))  # hb_pi05 switched H2 to head+chest+wrist


def num(p, pat):
    m = re.search(pat, p.name)
    return int(m.group(1)) if m else -1


def groot(all_steps):
    for root in sorted(G.glob("groot-runs*")):
        for t in sorted(root.iterdir()):
            if not t.is_dir() or t.name not in E.TASKS:
                continue
            cks = sorted((t / "production").glob("checkpoint-*"), key=lambda p: num(p, r"(\d+)$"))
            for c in cks:  # every saved step (2000..4000 sweep)
                yield "groot", root.name.replace("groot-runs", "g") or "g", t.name, c, num(c, r"(\d+)$"), []


def pi05(all_steps):
    for c in sorted(G.glob("pi05-runs/*/*/run/ckpt/*/run")):
        cfg, task = c.parts[-6], c.parts[-5]
        if task not in E.TASKS:
            continue
        steps = sorted([p for p in c.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name))
        for s in (steps if all_steps else steps[-1:]):
            extra = []
            if E.TASKS[task][2] == "h2" and (s / "model.safetensors").stat().st_mtime < H2_3CAM_SINCE:
                extra = ["--opt", "cams=head,chest"]
            yield "pi05", cfg, task, s, int(s.name), extra


def psi_like(model, pattern):
    for run in sorted(G.glob(pattern)):
        parts = run.parts
        i = [k for k, x in enumerate(parts) if x in E.TASKS]
        if not i:
            continue
        task = parts[i[-1]]
        cfg = parts[i[-1] + 1] if model == "psi0" else parts[i[-1] - 1].replace("act-runs-", "")
        cks = sorted((run / "checkpoints").glob("ckpt_*"), key=lambda p: num(p, r"_(\d+)$"))
        if not cks and model == "psi0":
            cks = sorted((run.parents[2] / "preserved").glob("ckpt_*"), key=lambda p: num(p, r"_(\d+)$"))
        yield from ((model, cfg, task, c, num(c, r"_(\d+)$"), []) for c in cks)


def psi0(all_steps):
    rows = list(psi_like("psi0", "psi0-runs/*/*/train/finetune/*"))
    yield from last_per_run(rows, all_steps)


def act(all_steps):
    rows = list(psi_like("act", "act-runs-*/*/train/act-g1/*"))
    yield from last_per_run(rows, all_steps)


def last_per_run(rows, all_steps):
    if all_steps:
        yield from rows
        return
    best = {}
    for r in rows:
        k = (r[0], r[1], r[2], str(r[3].parent))
        if k not in best or r[4] > best[k][4]:
            best[k] = r
    yield from best.values()


def deltaflow(all_steps):
    for t in sorted((G / "deltaflow-runs").iterdir()):
        if t.name not in E.TASKS:  # variants (*.velocity-*, *.delay-*, *.2cam-*, failed) are other protocols
            continue
        ws = sorted((t / "run/production/checkpoints/weights").glob("step_*.pt"), key=lambda p: num(p, r"(\d+)"))
        for w in (ws if all_steps else ws[-1:]):
            yield "deltaflow", "main", t.name, w, num(w, r"(\d+)"), []


def dit4dit(all_steps):
    rows = []
    for c in sorted(G.glob("dit4dit-runs/*/*/run/checkpoints/*.pt")):
        task, chain = c.parts[-4], c.parts[-5]
        if task in E.TASKS:
            rows.append(("dit4dit", chain, task, c, num(c, r"(\d+)"), []))
    yield from last_per_run(rows, all_steps)


def fastwam(all_steps):
    # 2026-11-16: every saved weights file (2000, 4000) of $G/fastwam/runs/<task>/out
    for t in sorted((G / "fastwam/runs").iterdir()) if (G / "fastwam/runs").is_dir() else []:
        if t.name not in E.TASKS:
            continue
        ws = sorted((t / "out/checkpoints/weights").glob("step_*.pt"), key=lambda p: num(p, r"(\d+)"))
        for w in ws:
            yield "fastwam", "main", t.name, w, num(w, r"(\d+)"), []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-steps", action="store_true")
    ap.add_argument("--models", default="fastwam,groot,pi05,psi0,act,deltaflow,dit4dit")
    ap.add_argument("--out", default=str(E.ROOT / "jobs.txt"))
    a = ap.parse_args()
    fns = {"groot": groot, "pi05": pi05, "psi0": psi0, "act": act, "deltaflow": deltaflow, "dit4dit": dit4dit,
           "fastwam": fastwam}
    jobs = []
    for m in a.models.split(","):
        for model, cfg, task, ck, step, extra in fns[m](a.all_steps):
            for split in ("heldout", "train10"):
                if not (E.split_dir(task, split) / "meta.json").exists():
                    continue
                jid = f"{model}.{cfg}.{task}.s{step}.{split}"
                jobs.append((-1 if model == "fastwam" else 0 if (task == "b6" and split == "heldout") else 1, task, model, jid,
                             f"{jid} {model} {task} {split} {ck} {' '.join(extra)}".strip()))
    jobs.sort(key=lambda j: (j[0], j[1], j[2], j[3]))
    hdr = (f"# generated {time.strftime('%F %T')} by make_jobs.py ({len(jobs)} jobs)\n"
           "# id model task split ckpt [extra eval_run.py args]\n")
    Path(a.out).write_text(hdr + "\n".join(j[4] for j in jobs) + "\n")
    print(f"{len(jobs)} jobs -> {a.out}")
    for j in jobs[:8]:
        print("  ", j[4][:160])


if __name__ == "__main__":
    main()
