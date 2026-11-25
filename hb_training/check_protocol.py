#!/usr/bin/env python3
"""Check a finished (or running) Humanoid Benchmark training against hb_training/protocol.yaml.

Reads only what the run itself saved, so the verdict reflects what was actually trained:
  groot    <run root>   e.g. $G/groot-runs-2x32/<task>      production/checkpoint-*/training_args.bin, production-complete.json
  psi0     <run root>   e.g. $G/psi0-runs/<task>/b512        train_args.txt, decision.json
  fastwam  <run dir>    e.g. $G/fastwam/runs/<task>[-cont2k] out/config.yaml, complete.json
Every line is PASS / FAIL / WARN; exit code 1 when anything FAILs. A FAIL that is a registered exception
(protocol.yaml `exceptions`, REPRODUCIBILITY.md section 7) is acceptable only if it is listed there.

  python3 hb_training/check_protocol.py groot $G/groot-runs-2x32/b7t
  python3 hb_training/check_protocol.py psi0 $G/psi0-runs/b7t/b512
  python3 hb_training/check_protocol.py fastwam $G/fastwam/runs/b1-cont2k
GR00T needs torch + transformers to read training_args.bin; without them the script re-runs that part with
$G/envs/groot-py312/bin/python when it can find it.
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = []


def say(status, key, got, want=None):
    RESULTS.append(status)
    extra = "" if want is None or status == "PASS" else f"   (expected {want})"
    print(f"{status:4s}  {key} = {got}{extra}")


def same(a, b):
    try:
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def expect(key, got, want):
    say("PASS" if same(got, want) else "FAIL", key, got, want)


def load_protocol():
    p = HERE / "protocol.yaml"
    try:
        import yaml
    except ImportError:
        sys.exit("PyYAML missing: run with an environment python, e.g. $G/envs/psi0/bin/python " + " ".join(sys.argv))
    return yaml.safe_load(p.read_text())


def find_g(run):
    for p in [run, *run.parents]:
        if (p / "envs").is_dir() and (p / "relay").is_dir():
            return p
    return Path(os.environ["G"]) if os.environ.get("G") else None


def task_of(run, model):
    name = run.name if model != "psi0" else run.parent.name
    return re.sub(r"-(cont\d*k?)$", "", name)


def inloop_points(run):
    pts = set()
    for f in list(run.rglob("inloop-mse.jsonl"))[:3]:
        for line in f.read_text().splitlines():
            try:
                pts.add(int(json.loads(line)["step"]))
            except (ValueError, KeyError):
                pass
    if not pts:  # FastWAM / Psi-Zero also log the evaluations to the training log
        for f in list(run.rglob("node-0.log"))[:2]:
            for m in re.finditer(r"HB_INLOOP \w+ step (\d+):", f.read_text(errors="ignore")):
                pts.add(int(m.group(1)))
    return sorted(pts)


def check_inloop(run, steps, every):
    pts = inloop_points(run)
    want = list(range(every, steps + 1, every))
    missing = [s for s in want if s not in pts]
    if not pts:
        say("FAIL", "in-loop MSE points", 0, f"every {every} steps (none found: the run trained without in-loop MSE)")
    elif missing:
        say("WARN", "in-loop MSE points", f"{len(pts)} (missing {missing[:6]}{'...' if len(missing) > 6 else ''})",
            f"{len(want)}")
    else:
        say("PASS", "in-loop MSE points", f"{len(pts)} (every {every} to {steps})")


def groot_args(path, g):
    code = ("import json,sys,torch; a=torch.load(sys.argv[1],weights_only=False); print(json.dumps({k:getattr(a,k,None) "
            "for k in ('seed','data_seed','per_device_train_batch_size','gradient_accumulation_steps','max_steps',"
            "'learning_rate','warmup_ratio','weight_decay','lr_scheduler_type','save_steps','bf16')}, default=str))")
    exes = [str(g / "envs/groot-py312/bin/python")] if g and (g / "envs/groot-py312/bin/python").exists() else []
    for exe in exes + [sys.executable]:  # the GR00T environment unpickles its own TrainingArguments reliably
        out = subprocess.run([exe, "-c", code, str(path)], capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return json.loads(out.stdout.strip().splitlines()[-1])
    return None


def check_groot(run, proto):
    m = proto["models"]["groot"]
    task = task_of(run, "groot")
    long_task = proto["tasks"].get(task, {}).get("long_task", False)
    steps = m["max_steps_c6"] if long_task else m["max_steps"]
    prod = run / "production"
    cks = sorted((p for p in prod.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
                 key=lambda p: int(p.name.split("-")[-1]))
    if not cks:
        say("FAIL", "checkpoints", "none", str(prod))
        return steps, m
    comp = json.loads((run / "production-complete.json").read_text()) if (run / "production-complete.json").is_file() else {}
    world = comp.get("world_size") or comp.get("num_gpus") or 8 * int(comp.get("num_nodes") or (4 if "quad" in str(comp) else 2))
    a = groot_args(cks[-1] / "training_args.bin", find_g(run))
    if a is None:
        say("WARN", "training_args.bin", "unreadable (need torch + transformers)")
    else:
        expect("seed", a["seed"], m["seed"])
        say("PASS" if a["data_seed"] in (None, m["seed"]) else "FAIL", "data_seed", a["data_seed"], f"None or {m['seed']}")
        expect("gradient_accumulation_steps", a["gradient_accumulation_steps"], 1)
        expect("effective global batch (per_gpu x world x accum)",
               a["per_device_train_batch_size"] * world * int(a["gradient_accumulation_steps"] or 1), m["global_batch"])
        expect("max_steps", a["max_steps"], steps)
        expect("learning_rate", a["learning_rate"], m["learning_rate"])
        expect("warmup_ratio", a["warmup_ratio"], m["warmup_ratio"])
        expect("weight_decay", a["weight_decay"], m["weight_decay"])
        say("PASS" if "cosine" in str(a["lr_scheduler_type"]).lower() else "FAIL", "lr_scheduler_type", a["lr_scheduler_type"], "cosine")
    if comp:
        expect("production-complete global_batch", comp.get("global_batch"), m["global_batch"])
    have = [int(p.name.split("-")[-1]) for p in cks]
    want = [s for s in range(steps // 2, steps + 1, 500)]
    say("PASS" if all(s in have for s in want) else "WARN", "checkpoints", have, want)
    return steps, m


def parse_argv(text):
    toks, d, i = text.split(), {}, 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            k, _, v = t[2:].partition("=")
            if not _:
                vals = []
                while i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                    i += 1
                    vals.append(toks[i])
                v = " ".join(vals) if vals else True
            d[k] = v
        i += 1
    return d


def check_psi0(run, proto):
    m = proto["models"]["psi0"]
    task = task_of(run, "psi0")
    long_task = proto["tasks"].get(task, {}).get("long_task", False)
    steps = m["max_steps_c6"] if long_task else m["max_steps"]
    f = run / "train_args.txt"
    if not f.is_file():
        say("FAIL", "train_args.txt", "missing", str(f))
        return steps, m
    a = parse_argv(f.read_text())
    dec = json.loads((run / "decision.json").read_text()) if (run / "decision.json").is_file() else {}
    expect("seed", a.get("seed"), m["seed"])
    expect("hb-bias-seed", a.get("data.transform.repack.hb-bias-seed"), m["bias_seed"])
    expect("gradient_accumulation_steps", a.get("train.gradient_accumulation_steps"), 1)
    expect("global batch (decision.json)", dec.get("global_batch"), m["global_batch"])
    if dec.get("num_nodes"):
        expect("global batch (per_gpu x 8 x nodes)", int(a.get("train.train_batch_size", 0)) * 8 * int(dec["num_nodes"]), m["global_batch"])
    expect("max_training_steps", a.get("train.max_training_steps"), steps)
    expect("learning_rate", a.get("train.learning_rate"), m["learning_rate"])
    expect("lang-backbone-lr", a.get("model.lang-backbone-lr"), m["param_group_lr"]["lang_backbone"])
    expect("vision-tower-lr", a.get("model.vision-tower-lr"), m["param_group_lr"]["vision_tower"])
    expect("mm-projector-lr", a.get("model.mm-projector-lr"), m["param_group_lr"]["mm_projector"])
    expect("warmup_steps", a.get("train.warmup_steps"), max(1, round(m["warmup_ratio"] * steps)))
    expect("lr_scheduler_type", a.get("train.lr_scheduler_type"), m["lr_schedule"])
    expect("weight_decay", a.get("train.lr_scheduler_kwargs.weight_decay"), m["weight_decay"])
    expect("max_grad_norm", a.get("train.max_grad_norm"), m["max_grad_norm"])
    expect("mixed_precision", a.get("train.mixed_precision"), m["mixed_precision"])
    expect("hb-delay-prob (proprio delay)", a.get("data.transform.repack.hb-delay-prob"), m["hb_delay_prob"])
    expect("native img-aug off", "no-img-aug" in " ".join(a), True)
    expect("hb-aug on", "data.transform.model.hb-aug" in a, True)
    return steps, m


def load_yaml(path):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def check_fastwam(run, proto):
    m = proto["models"]["fastwam"]
    task = task_of(run, "fastwam")
    long_task = proto["tasks"].get(task, {}).get("long_task", False)
    cont = re.search(r"-cont(\d*)k?$", run.name)
    if cont:
        steps = 4000 if run.name.endswith("cont4k") else m["continuation_steps"]
        lr = m["continuation_learning_rate"]
    else:
        steps, lr = (m["max_steps_c6"] if long_task else m["max_steps"]), m["learning_rate"]
    cfg = run / "out/config.yaml"
    if not cfg.is_file():
        say("FAIL", "out/config.yaml", "missing", str(cfg))
        return steps, m
    c = load_yaml(cfg)
    comp = json.loads((run / "complete.json").read_text()) if (run / "complete.json").is_file() else {}
    expect("seed", c.get("seed"), m["seed"])
    expect("gradient_accumulation_steps", c.get("gradient_accumulation_steps"), 1)
    expect("per_gpu batch", c.get("batch_size"), m["layout"]["per_gpu_batch"])
    if comp.get("global_batch"):
        expect("global batch", comp["global_batch"], m["global_batch"])
    expect("max_steps", c.get("max_steps"), steps)
    expect("learning_rate", c.get("learning_rate"), lr)
    expect("lr_scheduler_type", c.get("lr_scheduler_type"), m["lr_schedule"])
    expect("weight_decay", c.get("weight_decay"), m["weight_decay"])
    expect("max_grad_norm", c.get("max_grad_norm"), m["max_grad_norm"])
    expect("mixed_precision", c.get("mixed_precision"), m["mixed_precision"])
    return steps, m


def check_aug_seed(run, seed):
    """2026-11-23: GR00T / Psi-Zero / FastWAM entries print HB_AUG_SEED=<seed> (seeded augmentation stream)."""
    for f in sorted((run / "logs").glob("*.log"))[:8]:
        try:
            m = re.search(rb"HB_AUG_SEED=(\d+)", open(f, "rb").read(8 << 20))
        except OSError:
            continue
        if m:
            expect("augmentation stream seed (HB_AUG_SEED)", int(m.group(1)), seed)
            return
    say("WARN", "augmentation stream seed", "no HB_AUG_SEED line",
        "seeded stream (runs started before 2026-11-23 13:05 UTC are not seeded: registered exception)")


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ("groot", "psi0", "fastwam"):
        sys.exit(__doc__)
    model, run = sys.argv[1], Path(sys.argv[2]).resolve()
    proto = load_protocol()
    print(f"== {model} {run}  (protocol {proto['version']})")
    steps, m = {"groot": check_groot, "psi0": check_psi0, "fastwam": check_fastwam}[model](run, proto)
    check_inloop(run, steps, proto["interface"]["inloop_mse"]["every"])
    check_aug_seed(run, m["seed"])
    n_fail, n_warn = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"== {'OK' if not n_fail else 'NOT ALIGNED'}: {RESULTS.count('PASS')} pass, {n_fail} fail, {n_warn} warn"
          + ("  -> fix it, or register the deviation as an exception (REPRODUCIBILITY.md section 7)" if n_fail else ""))
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
