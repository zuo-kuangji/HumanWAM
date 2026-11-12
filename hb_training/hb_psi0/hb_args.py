"""Per-task native Psi0 argument lists, step math, and the probe decision.

  hb_args.py plan   <task> <per_gpu_batch>                     -> JSON step plan
  hb_args.py args   <task> (--decision DEC | --batch B --steps S --save N --warmup W)
                    --out DIR --timestamp TS [--workers N] [--no-resume]   -> one arg per line
  hb_args.py decide <task> <probe_dir> <decision_json> <cands...>  -> exit 0 decided / 3 not yet

Native recipe = scripts/train/psi0/finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh (same model
flags; verified against the init dir's action_header.safetensors: 12 blocks, qk rms_norm, state_null,
view_feature_dim 2048, pooled 768). Changed: action/state dims and keys come from the view receipt
(per-robot layout written by prepare_data.py: G1 A76/S44 body 29, H2 a18 A75/S78 body 31), H30 kept,
native augmentations off (HB v2 instead), steps/warmup/checkpoint cadence from the 20-epoch rule,
no wandb, no validation.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
G = Path(os.environ.get("G", H.parent))
TASKS = ("b6", "basket", "a3", "a7", "a18", "trolley", "a17", "b7", "a6", "c6t", "c6")
WORLD = int(os.environ.get("HB_WORLD", "32"))          # 4 nodes x 8 GPUs
EPOCHS, SAVE_EPOCH, WARMUP_FRAC = 20, 10, 0.025        # warmup: native 1000/40000 = 2.5 %
HORIZON = 30
POST = Path(os.environ.get("HB_POST_DIR", G / "models/psi0/psi0/postpre.sonic1.1.unifolm.2609181726.40k"))
CLIP = Path(os.environ.get("HB_CLIP_DIR", G / "models/clip-vit-large-patch14"))
DATA = Path(os.environ.get("HB_DATA_ROOT", G / "psi0-data"))


def receipt(task):
    p = DATA / task / "hb-view-receipt.json"
    if not p.is_file():
        sys.exit(f"missing {p}: run prepare_data.py {task} first")
    return json.loads(p.read_text())


def plan(frames, batch):
    gb = batch * WORLD
    steps = math.ceil(EPOCHS * frames / gb)
    save = math.ceil(SAVE_EPOCH * frames / gb)       # saves at `save` and at `steps` only (2*save >= steps)
    return {"frames": frames, "per_gpu_batch": batch, "global_batch": gb, "epochs": EPOCHS, "steps": steps,
            "save_steps": save, "warmup_steps": max(1, round(WARMUP_FRAC * steps))}


def build_args(task, batch, steps, save, warmup, out, ts, workers, resume=True):
    r = receipt(task)
    cams = r["image_keys"]
    SK, AK, SD, AD, NB = r["state_key"], r["action_key"], int(r["state_dim"]), int(r["action_dim"]), int(r["body_dims"])
    stats = str(DATA / task / "meta" / "stats_psi0.json")
    drop = os.environ.get("HB_STATE_DROP_PROB", "0.0")
    a = [
        "--seed=292285", f"--exp=hb-{task}", f"--timestamp={ts}",
        *(["--train.resume_from_checkpoint=latest"] if resume else []),
        "--train.name=finetune", f"--train.output_dir={out}",
        "--train.data_parallel=ddp", "--train.mixed_precision=bf16",
        f"--train.train_batch_size={batch}", f"--train.val_batch_size={batch}",
        "--train.max_checkpoints_to_keep=2", "--train.preserve_checkpoint_every=None",
        "--train.gradient_accumulation_steps=1", "--train.learning_rate=1e-4",
        f"--train.max_training_steps={steps}", "--train.warmup_ratio=None", f"--train.warmup_steps={warmup}",
        f"--train.checkpointing_steps={save}", "--train.validation_steps=0",
        "--train.max_grad_norm=1.0", "--train.lr_scheduler_type=cosine",
        "--train.lr_scheduler_kwargs.weight_decay=1e-6", "--train.lr_scheduler_kwargs.betas", "0.95", "0.999",
        f"--data.root_dir={DATA}", "--data.train_repo_ids", task, "--data.val_repo_ids", task,
        f"--data.num_workers={workers}", "--data.prefetch_factor=4",
        "--data.transform.repack.image-keys", *cams,
        "--data.transform.repack.state-keys", SK,
        "--data.transform.repack.action-keys", AK,
        f"--data.transform.repack.dataset-name=hb_{task}",
        f"--data.transform.repack.pad-action-dim={AD}", f"--data.transform.repack.pad-state-dim={SD}",
        "--data.transform.repack.instruction-key=task",
        f"--data.transform.repack.action-chunk-size={HORIZON}",
        "--data.transform.repack.state-temporal-jitter=0",
        # 2026-11-12 user: align with GR00T A20 -> no proprio delay (Psi0 has no action-history input, so no history drop)
        "--data.transform.repack.hb-delay-prob=0.0", "--data.transform.repack.hb-delay-max-frames=2",
        f"--data.transform.repack.hb-body-dims={NB}", "--data.transform.repack.hb-bias-seed=292285",
        f"--data.transform.field.stat-path={stats}", "--data.transform.field.state-noise-std=0.0",
        "--data.transform.field.stat-action-keys", AK,
        "--data.transform.field.stat-state-keys", SK,
        "--data.transform.field.action_norm_type=bounds", "--data.transform.field.no-use-norm-mask",
        "--data.transform.field.normalize-state",
        f"--data.transform.field.pad-action-dim={AD}", f"--data.transform.field.pad-state-dim={SD}",
        "--data.transform.model.no-img-aug", "--data.transform.model.no-view-aug",
        "--data.transform.model.resize.size", "270", "480",
        "--data.transform.model.center_crop.size", "270", "480",
        "--data.transform.model.hb-aug",
        f"--model.model_name_or_path={POST}", f"--model.pretrained-action-header-path={POST}",
        "--model.noise-scheduler=flow", "--model.train-diffusion-steps=1000", "--model.n_conditions=0",
        f"--model.action-chunk-size={HORIZON}", f"--model.action-dim={AD}",
        f"--model.action-exec-horizon={HORIZON}", "--model.observation-horizon=1", f"--model.odim={SD}",
        "--model.dropout=0.0", "--model.state-feature-dropout=0.0", "--model.view_feature_dim=2048",
        "--model.tune-vlm", "--model.lang-backbone-lr=1e-6", "--model.vision-tower-lr=1e-5",
        "--model.mm-projector-lr=1e-4", "--model.gradient-checkpointing", "--model.no-use_film",
        "--model.qk-norm=rms_norm", "--model.combined-temb", "--model.num-blocks=12",
        "--model.vlm-layer-indices", *"3 5 8 10 12 14 17 19 21 23 26 28".split(),
        f"--model.state-drop-prob={drop}", "--model.state-as-action-token", "--model.state-null-token",
        "--model.pooled-text-encoder=clip", f"--model.pooled-text-encoder-path={CLIP}",
        "--model.pooled-projection-dim=768", "--model.pooled-cache-path=clip_pooled_cache.pt",
        "--model.no-rtc", "--model.max-delay=8",
    ]
    return a


def decide(task, probe_dir, dec_path, cands, frac=None):
    frac = float(os.environ.get("HB_PROBE_MEM_FRAC", "0.92")) if frac is None else frac
    rows = []
    for b in sorted({int(c) for c in cands}, reverse=True):
        d = Path(probe_dir) / f"bs{b}"
        rc_f = Path(probe_dir) / f"bs{b}.rc"
        if not rc_f.is_file():
            continue
        ranks = [json.loads(p.read_text()) for p in sorted(d.glob("rank*.json"))]
        ok = rc_f.read_text().strip() == "0" and len(ranks) >= 2 and all(r.get("ok") for r in ranks)
        pk = max((r.get("peak_frac", 1.0) for r in ranks), default=None)
        sps = max((r.get("sec_per_step", 0.0) for r in ranks), default=None)
        rows.append({"per_gpu_batch": b, "ok": ok, "peak_frac": pk, "sec_per_step": sps,
                     "fits": bool(ok and pk is not None and pk <= frac),
                     "error": next((r.get("error") for r in ranks if r.get("error")), None)})
    good = [r for r in rows if r["fits"]]
    print(json.dumps(rows, indent=1))
    if not good:
        return 3
    best = max(good, key=lambda r: r["per_gpu_batch"])
    r = receipt(task)
    dec = plan(r["frames"], best["per_gpu_batch"])  # probed per task, i.e. per robot layout / camera set
    dec.update(task=task, robot=r.get("robot"), state_dim=r["state_dim"], action_dim=r["action_dim"],
               image_keys=r["image_keys"], mem_frac_limit=frac, probe=rows,
               est_hours=(dec["steps"] * best["sec_per_step"] / 3600.0) if best["sec_per_step"] else None)
    tmp = Path(str(dec_path) + ".tmp")
    tmp.write_text(json.dumps(dec, indent=2) + "\n")
    os.replace(tmp, dec_path)
    print("DECISION", json.dumps({k: dec[k] for k in ("per_gpu_batch", "global_batch", "steps", "save_steps")}))
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("task"); p.add_argument("batch", type=int)
    p = sub.add_parser("args"); p.add_argument("task")
    p.add_argument("--decision"); p.add_argument("--batch", type=int); p.add_argument("--steps", type=int)
    p.add_argument("--save", type=int); p.add_argument("--warmup", type=int)
    p.add_argument("--out", required=True); p.add_argument("--timestamp", required=True)
    p.add_argument("--workers", type=int, default=int(os.environ.get("HB_NUM_WORKERS", "8")))
    p.add_argument("--no-resume", action="store_true")
    p = sub.add_parser("decide"); p.add_argument("task"); p.add_argument("probe_dir"); p.add_argument("decision")
    p.add_argument("cands", nargs="+")
    a = ap.parse_args()
    if a.cmd != "decide" and a.task not in TASKS:
        sys.exit(f"unknown task {a.task}; known {TASKS}")
    if a.cmd == "plan":
        print(json.dumps(plan(receipt(a.task)["frames"], a.batch), indent=2))
    elif a.cmd == "args":
        if a.decision:
            d = json.loads(Path(a.decision).read_text())
            assert d["task"] == a.task and d["frames"] == receipt(a.task)["frames"], "stale decision file"
            b, s, sv, w = d["per_gpu_batch"], d["steps"], d["save_steps"], d["warmup_steps"]
        else:
            b, s, sv, w = a.batch, a.steps, a.save, a.warmup
            assert None not in (b, s, sv, w), "--batch/--steps/--save/--warmup or --decision"
        print("\n".join(build_args(a.task, b, s, sv, w, a.out, a.timestamp, a.workers, not a.no_resume)))
    else:
        sys.exit(decide(a.task, a.probe_dir, a.decision, a.cands))


if __name__ == "__main__":
    main()
