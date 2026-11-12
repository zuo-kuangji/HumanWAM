"""ACT args for the pinned Psi0 scripts/train.py (config hb_act_config), on the Psi0 views $G/psi0-data/<task>.
Same 20-epoch plan / probe decision as Psi-Zero (hb_psi0/hb_args.py: plan, decide, receipt); only the args differ.

  hb_act_args.py plan <task> <batch>
  hb_act_args.py args <task> (--decision D | --batch B --steps S --save V --warmup W) --out DIR --timestamp T
  hb_act_args.py decide <task> <probe_dir> <decision.json> <cand ...>
"""
import argparse
import json
import os
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H.parent / "hb_psi0"))
import hb_args as HA  # noqa: E402  (TASKS, DATA, receipt, plan, decide; EPOCHS 20, save at 10, warmup 2.5 %)

CHUNK = 16  # native ACT G1 simulation recipe (train_act_g1_simple.sh)


def build_args(task, batch, steps, save, warmup, out, ts, workers, resume=True):
    r = HA.receipt(task)
    cams = r["image_keys"]
    SK, AK, SD, AD, NB = r["state_key"], r["action_key"], int(r["state_dim"]), int(r["action_dim"]), int(r["body_dims"])
    assert SD in (44, 78), f"raw state must be 44 / 78 (no joint velocity), got {SD}"
    stats = str(HA.DATA / task / "meta" / "stats_psi0.json")
    return [
        "--seed=292285", f"--exp=hb-act-{task}", f"--timestamp={ts}",
        *(["--train.resume_from_checkpoint=latest"] if resume else []),
        "--train.name=act-g1", f"--train.output_dir={out}",
        "--train.data_parallel=ddp", "--train.mixed_precision=bf16",
        f"--train.train_batch_size={batch}", f"--train.val_batch_size={batch}",
        "--train.max_checkpoints_to_keep=3", "--train.preserve_checkpoint_every=None",
        "--train.gradient_accumulation_steps=1", "--train.learning_rate=1e-4",
        f"--train.max_training_steps={steps}", "--train.warmup_ratio=None", f"--train.warmup_steps={warmup}",
        f"--train.checkpointing_steps={save}", "--train.validation_steps=0",
        "--train.max_grad_norm=1.0", "--train.lr_scheduler_type=cosine",
        "--train.lr_scheduler_kwargs.weight_decay=1e-6", "--train.lr_scheduler_kwargs.betas", "0.95", "0.999",
        f"--data.root_dir={HA.DATA}", "--data.train_repo_ids", task, "--data.val_repo_ids", task,
        f"--data.num_workers={workers}", "--data.prefetch_factor=4",
        "--data.transform.repack.image-keys", *cams,
        "--data.transform.repack.state-keys", SK,
        "--data.transform.repack.action-keys", AK,
        f"--data.transform.repack.dataset-name=hb_{task}",
        f"--data.transform.repack.pad-action-dim={AD}", f"--data.transform.repack.pad-state-dim={SD}",
        "--data.transform.repack.instruction-key=task",
        f"--data.transform.repack.action-chunk-size={CHUNK}",
        "--data.transform.repack.state-temporal-jitter=0",
        "--data.transform.repack.hb-delay-prob=0.0", "--data.transform.repack.hb-delay-max-frames=2",
        f"--data.transform.repack.hb-body-dims={NB}", "--data.transform.repack.hb-bias-seed=292285",
        f"--data.transform.field.stat-path={stats}", "--data.transform.field.state-noise-std=0.0",
        "--data.transform.field.stat-action-keys", AK,
        "--data.transform.field.stat-state-keys", SK,
        "--data.transform.field.action_norm_type=bounds", "--data.transform.field.no-use-norm-mask",
        "--data.transform.field.normalize-state",
        f"--data.transform.field.pad-action-dim={AD}", f"--data.transform.field.pad-state-dim={SD}",
        "--data.transform.model.no-img-aug", "--data.transform.model.hb-aug",
        f"--model.chunk-size={CHUNK}", f"--model.n-action-steps={CHUNK}",
        f"--model.action-dim={AD}", f"--model.state-dim={SD}",
        "--model.use-vae", "--model.kl-weight=10.0",
    ]


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
    if a.cmd != "decide" and a.task not in HA.TASKS:
        sys.exit(f"unknown task {a.task}; known {HA.TASKS}")
    if a.cmd == "plan":
        print(json.dumps(HA.plan(HA.receipt(a.task)["frames"], a.batch), indent=2))
    elif a.cmd == "args":
        if a.decision:
            d = json.loads(Path(a.decision).read_text())
            assert d["task"] == a.task and d["frames"] == HA.receipt(a.task)["frames"], "stale decision file"
            b, s, sv, w = d["per_gpu_batch"], d["steps"], d["save_steps"], d["warmup_steps"]
        else:
            b, s, sv, w = a.batch, a.steps, a.save, a.warmup
            assert None not in (b, s, sv, w), "--batch/--steps/--save/--warmup or --decision"
        print("\n".join(build_args(a.task, b, s, sv, w, a.out, a.timestamp, a.workers, not a.no_resume)))
    else:
        sys.exit(HA.decide(a.task, a.probe_dir, a.decision, a.cands))


if __name__ == "__main__":
    main()
