"""Portable launcher for the validated R1Lite I2I+A2A+IC training protocol."""
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def schedule(frames, world_size, batch_size, epochs=20):
    if min(frames, world_size, batch_size, epochs) <= 0:
        raise ValueError("Frames, GPUs, batch size and epochs must be positive")
    per_epoch = math.ceil(frames / (world_size * batch_size))
    return per_epoch, per_epoch * epochs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["cups", "flower"], default="cups")
    p.add_argument("--dataset", type=Path, required=True, help="Prepared two-view dataset")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--port", type=int, default=29713)
    p.add_argument("--resume", type=Path)
    p.add_argument("--print-command", action="store_true")
    args = p.parse_args()
    dataset, output = args.dataset.resolve(), args.output.resolve()
    task = "r1lite_cups_a2ai2i_20ep" if args.task == "cups" else "r1lite_insert_flower_20ep"
    info = json.loads((dataset / "meta/info.json").read_text())
    per_epoch, steps = schedule(int(info["total_frames"]), args.gpus, args.batch_size, args.epochs)
    meta = dataset / "imagewam_training_meta"
    for key in ["FLUX2_SRC", "FLUX2_MODEL_PATH", "FLUX2_AE_MODEL_PATH", "ACTION_INIT"]:
        if not os.environ.get(key):
            p.error(f"Set {key} first (see docs/R1LITE_REAL_ROBOT.md)")
    for path in [meta / "dataset_stats.json", meta / "qwen3_flux2_len128"]:
        if not path.exists():
            p.error(f"Missing preparation output: {path}")
    if args.resume and not (args.resume / "trainer_state.json").is_file():
        p.error("--resume must be a complete training-state directory, not inference weights")
    if not args.resume and output.exists() and any(output.iterdir()):
        p.error("Refusing to overwrite a non-empty training output without --resume")
    overrides = [f"task={task}", f"output_dir={output}", f"batch_size={args.batch_size}",
        f"num_epochs={args.epochs}", f"max_steps={steps}", f"warmup_steps={math.floor(steps * .05)}",
        f"save_every={per_epoch}", "save_at_end=true", "log_every=10", "rank_timer_every=100",
        "wandb.enabled=false", "gradient_accumulation_steps=1",
        f"data.train.dataset_dirs=[{dataset}]", f"data.train.qwen_text_cache_dir={meta}/qwen3_flux2_len128",
        f"data.train.pretrained_norm_stats={meta}/dataset_stats.json",
        f"model.flux2_src_path={os.environ['FLUX2_SRC']}",
        f"model.flux2_model_path={os.environ['FLUX2_MODEL_PATH']}",
        f"model.ae_model_path={os.environ['FLUX2_AE_MODEL_PATH']}",
        f"model.action_dit_pretrained_path={os.environ['ACTION_INIT']}"]
    if args.resume:
        overrides.append(f"resume={args.resume.resolve()}")
    command = [sys.executable, "-m", "accelerate.commands.launch", "--config_file",
        "scripts/accelerate_configs/accelerate_zero1_ds.yaml", "--num_processes", str(args.gpus),
        "--main_process_port", str(args.port), "scripts/train.py", *overrides]
    print(json.dumps({"frames": info["total_frames"], "steps_per_epoch": per_epoch,
                      "optimizer_steps": steps, "global_batch": args.gpus * args.batch_size}))
    print(shlex.join(command), flush=True)
    if args.print_command:
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""))
    subprocess.run(command, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
