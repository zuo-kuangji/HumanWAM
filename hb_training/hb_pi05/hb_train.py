"""torchrun entry point: OpenPI scripts/train_pytorch.py:train_loop, unmodified, with G1 hooks installed.

Hooks (module attributes that OpenPI looks up at call time; no OpenPI file is edited):
  openpi.training.data_loader.create_torch_dataset       -> hb_g1_data.HBCachedDataset + HB aug v2 (training only)
  models_pytorch.preprocessing_pytorch.preprocess_observation_pytorch -> train=False (OpenPI crop/rotate/jitter off)
  torch DistributedSampler -> advances its epoch on every pass (DataLoaderImpl has no set_epoch, so upstream would
                              replay the same shuffle every epoch) and encodes the pass into the index (index + pass*N)
  train_pytorch.set_seed  -> seeded by global rank (upstream uses local rank: identical noise/time draws across nodes)
  train_pytorch.save_checkpoint -> saves only at the requested steps (epoch 10 and the last step), then barrier;
                              older checkpoints lose optimizer.pt (resume state only for the latest one)
Budget: num_train_steps = ceil(epochs * frames / (per_gpu_batch * world_size)); frames = rows of the task's training view.
Probe mode (--probe-dir): run a few steps without base weights, write ok_rank<r>.json / oom_rank<r>.json markers.
"""
import argparse
import dataclasses
import datetime
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_g1_config as C  # noqa: E402  (registers the G1 configs)
import hb_g1_data as D  # noqa: E402


class _EpochAdvancingSampler(torch.utils.data.distributed.DistributedSampler):
    """Reshuffles every pass, and passes the pass number to the dataset as index + pass * N (HBCachedDataset decodes
    it with divmod) so the per-trajectory joint bias can be redrawn per pass identically on all ranks/workers."""

    def __iter__(self):
        epoch, n = self.epoch, len(self.dataset)
        indices = list(super().__iter__())
        self.set_epoch(epoch + 1)
        return iter([i + epoch * n for i in indices])


def load_train_pytorch():
    path = Path(os.environ["HB_OPENPI"]) / "scripts/train_pytorch.py"
    spec = importlib.util.spec_from_file_location("openpi_train_pytorch", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def is_oom(e: BaseException) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def marker(d: str, name: str, payload: dict):
    Path(d).mkdir(parents=True, exist_ok=True)
    tmp = Path(d) / f".{name}.tmp"
    tmp.write_text(json.dumps(payload))
    tmp.rename(Path(d) / name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(D.TASKS))
    ap.add_argument("--exp-name", required=True)
    ap.add_argument("--per-gpu-batch", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("HB_EPOCHS", "20")))
    ap.add_argument("--save-epochs", default=os.environ.get("HB_SAVE_EPOCHS", "10"), help="comma list; last step always saved")
    ap.add_argument("--num-train-steps", type=int, default=None, help="override (probe)")
    ap.add_argument("--ckpt-base", default=None)
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("HB_NUM_WORKERS", "8")))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--probe-dir", default=None)
    ap.add_argument("--probe-max-frac", type=float, default=float(os.environ.get("HB_PROBE_MAX_FRAC", "0.95")))
    ap.add_argument("--nccl-timeout-min", type=int, default=60)
    a = ap.parse_args()

    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    tp = load_train_pytorch()
    tp.init_logging()

    D.install_hooks(augment=not a.no_aug)
    torch.utils.data.distributed.DistributedSampler = _EpochAdvancingSampler
    orig_seed = tp.set_seed
    tp.set_seed = lambda seed, local_rank: orig_seed(seed, rank)  # noqa: ARG005

    frames = json.loads((D.task_dir(a.task) / "meta.json").read_text())["frames"]
    global_batch = a.per_gpu_batch * world
    # 2026-11-12 user: fixed optimizer steps instead of epochs (4000; HB_FIXED_STEPS=0 restores the epoch budget);
    # checkpoints every 2000 steps counted back from the last step, only the last two (resume state only for the last).
    fixed_total = int(os.environ.get("HB_FIXED_STEPS", "4000"))
    fixed = not a.num_train_steps and fixed_total > 0
    if fixed:
        steps = fixed_total
        save_steps = set() if a.probe_dir else {s for s in (steps - 2000, steps) if s > 0}
    else:
        steps = a.num_train_steps or math.ceil(a.epochs * frames / global_batch)
        save_steps = set() if a.probe_dir else {steps} | {
            math.ceil(float(e) * frames / global_batch) for e in a.save_epochs.split(",") if e.strip()}
    orig_save = tp.save_checkpoint

    def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
        if global_step in save_steps:
            # upstream condition `global_step % save_interval == 0` is made true for exactly this step
            orig_save(model, optimizer, global_step, dataclasses.replace(config, save_interval=global_step), is_main,
                      data_config)
            if is_main:  # resume state only for the latest checkpoint: older steps keep weights, lose optimizer.pt
                for d in Path(config.checkpoint_dir).iterdir():
                    if d.is_dir() and d.name.isdigit() and int(d.name) != global_step and (d / "optimizer.pt").exists():
                        (d / "optimizer.pt").unlink()
                        print(f"[hb] dropped resume state {d}/optimizer.pt", flush=True)
            if dist.is_initialized():
                dist.barrier()

    tp.save_checkpoint = save_checkpoint

    cfg = C.make_train_config(a.task, exp_name=a.exp_name, batch_size=global_batch, num_train_steps=steps,
                              num_workers=a.num_workers, checkpoint_base_dir=a.ckpt_base, resume=a.resume,
                              pytorch_weight_path=None if a.probe_dir else "default",
                              log_interval=1 if a.probe_dir else 10)
    plan = dict(task=a.task, layout=D.TASKS[a.task].layout, config=cfg.name, exp_name=a.exp_name, frames=frames, world_size=world, per_gpu_batch=a.per_gpu_batch,
                global_batch=global_batch, epochs=None if fixed else a.epochs, fixed_steps=fixed,
                epochs_equivalent=round(steps * global_batch / frames, 2), num_train_steps=steps, save_steps=sorted(save_steps),
                warmup_steps=cfg.lr_schedule.warmup_steps, peak_lr=cfg.lr_schedule.peak_lr,
                weights=cfg.pytorch_weight_path, checkpoint_dir=str(cfg.checkpoint_dir), augment=not a.no_aug,
                aug_version=D.aug.VERSION, max_token_len=cfg.model.max_token_len)
    if rank == 0:
        print("HB_PLAN", json.dumps(plan), flush=True)
        if not a.probe_dir:
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (cfg.checkpoint_dir / "hb_plan.json").write_text(json.dumps(plan, indent=2))
    if not a.probe_dir and cfg.pytorch_weight_path and not Path(cfg.pytorch_weight_path, "model.safetensors").exists():
        sys.exit(f"missing init weights {cfg.pytorch_weight_path}/model.safetensors (run convert_base.sh)")

    if world > 1 and not dist.is_initialized():  # train_pytorch.setup_ddp skips init if already initialized
        torch.cuda.set_device(local)
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=a.nccl_timeout_min))

    t0 = time.time()
    try:
        tp.train_loop(cfg)
    except BaseException as e:
        if a.probe_dir and is_oom(e):
            marker(a.probe_dir, f"oom_rank{rank}.json", {"error": str(e)[:500]})
        raise
    if a.probe_dir:
        peak = torch.cuda.max_memory_reserved(local)
        total = torch.cuda.get_device_properties(local).total_memory
        name = "ok" if peak / total <= a.probe_max_frac else "oom"
        marker(a.probe_dir, f"{name}_rank{rank}.json", {"per_gpu_batch": a.per_gpu_batch, "peak_reserved_gb": peak / 1e9,
                                                          "total_gb": total / 1e9, "seconds": time.time() - t0,
                                                          "steps": steps})
    elif rank == 0:
        (cfg.checkpoint_dir / "HB_DONE").write_text(json.dumps({"seconds": time.time() - t0, **plan}))


if __name__ == "__main__":
    main()
