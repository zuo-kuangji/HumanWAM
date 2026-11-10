"""Entry point started by `accelerate launch` (DeepSpeed ZeRO-2, bf16; one process per GPU).

Runs the UNMODIFIED DiT4DiT/training/train.py:main(cfg) of the pinned checkout ($HB_D4D_SRC @ 66a6f3a) after installing
hb_patches (see that module for the list). cfg = native dit4dit_g1.yaml + hb_d4d_config.overrides(plan).
Budget: steps = ceil(20 * frames / (per_gpu_batch * world)); checkpoints at ceil(10 * frames / global_batch) and the last
step. Resume is automatic when <run>/checkpoints/steps_*_pytorch_model.pt exists (weights via the native is_resume
path, optimizer/RNG via the matching steps_<N>_resume dir).
Probe mode (--probe-dir): a few steps, no checkpoints, writes ok_rank<r>.json / oom_rank<r>.json with peak memory.
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))


def is_oom(e: BaseException) -> bool:
    return "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError"


def marker(d, name, payload):
    Path(d).mkdir(parents=True, exist_ok=True)
    tmp = Path(d) / f".{name}.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, Path(d) / name)


def load_train_module(src: Path):
    path = src / "DiT4DiT" / "training" / "train.py"
    spec = importlib.util.spec_from_file_location("dit4dit_native_train", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # builds the native Accelerator(deepspeed_plugin=...) here
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--per-gpu-batch", type=int, required=True)
    ap.add_argument("--run-root", required=True, help="output_dir = <run-root>/run")
    ap.add_argument("--ds-config", required=True)
    ap.add_argument("--epochs", type=float, default=float(os.environ.get("HB_EPOCHS", "20")))
    ap.add_argument("--save-epochs", default=os.environ.get("HB_SAVE_EPOCHS", "10"))
    ap.add_argument("--num-train-steps", type=int, default=None, help="override (probe)")
    ap.add_argument("--num-workers", type=int, default=int(os.environ.get("HB_NUM_WORKERS", "6")))
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--probe-dir", default=None)
    ap.add_argument("--probe-max-frac", type=float, default=float(os.environ.get("HB_PROBE_MAX_FRAC", "0.92")))
    a = ap.parse_args()

    import hb_d4d_config as C
    import hb_d4d_data as D
    import hb_patches as PT

    src = Path(os.environ["HB_D4D_SRC"]).resolve()
    sys.path.insert(0, str(src))
    os.chdir(src)  # native code uses paths relative to the repo root
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    cosmos = Path(os.environ.get("HB_COSMOS_DIR", ""))
    if not (cosmos / "model_index.json").exists():
        sys.exit(f"Cosmos pipeline dir {cosmos} missing model_index.json (run fetch_models.sh)")

    meta = D.load_meta(a.task)
    plan = C.plan(a.task, a.per_gpu_batch, world, frames=meta["frames"], epochs=a.epochs,
                  save_epochs=[e for e in a.save_epochs.split(",") if e.strip()], steps_override=a.num_train_steps)
    plan.update(probe=bool(a.probe_dir), aug=not a.no_aug, aug_version=D.aug.VERSION, ds_config=a.ds_config,
                cosmos_dir=str(cosmos), num_workers=a.num_workers, prompt=meta["prompt"])
    ds = json.loads(Path(a.ds_config).read_text())
    assert ds["train_micro_batch_size_per_gpu"] == a.per_gpu_batch and ds["train_batch_size"] == plan["global_batch"], (
        "ds_config does not match --per-gpu-batch / WORLD_SIZE", ds, plan)
    run_root = Path(a.run_root)
    out_dir = run_root / "run"
    prev = None if a.probe_dir else PT.latest_weights(out_dir / "checkpoints")
    plan["resume_from_step"] = prev[0] if prev else 0
    cfg = C.make_cfg(plan, run_root=str(run_root), run_id="run", cosmos_dir=str(cosmos), resume=bool(prev),
                     probe=bool(a.probe_dir))

    PT.install_pre(plan, a.ds_config, augment=not a.no_aug, num_workers=a.num_workers)
    train = load_train_module(src)
    PT.restore_plugin()
    PT.install_post(train)

    if rank == 0:
        print("HB_PLAN", json.dumps(plan), flush=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        from omegaconf import OmegaConf

        OmegaConf.save(cfg, str(out_dir / "hb_full_config.yaml"))
        if not a.probe_dir:
            (out_dir / "hb_plan.json").write_text(json.dumps(plan, indent=2))
    t0 = time.time()
    try:
        train.main(cfg)
    except BaseException as e:
        if a.probe_dir and is_oom(e):
            marker(a.probe_dir, f"oom_rank{rank}.json", {"error": str(e)[:800], "per_gpu_batch": a.per_gpu_batch})
        raise
    if a.probe_dir:
        import torch

        peak = torch.cuda.max_memory_reserved(local)
        total = torch.cuda.get_device_properties(local).total_memory
        t = PT.STEP_T
        sps = (t[-1] - t[1]) / (len(t) - 2) if len(t) >= 4 else None
        name = "ok" if peak / total <= a.probe_max_frac else "oom"
        marker(a.probe_dir, f"{name}_rank{rank}.json", {
            "per_gpu_batch": a.per_gpu_batch, "peak_reserved_gb": peak / 1e9, "total_gb": total / 1e9,
            "peak_frac": peak / total, "max_frac": a.probe_max_frac, "steps": len(t), "sec_per_step": sps,
            "seconds": time.time() - t0, "world": world})


if __name__ == "__main__":
    main()
