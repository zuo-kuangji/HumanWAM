"""DiT4DiT run configuration for the HB tasks: step plan, OmegaConf config, DeepSpeed / accelerate launch files.

Base config = the pinned native DiT4DiT/config/real_robot/dit4dit_g1.yaml, with the values of the native G1 launch
script (examples/Real_G1/train_files/run_real_robot.sh) and of the released DiT4DiT checkpoint config
(mondo-robotics/dit4dit-model@46237ae dit4dit_robocasa_gr1/config.yaml) where the yaml differs:
  frozen text_encoder + vae, flow-matching time distribution uniform, no high-sigma forcing, extract_layer 17,
  conditional_frame_timestep 1e-4, action_video_freq_ratio 2, repeated_diffusion_steps 4,
  lr: backbone_interface 1e-5 / action_model 1e-4 / base 3e-5, cosine_with_min_lr (min_lr 5e-7), AdamW (0.9, 0.95),
  eps 1e-8, wd 1e-8, grad clip 1.0, DeepSpeed ZeRO-2 bf16 (DiT4DiT/config/deepseeds/ds_config.yaml values).
HB changes: dims (G1 A76/S73, H2 A75/S153 = native sin/cos of the raw 44 / 78 state; no padding, no velocity), base_model = local Cosmos dir, 20-epoch budget
(steps = ceil(20*frames/global_batch), warmup = 5% like native 5000/100000), checkpoints at epoch 10 + last step,
no in-loop action eval (it consumes a training batch), explicit DeepSpeed micro batch.

CLI:
  hb_d4d_config.py plan <task> <per_gpu_batch> [--world 32]
  hb_d4d_config.py launch-files --task T --batch B --world W --machines M --out DIR   -> DIR/ds_config.json, accelerate.yaml
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_d4d_data as D  # noqa: E402

EPOCHS, SAVE_EPOCHS, WARMUP_FRAC = 20, (10,), 0.05
# 2026-11-12 user: fixed optimizer steps instead of epochs (4000 for every model except Delta Flow); checkpoints every
# 2000 steps counted back from the last step, only the last two kept (resume state only for the last one).
FIXED_STEPS = int(os.environ.get("HB_FIXED_STEPS", "4000"))
CKPT_EVERY, KEEP_CKPTS = 2000, 2
LR = {"base": 3e-5, "backbone_interface": 1e-5, "action_model": 1e-4}
FREEZE = "backbone_interface.extractor.text_encoder,backbone_interface.extractor.vae"
# written into every run's hb_plan.json (CLAUDE.md: deviations are recorded in the experiment directory)
DEVIATIONS = [
    "CLAUDE.md v2 sec.3 history-frame drop: not applicable (DiT4DiT has no action/state history input)",
    "CLAUDE.md v2 sec.4 joint velocity: not applicable (native DiT4DiT state input has no velocity channel)",
    "per-trajectory joint bias is fixed per (loader pass, episode) and redrawn every pass (same as pi0.5 / Psi-Zero)",
    "native image augmentation: none in the native G1 data config (UnitreeG1DataConfig has no video transforms)",
    "cameras: two views side by side [cam0|cam1] 224x448 (native multi-view concat path); native G1 recipe used ego only",
    "state: benchmark input = raw 44 (G1) / 78 (H2) dims without joint velocity (same for every model); encoded as in the "
    "native G1 recipe (sin/cos on joint-angle keys): sin/cos on every joint angle, min_max on the non-angle dims that "
    "the native G1 data has no counterpart for (G1 inspire hands 0..1, projected gravity) -> 73 (G1) / 153 (H2)",
    "sampling: per-pass permutation without replacement (native: with-replacement hash sampling, same order every pass)",
    "no in-loop action eval (native eval_interval consumes a training batch); prompt embeddings memoised (same values)",
    "Cosmos-Predict2.5-2B transformer weights from the official gated repo (user's HF token), converted with the official recipe",
]


def native_yaml() -> Path:
    return Path(os.environ.get("HB_D4D_SRC", D.G / "HumanWAM/third_party/DiT4DiT")) / "DiT4DiT/config/real_robot/dit4dit_g1.yaml"


def plan(task: str, per_gpu_batch: int, world: int, *, frames: int | None = None, epochs: float = EPOCHS,
         save_epochs=SAVE_EPOCHS, steps_override: int | None = None) -> dict:
    frames = int(frames if frames is not None else D.load_meta(task)["frames"])
    gb = int(per_gpu_batch) * int(world)
    if steps_override:  # probe
        steps, save, fixed = int(steps_override), [], False
    elif FIXED_STEPS > 0:
        steps, fixed = FIXED_STEPS, True
        save = [steps - CKPT_EVERY * k for k in range(KEEP_CKPTS - 1, -1, -1) if steps - CKPT_EVERY * k > 0]
    else:  # legacy epoch budget (HB_FIXED_STEPS=0)
        steps, fixed = int(math.ceil(epochs * frames / gb)), False
        save = sorted({steps} | {math.ceil(float(e) * frames / gb) for e in save_epochs})
    L = D.layout_of(task)
    return dict(task=task, layout=L.name, frames=frames, per_gpu_batch=int(per_gpu_batch), world_size=int(world),
                global_batch=gb, epochs=None if fixed else epochs, fixed_steps=fixed,
                epochs_equivalent=round(steps * gb / frames, 2), steps=steps, save_steps=save,
                warmup_steps=max(1, round(WARMUP_FRAC * steps)), batches_per_pass=frames // gb,
                action_dim=L.action_dim, state_dim=L.state_dim, cameras=list(L.cameras), deviations=list(DEVIATIONS))


def overrides(p: dict, *, run_root: str, run_id: str = "run", cosmos_dir: str, resume: bool, probe: bool) -> dict:
    steps = p["steps"]
    return {
        "run_id": run_id, "run_root_dir": str(run_root), "seed": 42, "is_debug": False,
        "wandb_project": "hb_dit4dit", "wandb_entity": None,
        "framework": {
            "name": "DiT4DiT",
            "cosmos25": {
                "base_model": str(cosmos_dir), "extract_layer": 17, "vl_hidden_dim": 2048, "training": "joint",
                "future_loss_type": "flow_matching", "future_num_inference_steps": 1,
                "conditional_frame_timestep": 0.0001,
                "flow_matching": {"time_distribution": "uniform", "high_sigma_ratio": None, "high_sigma_min": None},
            },
            "action_model": {
                "action_model_type": "DiT-B", "action_dim": p["action_dim"], "state_dim": p["state_dim"],
                "action_horizon": D.HORIZON, "future_action_window_size": D.HORIZON - 1, "past_action_window_size": 0,
            },
        },
        "datasets": {"vla_data": {
            "dataset_py": "hb_dit4dit_cached", "data_root_dir": str(D.data_root()), "data_mix": p["task"],
            "include_state": True, "per_device_batch_size": p["per_gpu_batch"],
            "max_state_dim": p["state_dim"], "max_action_dim": p["action_dim"],
            "video_delta_indices": list(D.VIDEO_DELTA), "action_video_freq_ratio": D.VIDEO_RATIO,
            "image_size": [D.IMG, D.IMG], "obs": ["image_0"],
        }},
        "trainer": {
            "max_train_steps": steps, "num_warmup_steps": p["warmup_steps"],
            "save_interval": 1,  # hb_patches gates saving to plan["save_steps"]
            "eval_interval": steps + 1, "logging_frequency": 1 if probe else 10,
            "learning_rate": dict(LR), "lr_scheduler_type": "cosine_with_min_lr",
            "scheduler_specific_kwargs": {"min_lr": 5.0e-07},
            "freeze_modules": FREEZE, "repeated_diffusion_steps": 4,
            "gradient_clipping": 1.0, "gradient_accumulation_steps": 1,
            "optimizer": {"name": "AdamW", "betas": [0.9, 0.95], "eps": 1.0e-08, "weight_decay": 1.0e-08},
            "is_resume": bool(resume), "pretrained_checkpoint": None, "reload_modules": None,
        },
    }


def make_cfg(p: dict, **kw):
    from omegaconf import OmegaConf

    base = OmegaConf.load(str(native_yaml()))
    return OmegaConf.merge(base, OmegaConf.create(overrides(p, **kw)))


def ds_config(per_gpu_batch: int, world: int) -> dict:
    """DiT4DiT/config/deepseeds/ds_config.yaml with the 'auto' batch sizes made explicit."""
    return {
        "fp16": {"enabled": False}, "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": int(per_gpu_batch), "train_batch_size": int(per_gpu_batch) * int(world),
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 2, "allgather_partitions": True, "allgather_bucket_size": 5e8,
                              "reduce_scatter": True, "reduce_bucket_size": 5e8, "overlap_comm": True,
                              "contiguous_gradients": True, "cpu_offload": False},
        "gradient_clipping": 1.0, "steps_per_print": 10,
    }


def accelerate_yaml(ds_json: str, machines: int, processes: int) -> str:
    """DiT4DiT/config/deepseeds/deepspeed_zero2.yaml, pointing at our ds json (machine rank / ip / port on the CLI)."""
    return "\n".join([
        "compute_environment: LOCAL_MACHINE", "debug: false", "deepspeed_config:",
        f"  deepspeed_config_file: {ds_json}", "  deepspeed_multinode_launcher: standard", "  zero3_init_flag: false",
        "distributed_type: DEEPSPEED", "machine_rank: 0", "main_training_function: main",
        f"num_machines: {machines}", f"num_processes: {processes}", "rdzv_backend: static", "same_network: true",
        "use_cpu: false", ""])


def write_launch_files(task, batch, world, machines, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = out / "ds_config.json"
    ds.write_text(json.dumps(ds_config(batch, world), indent=2))
    acc = out / "accelerate.yaml"
    acc.write_text(accelerate_yaml(str(ds), machines, world))
    print(json.dumps({"ds_config": str(ds), "accelerate_yaml": str(acc), "task": task}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("plan")
    s.add_argument("task", choices=sorted(D.TASKS))
    s.add_argument("batch", type=int)
    s.add_argument("--world", type=int, default=32)
    s = sub.add_parser("launch-files")
    s.add_argument("--task", required=True, choices=sorted(D.TASKS))
    s.add_argument("--batch", type=int, required=True)
    s.add_argument("--world", type=int, required=True)
    s.add_argument("--machines", type=int, required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "plan":
        print(json.dumps(plan(a.task, a.batch, a.world), indent=2))
    else:
        write_launch_files(a.task, a.batch, a.world, a.machines, a.out)


if __name__ == "__main__":
    main()
