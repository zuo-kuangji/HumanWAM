"""Runtime patches on the pinned DiT4DiT trainer (Mondo-Robotics/DiT4DiT @ 66a6f3a); no upstream file is modified.

install_pre() -- before DiT4DiT/training/train.py is executed (it builds its Accelerator at import time):
 1. accelerate.DeepSpeedPlugin: train.py does `DeepSpeedPlugin(hf_ds_config="DiT4DiT/config/deepseeds/ds_config.yaml")`;
    we hand it our JSON instead (same ZeRO-2 / bf16 / clipping values, explicit micro batch, because the data loader no
    longer goes through accelerator.prepare). The original class is restored right after train.py is loaded.
 2. DiT4DiT.dataloader.build_dataloader (bound by name into train.py): HB cached dataset + per-pass rank-sharded batch
    stream (hb_d4d_data.HBLoader). Rank 0 writes <run>/dataset_statistics.json in the native schema.
 3. Cosmos25FeatureExtractor._get_prompt_embeds: per-prompt memo while the text encoder is frozen. Same function
    (each prompt padded to max_length=512 independently, no attention mask, Qwen2.5-VL has no dropout), so only the
    7B text-encoder forward per sample is saved. HB_PROMPT_CACHE=0 disables it.
install_post(train_module) -- after train.py is loaded:
 4. TrainerUtils.setup_distributed_training -> accelerator.prepare(model, optimizer); the HB loader is passed through.
 5. VLATrainer._create_data_iterators -> HBLoader.stream(start_step=completed_steps) (never exhausts, exact resume).
 6. VLATrainer._save_checkpoint -> only at plan["save_steps"] (2026-11-12: steps 2000 and 4000): the native weights save
    (checkpoints/steps_<N>_pytorch_model.pt + config.yaml + summary.jsonl), then on all ranks
    accelerator.save_state(checkpoints/steps_<N>_resume, exclude_frozen_parameters=True) (DeepSpeed module + optimizer,
    RNG); every older *_resume dir is deleted -> resume state only for the latest checkpoint (user rule 2026-11-10).
 7. VLATrainer.prepare_training -> native (incl. native weights resume + LR fast-forward when trainer.is_resume), then
    accelerator.load_state(steps_<N>_resume, load_module_strict=False) when that dir exists (optimizer moments, RNG).
 8. VLATrainer._finalize_training -> final_model/pytorch_model.pt = hard link (else symlink) to the last step's weights
    instead of a second ~22 GB copy; <run>/HB_DONE.
 9. VLATrainer._log_metrics -> also <run>/hb_metrics.jsonl (rank 0) with peak GPU memory; step timestamps for the probe.
 10. probe mode (plan["probe"]): no checkpoint / final-model I/O.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import torch

PLAN: dict = {}
STEP_T: list = []
_ORIG_PLUGIN = None


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def install_pre(plan: dict, ds_json: str, *, augment: bool, num_workers: int) -> None:
    PLAN.clear()
    PLAN.update(plan)
    global _ORIG_PLUGIN
    import accelerate

    if _ORIG_PLUGIN is None:
        _ORIG_PLUGIN = accelerate.DeepSpeedPlugin
    orig_plugin = _ORIG_PLUGIN

    def hb_deepspeed_plugin(*args, **kwargs):
        kwargs["hf_ds_config"] = ds_json
        return orig_plugin(*args, **kwargs)

    accelerate.DeepSpeedPlugin = hb_deepspeed_plugin

    # ---- 2. data
    import torch.distributed as dist

    import DiT4DiT.dataloader as dl
    import hb_d4d_data as D

    def hb_build_dataloader(cfg, dataset_py=None):  # noqa: ARG001 (native signature)
        rank = dist.get_rank() if dist.is_initialized() else 0
        world = dist.get_world_size() if dist.is_initialized() else 1
        bs = int(cfg.datasets.vla_data.per_device_batch_size)
        assert bs == PLAN["per_gpu_batch"] and world == PLAN["world_size"], (bs, world, PLAN)
        ds = D.HBD4DDataset(PLAN["task"], augment=augment)
        assert ds.layout.action_dim == int(cfg.framework.action_model.action_dim) == int(cfg.datasets.vla_data.max_action_dim)
        assert ds.layout.state_dim == int(cfg.framework.action_model.state_dim) == int(cfg.datasets.vla_data.max_state_dim)
        assert list(cfg.datasets.vla_data.video_delta_indices) == D.VIDEO_DELTA
        assert int(cfg.datasets.vla_data.action_video_freq_ratio) == D.VIDEO_RATIO
        loader = D.HBLoader(ds, per_gpu_batch=bs, rank=rank, world=world, num_workers=num_workers,
                            seed=int(os.environ.get("HB_DATA_SEED", "42")))
        assert len(loader) == PLAN["batches_per_pass"], (len(loader), PLAN["batches_per_pass"])
        if rank == 0:
            ds.save_dataset_statistics(Path(cfg.output_dir) / "dataset_statistics.json")
        return loader

    dl.build_dataloader = hb_build_dataloader

    # ---- 3. prompt-embedding memo
    from DiT4DiT.model.modules.vlm import Cosmos25 as C25

    Ext = C25.Cosmos25FeatureExtractor
    orig_embeds = getattr(Ext._get_prompt_embeds, "_hb_orig", Ext._get_prompt_embeds)

    def hb_get_prompt_embeds(self, prompt, *, device, dtype, max_sequence_length):
        if os.environ.get("HB_PROMPT_CACHE", "1") != "1" or any(p.requires_grad for p in self.text_encoder.parameters()):
            return orig_embeds(self, prompt, device=device, dtype=dtype, max_sequence_length=max_sequence_length)
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        cache = self.__dict__.setdefault("_hb_prompt_cache", {})
        out = []
        for p in prompts:
            key = (p, int(max_sequence_length), str(dtype), str(device))
            if key not in cache:
                with torch.no_grad():
                    cache[key] = orig_embeds(self, [p], device=device, dtype=dtype,
                                             max_sequence_length=max_sequence_length).detach()
                if _rank() == 0:
                    print(f"[hb] cached prompt embedding {tuple(cache[key].shape)} for {p!r}", flush=True)
            out.append(cache[key])
        return torch.cat(out, dim=0)

    hb_get_prompt_embeds._hb_orig = orig_embeds
    Ext._get_prompt_embeds = hb_get_prompt_embeds


def restore_plugin() -> None:
    import accelerate

    if _ORIG_PLUGIN is not None:
        accelerate.DeepSpeedPlugin = _ORIG_PLUGIN


def _link_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            os.symlink(os.path.relpath(src, dst.parent), dst)
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def install_post(train) -> None:
    from DiT4DiT.training.trainer_utils.trainer_tools import TrainerUtils

    VT = train.VLATrainer
    probe = bool(PLAN.get("probe"))
    save_steps = set(int(s) for s in PLAN.get("save_steps", []))

    # ---- 4. prepare only model + optimizer
    def setup_distributed_training(accelerator, *components):
        model, optimizer, loader = components
        model, optimizer = accelerator.prepare(model, optimizer)
        return model, optimizer, loader

    TrainerUtils.setup_distributed_training = staticmethod(setup_distributed_training)

    # ---- 5. data stream
    def _create_data_iterators(self):
        self.vla_iter = self.vla_train_dataloader.stream(start_step=self.completed_steps)
        if self.accelerator.is_main_process:
            print(f"[hb] data stream starts at step {self.completed_steps} "
                  f"(pass {self.completed_steps // len(self.vla_train_dataloader)}, "
                  f"{len(self.vla_train_dataloader)} batches/pass)", flush=True)

    VT._create_data_iterators = _create_data_iterators

    # ---- 6. checkpoints
    orig_save = VT._save_checkpoint

    def _save_checkpoint(self):
        step = int(self.completed_steps)
        if probe or step not in save_steps:
            return
        t0 = time.time()
        orig_save(self)  # main: steps_<N>_pytorch_model.pt, config.yaml, summary.jsonl; then wait_for_everyone
        ckdir = Path(self.checkpoint_dir)
        rdir = ckdir / f"steps_{step}_resume"
        if self.accelerator.is_main_process:
            shutil.rmtree(rdir, ignore_errors=True)
            rdir.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(str(rdir), exclude_frozen_parameters=True)
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            (rdir / "hb_resume.json").write_text(json.dumps({"step": step, "time": time.time(), "plan": PLAN}, indent=2))
            for d in ckdir.glob("steps_*_resume"):
                if d.is_dir() and d.name != rdir.name:
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[hb] dropped resume state {d} (resume state only for the latest checkpoint)", flush=True)
            print(f"[hb] checkpoint step {step} saved in {time.time() - t0:.0f}s", flush=True)
        self.accelerator.wait_for_everyone()

    VT._save_checkpoint = _save_checkpoint

    # ---- 7. resume optimizer / RNG
    orig_prepare = VT.prepare_training

    def prepare_training(self):
        orig_prepare(self)
        if self.completed_steps > 0:
            rdir = Path(self.checkpoint_dir) / f"steps_{self.completed_steps}_resume"
            if rdir.is_dir() and (rdir / "hb_resume.json").exists():
                self.accelerator.load_state(str(rdir), load_module_strict=False)
                if self.accelerator.is_main_process:
                    print(f"[hb] restored optimizer/RNG state from {rdir}", flush=True)
            elif self.accelerator.is_main_process:
                print(f"[hb] WARNING no resume state {rdir}: weights-only resume (native behaviour, fresh optimizer)",
                      flush=True)

    VT.prepare_training = prepare_training

    # ---- 8. final model
    orig_finalize = VT._finalize_training

    def _finalize_training(self):
        if probe:
            if self.accelerator.is_main_process:
                train.wandb.finish()
            self.accelerator.wait_for_everyone()
            return
        last = Path(self.checkpoint_dir) / f"steps_{int(self.completed_steps)}_pytorch_model.pt"
        if not last.exists():
            return orig_finalize(self)
        if self.accelerator.is_main_process:
            how = _link_or_copy(last, Path(self.config.output_dir) / "final_model" / "pytorch_model.pt")
            (Path(self.config.output_dir) / "HB_DONE").write_text(json.dumps(
                {"final_step": int(self.completed_steps), "final_model": how, "plan": PLAN, "time": time.time()}, indent=2))
            train.wandb.finish()
        self.accelerator.wait_for_everyone()

    VT._finalize_training = _finalize_training

    # ---- 9. metrics
    orig_log = VT._log_metrics
    every = 1 if probe else int(os.environ.get("HB_LOG_EVERY", "10"))

    def _log_metrics(self, metrics):
        orig_log(self, metrics)
        STEP_T.append(time.perf_counter())
        if not self.accelerator.is_main_process or self.completed_steps % every:
            return
        rec = {"step": int(self.completed_steps), "time": time.time(),
               "max_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if torch.cuda.is_available() else None}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                v = v.item()
            if isinstance(v, (int, float)):
                rec[k] = float(v)
        with open(Path(self.config.output_dir) / "hb_metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    VT._log_metrics = _log_metrics


def latest_weights(ckdir: Path):
    best = None
    for f in Path(ckdir).glob("steps_*_pytorch_model.pt"):
        m = re.match(r"steps_(\d+)_pytorch_model\.pt$", f.name)
        if m and (best is None or int(m.group(1)) > best[0]):
            best = (int(m.group(1)), f)
    return best
