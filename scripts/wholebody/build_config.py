#!/usr/bin/env python3
"""Build fresh-base production/smoke configs using measured native loader length.

This command is CPU-only: it instantiates the dataset and data loaders, but does
not iterate the loader, construct a model, allocate GPU tensors or train.
Statistics and pure-instruction Qwen caches must already have been prepared.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

STATS_SCHEMA = "wholebody-state44-action76-sameframe-all-episodes-v1"
NORMALIZATION = "fit_wholebody_motion64_body29_gravity3_fixed_inspire_hands"
MODEL_TARGET = "imagewam.wholebody_delta.create_imagewam_flux2_wholebody_delta"
DATASET_TARGET = "imagewam.datasets.wholebody.WholeBodyLeRobotDataset"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_statistics(stats):
    require(stats.get("schema") == STATS_SCHEMA, "Statistics use a different whole-body schema")
    require(stats.get("normalization_strategy") == NORMALIZATION,
            "Fit this dataset's own motion/body/gravity statistics; do not reuse a task's statistics")
    require(stats.get("action_dim") == 76 and stats.get("proprio_dim") == 44,
            "Expected action76 and current state44 statistics")
    require(stats.get("train_split_mode") == "all_episodes"
            and stats.get("heldout_episode_ids") == [], "This recipe trains on all episodes, without a held-out split")
    episodes = stats.get("train_episode_ids", [])
    require(bool(episodes) and len(episodes) == len(set(episodes)), "Statistics need nonempty, unique training episodes")
    require(isinstance(stats.get("train_frames"), int) and stats["train_frames"] > 0,
            "Statistics must declare a positive training frame count")
    require(stats.get("hand_target_time_offset") == 0, "Measured Inspire hand labels must use the same target row")
    return stats["train_frames"]


def derive_schedule(measurement, epochs, accumulation):
    require(epochs > 0 and accumulation > 0, "Epochs and accumulation must be positive")
    updates = math.ceil(measurement["per_rank_microbatches"] / accumulation)
    total = updates * epochs
    require(total >= 5, "The production schedule must contain at least the five smoke-test updates")
    warmup = max(1, int(total * 0.05))
    require(warmup < total, "Warmup must leave at least one cosine update")
    return {"updates_per_epoch": updates, "total_updates": total, "warmup_updates": warmup}


def measure_native_loader(cfg):
    from accelerate.data_loader import prepare_data_loader
    from hydra.utils import instantiate
    from imagewam.trainer import Wan22Trainer

    dataset = instantiate(cfg.data.train)
    world = int(cfg.wholebody_contract.world_size)
    fake = SimpleNamespace(batch_size=int(cfg.batch_size), num_workers=int(cfg.num_workers),
        prefetch_factor=int(cfg.prefetch_factor), persistent_workers=bool(cfg.persistent_workers),
        seed=int(cfg.seed), accelerator=SimpleNamespace(num_processes=world, is_main_process=False))
    raw = Wan22Trainer._build_loader(fake, dataset, worker_init_fn=None)
    prepared = prepare_data_loader(raw, device=None, num_processes=world, process_index=0,
        split_batches=False, put_on_device=False, even_batches=True)
    return {"dataset_frames": len(dataset), "raw_loader_batches": len(raw),
        "per_rank_microbatches": len(prepared), "world_size": world,
        "microbatch": int(cfg.batch_size), "gradient_accumulation": int(cfg.gradient_accumulation_steps),
        "dataloader_drop_last": bool(raw.drop_last),
        "source": "native Wan22Trainer._build_loader + Accelerate.prepare_data_loader; no iteration/model/optimizer"}


def argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("dataset-dir", "stats-path", "text-cache-dir", "output-dir", "flux2-src",
                 "flux2-model", "ae-model", "qwen-model", "action-initializer"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--num-processes", type=int, default=8)
    return parser


def build(args, *, measure_loader=measure_native_loader):
    from omegaconf import OmegaConf

    for name in ("epochs", "batch_size", "accumulation", "num_processes"):
        require(getattr(args, name) > 0, name + " must be positive")
    paths = {name: getattr(args, name).expanduser().resolve(strict=True) for name in (
        "dataset_dir", "stats_path", "text_cache_dir", "flux2_src", "flux2_model", "ae_model", "qwen_model", "action_initializer")}
    for name in ("dataset_dir", "text_cache_dir", "flux2_src", "qwen_model"):
        require(paths[name].is_dir(), name + " must be a local directory")
    for name in ("stats_path", "flux2_model", "ae_model", "action_initializer"):
        require(paths[name].is_file(), name + " must be a local file")
    root = args.output_dir.expanduser().resolve()
    require(not root.exists(), "Refusing an existing output directory; choose a new independent run")
    require(not root.is_relative_to(paths["dataset_dir"]), "Do not write run outputs inside the source dataset")
    stats = json.loads(paths["stats_path"].read_text())
    frames = validate_statistics(stats)
    template = Path(__file__).resolve().parents[2] / "configs/wholebody/deltaflow76_p25.yaml"
    cfg = OmegaConf.load(template)
    cfg.batch_size = args.batch_size
    cfg.gradient_accumulation_steps = args.accumulation
    cfg.num_epochs = args.epochs
    cfg.data.train.dataset_dir = str(paths["dataset_dir"])
    cfg.data.train.stats_path = str(paths["stats_path"])
    cfg.data.train.qwen_text_cache_dir = str(paths["text_cache_dir"])
    for key, name in (("flux2_src_path", "flux2_src"), ("flux2_model_path", "flux2_model"),
                      ("ae_model_path", "ae_model"), ("qwen3_model_spec", "qwen_model"),
                      ("action_dit_pretrained_path", "action_initializer")):
        cfg.model[key] = str(paths[name])
    cfg.model.action_initializer_sha256 = sha256(paths["action_initializer"])
    cfg.model.loss.action_a2a.source_noise_dim_mask = [True] * 76
    cfg.wholebody_contract = {"world_size": args.num_processes, "zero_stage": 1,
        "statistics_sha256": sha256(paths["stats_path"]), "stats_schema": STATS_SCHEMA,
        "normalization_strategy": NORMALIZATION, "smoke_output_dir": str(root / "smoke"),
        "production_output_dir": str(root / "production"), "fresh_base": True}
    measurement = measure_loader(cfg)
    require(measurement["dataset_frames"] == frames, "Measured dataset length disagrees with its fitted statistics")
    require(measurement["world_size"] == args.num_processes
            and measurement["microbatch"] == args.batch_size
            and measurement["gradient_accumulation"] == args.accumulation,
            "Native loader measurement does not match the requested distribution")
    schedule = derive_schedule(measurement, args.epochs, args.accumulation)
    cfg.max_steps = cfg.lr_scheduler_total_steps = schedule["total_updates"]
    cfg.warmup_steps = schedule["warmup_updates"]
    cfg.save_every = schedule["updates_per_epoch"]
    cfg.wholebody_contract.measured_loader = measurement
    cfg.wholebody_contract.updates_per_epoch = schedule["updates_per_epoch"]
    cfg.output_dir = str(root / "production")
    production = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    smoke = copy.deepcopy(production)
    smoke.update(output_dir=str(root / "smoke"), max_steps=5, save_every=0,
                 save_steps=[], save_at_end=False, log_every=1, rank_timer_every=1)
    # Keep loader length, world, LR scheduler and initialization identical.
    root.mkdir(parents=True, exist_ok=False)
    config_paths = [root / "production.yaml", root / "smoke.yaml"]
    for path, payload in zip(config_paths, (production, smoke)):
        with path.open("x") as stream:
            OmegaConf.save(OmegaConf.create(payload), stream, resolve=True)
    receipt = {"status": "CONFIGS_BUILT_NO_TRAINING", "epochs": args.epochs,
        "actual_native_loader": measurement, **schedule,
        "nominal_global_batch": args.batch_size * args.accumulation * args.num_processes,
        "production_config": str(config_paths[0]), "smoke_config": str(config_paths[1]),
        "production_output_dir": production["output_dir"], "smoke_output_dir": smoke["output_dir"],
        "config_sha256": {path.name: sha256(path) for path in config_paths},
        "statistics_sha256": cfg.wholebody_contract.statistics_sha256,
        "action_initializer_sha256": cfg.model.action_initializer_sha256,
        "initializer_scope": "Factory strictly audits the 114-key base-derived initializer; no task checkpoint or optimizer is loaded",
        "native_initial_lr": 1e-4 / schedule["warmup_updates"], "peak_lr": 1e-4,
        "cosine_min_lr": 1e-6, "validation_enabled": False,
        "visual_state_route": "none", "text_state_route": "none", "action_to_visual_gradients": False,
        "visual_source": "0.5 * z_current + sqrt(0.75) * epsilon",
        "action_source": "normalized_past_action76 + 0.5 * epsilon_all76",
        "checkpoint_retention": "latest four weight snapshots and latest one training state"}
    with (root / "build_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    print(json.dumps(build(argument_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
