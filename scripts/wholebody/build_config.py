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


JOINT_MODEL_TARGET = (
    "imagewam.wholebody_joint_reference.create_imagewam_flux2_wholebody_joint_reference"
)
JOINT_DATASET_TARGET = (
    "imagewam.datasets.wholebody_joint_reference.JointReferenceLeRobotDataset"
)
JOINT_NORMALIZATION = (
    "fit_joint_reference_body29_orientation6d_q01q99_fixed_inspire_hands"
)


def canonical_representation(value):
    return {"motion_token": "motion76", "joint": "joint47"}.get(value, value)


def recipe_spec(stats, representation="motion76"):
    representation = canonical_representation(representation)
    if representation == "motion76":
        return {
            "action_dim": 76,
            "model_target": MODEL_TARGET,
            "dataset_target": DATASET_TARGET,
            "stats_schema": STATS_SCHEMA,
            "normalization": NORMALIZATION,
            "action_stride": 1,
        }
    require(representation in ("joint41", "joint47"), "Unknown action representation")
    dim = int(representation[5:])
    orientation = "external" if dim == 41 else "predicted_rotation6d"
    body = stats.get("body_target")
    require(
        body in ("wbc", "measured", "direct_reference")
        and stats.get("orientation_target") == orientation,
        "Joint statistics body/orientation labels differ from the requested representation",
    )
    if stats.get("recipe_version") == "v2":
        from imagewam.joint_reference_contract import (
            NORMALIZATION_V2,
            expected_schema,
            make_joint_recipe,
            source_contract_sha256,
        )

        recipe = make_joint_recipe(
            stats.get("reference_source_contract"), orientation_target=orientation
        )
        require(
            stats.get("joint_recipe") == recipe,
            "Joint v2 recipe/provenance changed; prepare new statistics",
        )
        require(
            stats.get("reference_source_contract_sha256")
            == source_contract_sha256(recipe["reference_source_contract"]),
            "Joint v2 reference source contract digest differs",
        )
        require(
            all(
                stats.get(k) == v
                for k, v in {
                    "fps": 30,
                    "action_stride": 2,
                    "action_horizon": 40,
                    "past_action_size": 16,
                    "execution_horizon": 16,
                    "future_image_offset": 32,
                }.items()
            ),
            "Joint v2 requires prediction40/history16/execution16 with genuine lookahead",
        )
        return {
            "action_dim": dim,
            "model_target": JOINT_MODEL_TARGET,
            "dataset_target": JOINT_DATASET_TARGET,
            "stats_schema": expected_schema(body, orientation, "v2"),
            "normalization": NORMALIZATION_V2,
            "action_stride": 2,
            "body_target": body,
            "orientation_target": orientation,
            "recipe_version": "v2",
            "joint_recipe": recipe,
        }
    label = "wbc29" if body == "wbc" else "measured29"
    suffix = "external-orientation" if dim == 41 else "quatwxyz-rotation6d-columns"
    schema = f"wholebody-state44-joint{dim}-{label}-measured-inspire12-{suffix}-v1"
    require(
        stats.get("fps") == 30 and stats.get("action_stride") == 2,
        "Joint H16 uses stride2 at30FPS, spanning1s",
    )
    require(
        stats.get("action_horizon") == stats.get("past_action_size") == 16
        and stats.get("future_image_offset") == 32,
        "Joint statistics must be prepared for H16/history16 and future image offset32",
    )
    return {
        "action_dim": dim,
        "model_target": JOINT_MODEL_TARGET,
        "dataset_target": JOINT_DATASET_TARGET,
        "stats_schema": schema,
        "normalization": JOINT_NORMALIZATION,
        "action_stride": 2,
        "body_target": body,
        "orientation_target": orientation,
    }


def validate_statistics(stats, spec=None):
    spec = spec or recipe_spec(stats)
    require(
        stats.get("schema") == spec["stats_schema"],
        "Statistics use a different whole-body schema",
    )
    require(
        stats.get("normalization_strategy") == spec["normalization"],
        "Fit this dataset's own motion/body/gravity statistics; do not reuse a task's statistics",
    )
    require(
        stats.get("action_dim") == spec["action_dim"]
        and stats.get("proprio_dim") == 44,
        "Expected the selected action dimension and current state44 statistics",
    )
    require(
        stats.get("train_split_mode") == "all_episodes"
        and stats.get("heldout_episode_ids") == [],
        "This recipe trains on all episodes, without a held-out split",
    )
    episodes = stats.get("train_episode_ids", [])
    require(
        bool(episodes) and len(episodes) == len(set(episodes)),
        "Statistics need nonempty, unique training episodes",
    )
    require(
        isinstance(stats.get("train_frames"), int) and stats["train_frames"] > 0,
        "Statistics must declare a positive training frame count",
    )
    require(
        stats.get("hand_target_time_offset") == 0,
        "Measured Inspire hand labels must use the same target row",
    )
    if spec.get("recipe_version") == "v2":
        require(
            stats.get("orientation_prediction_frame") == "current_heading_relative"
            and stats.get("orientation_anchor_source_key")
            == "observation.root_orientation",
            "Joint v2 requires a current-IMU-heading anchor, not an unobserved global-yaw target",
        )
    elif spec["action_dim"] == 47:
        require(
            stats.get("orientation_validity_rule")
            == "smpl_frame_index>0_and_stream_mode_in_1_4"
            and stats.get("orientation_validity_source_keys")
            == ["teleop.smpl_frame_index", "teleop.stream_mode"]
            and stats.get("orientation_invalid_policy")
            == "mask_only_last6_future_and_zero_last6_normalized_future_history"
            and stats.get("orientation_statistics_use_valid_rows_only") is True
            and stats.get("orientation_validity_rejects_missing_flags") is True
            and stats.get("orientation_age_threshold_applied") is False,
            "Joint47 statistics must explicitly reject recorder fallback and mask invalid orientation labels",
        )
        valid, invalid = (
            stats.get("orientation_statistics_valid_frames"),
            stats.get("orientation_statistics_invalid_frames"),
        )
        require(
            type(valid) is int
            and type(invalid) is int
            and valid > 0
            and invalid >= 0
            and valid + invalid == stats["train_frames"],
            "Joint47 orientation validity counts differ from this collection",
        )
    if spec["action_dim"] != 76:
        require(
            isinstance(stats.get("instruction"), str) and stats["instruction"].strip(),
            "Joint statistics need their own nonempty recorded instruction",
        )
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
    for flag in (
        "dataset-dir",
        "stats-path",
        "text-cache-dir",
        "output-dir",
        "flux2-src",
        "flux2-model",
        "ae-model",
        "qwen-model",
        "action-initializer",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument(
        "--representation",
        choices=("motion_token", "joint", "motion76", "joint41", "joint47"),
        default="motion_token",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--num-processes", type=int, default=8)
    augmentation = parser.add_mutually_exclusive_group()
    augmentation.add_argument(
        "--augmentation-config", type=Path,
        help="Training-only augmentation YAML; defaults to configs/wholebody/augmentation_v2.yaml",
    )
    augmentation.add_argument(
        "--no-augmentation", action="store_true",
        help="Explicitly disable training augmentation while keeping the legacy inference path",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        choices=(16, 40),
        default=None,
        help="Prediction length: motion defaults16; joint v2 predicts40 including a24-point lookahead tail",
    )
    parser.add_argument(
        "--execution-horizon",
        type=int,
        choices=(16,),
        default=16,
        help="Joint v2 executes16 source intervals; prediction length includes a separate lookahead tail",
    )
    return parser


def resolve_training_augmentation(args):
    from imagewam.augmentation_config import SCHEMA, augmentation_sha256, normalize_augmentation
    from omegaconf import OmegaConf

    custom = getattr(args, "augmentation_config", None)
    disabled = getattr(args, "no_augmentation", False)
    require(type(disabled) is bool, "no_augmentation must be an explicit boolean")
    require(not (disabled and custom is not None), "Choose an augmentation profile or --no-augmentation")
    if disabled:
        config, source = normalize_augmentation(None), None
    else:
        profile = (Path(custom) if custom is not None else
                   Path(__file__).resolve().parents[2] / "configs/wholebody/augmentation_v2.yaml")
        profile = profile.expanduser().resolve(strict=True)
        config = normalize_augmentation(OmegaConf.to_container(OmegaConf.load(profile), resolve=True))
        source = str(profile)
    require(not config["enabled"] or config["schema"] == SCHEMA,
            "New training requires the whole-state conditioning gate profile, not legacy dimension dropout")
    return config, augmentation_sha256(config), source


def build(args, *, measure_loader=measure_native_loader):
    from omegaconf import OmegaConf

    for name in ("epochs", "batch_size", "accumulation", "num_processes"):
        require(getattr(args, name) > 0, name + " must be positive")
    augmentation, augmentation_digest, augmentation_source = resolve_training_augmentation(args)
    paths = {
        name: getattr(args, name).expanduser().resolve(strict=True)
        for name in (
            "dataset_dir",
            "stats_path",
            "text_cache_dir",
            "flux2_src",
            "flux2_model",
            "ae_model",
            "qwen_model",
            "action_initializer",
        )
    }
    for name in ("dataset_dir", "text_cache_dir", "flux2_src", "qwen_model"):
        require(paths[name].is_dir(), name + " must be a local directory")
    for name in ("stats_path", "flux2_model", "ae_model", "action_initializer"):
        require(paths[name].is_file(), name + " must be a local file")
    root = args.output_dir.expanduser().resolve()
    require(
        not root.exists(),
        "Refusing an existing output directory; choose a new independent run",
    )
    require(
        not root.is_relative_to(paths["dataset_dir"]),
        "Do not write run outputs inside the source dataset",
    )
    stats = json.loads(paths["stats_path"].read_text())
    representation = canonical_representation(
        getattr(args, "representation", "motion76")
    )
    spec = recipe_spec(stats, representation)
    frames = validate_statistics(stats, spec)
    template_name = (
        "deltaflow76_p25.yaml"
        if representation == "motion76"
        else "deltaflow_joint_reference_v2_p25.yaml"
        if spec.get("recipe_version") == "v2"
        else "deltaflow_joint_reference_p25.yaml"
    )
    template = Path(__file__).resolve().parents[2] / "configs/wholebody" / template_name
    cfg = OmegaConf.load(template)
    dim, stride = spec["action_dim"], spec["action_stride"]
    cfg.model._target_ = spec["model_target"]
    cfg.data.train._target_ = spec["dataset_target"]
    cfg.model.expected_action_dim = cfg.model.action_dit_config.action_dim = dim
    if representation != "motion76":
        cfg.data.train.body_target = cfg.model.body_target = spec["body_target"]
        cfg.data.train.orientation_target = cfg.model.orientation_target = spec[
            "orientation_target"
        ]
        cfg.data.train.action_stride = cfg.model.action_stride = stride
        cfg.data.train.future_image_offset = 32
        cfg.model.reference_fps = 30
        cfg.model.statistics_sha256 = sha256(paths["stats_path"])
        cfg.model.instruction_sha256 = hashlib.sha256(
            stats["instruction"].encode()
        ).hexdigest()
        cfg.model.state_only_schema_id = (
            f"wholebody.joint_reference{dim}.{spec['body_target']}29.state44.p25.v1"
        )
        if spec.get("recipe_version") == "v2":
            recipe = spec["joint_recipe"]
            cfg.model.recipe_version = cfg.data.train.recipe_version = "v2"
            cfg.model.past_action_horizon = 16
            cfg.model.execution_horizon = cfg.data.train.execution_horizon = 16
            cfg.model.future_image_offset = 32
            cfg.model.source_contract = cfg.data.train.source_contract = recipe[
                "reference_source_contract"
            ]
            cfg.model.source_contract_sha256 = recipe[
                "reference_source_contract_sha256"
            ]
            cfg.model.joint_recipe = recipe
            cfg.model.state_only_schema_id = (
                f"wholebody.joint_reference{dim}.{spec['body_target']}29.state44.p25.v2"
            )
        cfg.wandb.name = f"deltaflow-joint{dim}-p25"
        cfg.wandb.group = f"wholebody-joint{dim}-state44"
    cfg.batch_size = args.batch_size
    cfg.gradient_accumulation_steps = args.accumulation
    cfg.num_epochs = args.epochs
    horizon = getattr(args, "horizon", None)
    horizon = (
        (40 if spec.get("recipe_version") == "v2" else 16)
        if horizon is None
        else horizon
    )
    require(
        type(horizon) is int and horizon in (16, 40),
        "Horizon must be16 or explicit legacy40",
    )
    require(
        representation == "motion76"
        or horizon == (40 if spec.get("recipe_version") == "v2" else 16),
        "Joint v2 predicts40 including its tail; legacy joint v1 requires16",
    )
    history = 16 if spec.get("recipe_version") == "v2" else horizon
    cfg.data.train.action_horizon = horizon
    cfg.data.train.past_action_size = history
    cfg.model.expected_action_horizon = horizon
    cfg.EVALUATION = {"action_horizon": horizon, "num_inference_steps": 1}
    cfg.data.train.dataset_dir = str(paths["dataset_dir"])
    cfg.data.train.stats_path = str(paths["stats_path"])
    cfg.data.train.qwen_text_cache_dir = str(paths["text_cache_dir"])
    cfg.data.train.augmentation = augmentation
    for key, name in (
        ("flux2_src_path", "flux2_src"),
        ("flux2_model_path", "flux2_model"),
        ("ae_model_path", "ae_model"),
        ("qwen3_model_spec", "qwen_model"),
        ("action_dit_pretrained_path", "action_initializer"),
    ):
        cfg.model[key] = str(paths[name])
    cfg.model.action_initializer_sha256 = sha256(paths["action_initializer"])
    cfg.model.loss.action_a2a.source_noise_dim_mask = [True] * dim
    cfg.wholebody_contract = {
        "world_size": args.num_processes,
        "zero_stage": 1,
        "action_horizon": horizon,
        "past_action_horizon": history,
        "statistics_sha256": sha256(paths["stats_path"]),
        "stats_schema": spec["stats_schema"],
        "representation": representation,
        "action_dim": dim,
        "action_stride": stride,
        "normalization_strategy": spec["normalization"],
        "smoke_output_dir": str(root / "smoke"),
        "production_output_dir": str(root / "production"),
        "fresh_base": True,
        "training_augmentation": augmentation,
        "training_augmentation_sha256": augmentation_digest,
    }
    if spec.get("recipe_version") == "v2":
        cfg.wholebody_contract.joint_recipe = spec["joint_recipe"]
        cfg.wholebody_contract.execution_horizon = 16
    measurement = measure_loader(cfg)
    require(
        measurement["dataset_frames"] == frames,
        "Measured dataset length disagrees with its fitted statistics",
    )
    require(
        measurement["world_size"] == args.num_processes
        and measurement["microbatch"] == args.batch_size
        and measurement["gradient_accumulation"] == args.accumulation,
        "Native loader measurement does not match the requested distribution",
    )
    schedule = derive_schedule(measurement, args.epochs, args.accumulation)
    cfg.max_steps = cfg.lr_scheduler_total_steps = schedule["total_updates"]
    cfg.warmup_steps = schedule["warmup_updates"]
    cfg.save_every = schedule["updates_per_epoch"]
    cfg.wholebody_contract.measured_loader = measurement
    cfg.wholebody_contract.updates_per_epoch = schedule["updates_per_epoch"]
    cfg.output_dir = str(root / "production")
    production = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    smoke = copy.deepcopy(production)
    smoke.update(
        output_dir=str(root / "smoke"),
        max_steps=5,
        save_every=0,
        save_steps=[],
        save_at_end=False,
        log_every=1,
        rank_timer_every=1,
    )
    # Keep loader length, world, LR scheduler and initialization identical.
    root.mkdir(parents=True, exist_ok=False)
    config_paths = [root / "production.yaml", root / "smoke.yaml"]
    for path, payload in zip(config_paths, (production, smoke)):
        with path.open("x") as stream:
            OmegaConf.save(OmegaConf.create(payload), stream, resolve=True)
    receipt = {
        "status": "CONFIGS_BUILT_NO_TRAINING",
        "epochs": args.epochs,
        "action_horizon": horizon,
        "past_action_horizon": history,
        "representation": representation,
        "action_dim": dim,
        "action_stride": stride,
        "future_image_offset": int(
            cfg.data.train.get("future_image_offset", horizon * stride)
        ),
        "future_action_offsets": [0, (horizon - 1) * stride],
        "past_action_offsets": [-history * stride, -stride],
        "actual_native_loader": measurement,
        **schedule,
        "nominal_global_batch": args.batch_size
        * args.accumulation
        * args.num_processes,
        "production_config": str(config_paths[0]),
        "smoke_config": str(config_paths[1]),
        "production_output_dir": production["output_dir"],
        "smoke_output_dir": smoke["output_dir"],
        "config_sha256": {path.name: sha256(path) for path in config_paths},
        "statistics_sha256": cfg.wholebody_contract.statistics_sha256,
        "action_initializer_sha256": cfg.model.action_initializer_sha256,
        "initializer_scope": "Factory strictly audits the 114-key base-derived initializer; no task checkpoint or optimizer is loaded",
        "native_initial_lr": 1e-4 / schedule["warmup_updates"],
        "peak_lr": 1e-4,
        "cosine_min_lr": 1e-6,
        "validation_enabled": False,
        "visual_state_route": "none",
        "text_state_route": "none",
        "action_to_visual_gradients": False,
        "visual_source": "0.5 * z_current + sqrt(0.75) * epsilon",
        "action_source": f"normalized_past_action{dim} + 0.5 * epsilon_all{dim}",
        "checkpoint_retention": "latest four weight snapshots and latest one training state",
        "training_augmentation": augmentation,
        "training_augmentation_sha256": augmentation_digest,
        "training_augmentation_profile": augmentation_source,
    }
    if spec.get("recipe_version") == "v2":
        receipt.update(
            joint_recipe=spec["joint_recipe"],
            execution_horizon=16,
            lookahead_tail_horizon=24,
            prediction_span_seconds=39 * 2 / 30,
            execution_interval_seconds=2 / 30,
            action_source="normalized history16 then hold last prior24 + 0.5 * epsilon_all47",
        )
    with (root / "build_receipt.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    print(json.dumps(build(argument_parser().parse_args()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
