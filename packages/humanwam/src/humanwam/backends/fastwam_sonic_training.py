"""Fixed G1 measured-Inspire recipe on the pinned native Gaussian FM trainer.

Configuration composition is CPU-only. Torch and the native runtime are loaded
only by the explicit training entrypoint, never by command construction.
"""

from __future__ import annotations

import argparse
import importlib
import math
import os
import sys
from pathlib import Path

TASK = "libero_uncond_2cam224_1e-4"
PROFILE = "feedback76_h16"
UPSTREAM_COMMIT = "7faa71108368fbb3b6885649f112af607427a2d4"
SHAPE_META = {
    "images": [
        {"key": name, "raw_shape": [3, 224, 224], "shape": [3, 224, 224]}
        for name in ("ego_view", "chest")
    ],
    "action": [{"key": "default", "raw_shape": 76, "shape": 76}],
    "state": [{"key": "default", "raw_shape": 44, "shape": 44}],
}
SAFE_OVERRIDES = {
    "batch_size", "num_workers", "num_epochs", "max_steps", "learning_rate",
    "gradient_accumulation_steps", "weight_decay", "seed", "save_every", "eval_every",
    "log_every", "wandb.enabled", "model.mot_checkpoint_mixed_attn",
    "model.compile_training_denoise",
}


def _root(source_dir):
    root = Path(source_dir).expanduser().resolve()
    if not (root / "configs/train.yaml").is_file() or not (root / "scripts/train.py").is_file():
        raise FileNotFoundError(f"Pinned FastWAM checkout missing: {root}")
    return root


def compose_sonic_config(source_dir, overrides=None):
    """Return the exact H16/76D/44D config; no model, dataset or Torch imports.

    ``overrides`` is an internal mapping for launch paths and supported training
    settings. Contract fields are checked after merge to prevent schema drift.
    """
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    root = _root(source_dir)
    with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=["task=" + TASK])
    fixed = {
        "data.train.shape_meta": SHAPE_META,
        "data.train.num_frames": 17,
        "data.train.global_sample_stride": 1,
        "data.train.action_video_freq_ratio": 4,
        "data.train.video_size": [224, 448],
        "data.train.concat_multi_camera": "horizontal",
        "data.train.processor.action_output_dim": 76,
        "data.train.processor.proprio_output_dim": 44,
        "data.train.processor.delta_action_dim_mask": {"default": [False] * 76},
        "data.train.processor.action_state_transforms": None,
        "data.train.processor.use_stepwise_action_norm": False,
        "data.train.processor.norm_default_mode": "min/max",
        "data.train.processor.norm_exception_mode": None,
        "model.redirect_common_files": False,
        "model.action_scheduler.train_shift": 1.0,
        "model.action_scheduler.infer_shift": 1.0,
    }
    for key, value in {**fixed, **(overrides or {})}.items():
        OmegaConf.update(cfg, key, value, merge=False)
    # Do not let callers accidentally change semantics while retaining a name.
    for key, expected in fixed.items():
        actual = OmegaConf.select(cfg, key)
        if isinstance(actual, (dict, list)) or OmegaConf.is_config(actual):
            actual = OmegaConf.to_container(actual, resolve=True)
        if actual != expected:
            raise ValueError(f"Protected FastWAM SONIC recipe field: {key}")
    return cfg


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def training_overrides(config):
    """Validate the prepared data and turn explicit options into native fields."""
    from .fastwam_sonic_data import audit_compact_dataset

    if config.get("profile", PROFILE) != PROFILE:
        raise ValueError("FastWAM SONIC training requires feedback76_h16")
    if config.get("method", "gaussian_fm") != "gaussian_fm":
        raise ValueError("FastWAM SONIC training uses native Gaussian FM")
    unsupported = {"task", "action_dim", "state_dim", "horizon", "action_schema"}.intersection(config)
    if unsupported:
        raise ValueError("Fixed FastWAM SONIC recipe does not accept " + ", ".join(sorted(unsupported)))
    data = Path(config["dataset_dir"]).expanduser().resolve(strict=True)
    audit_compact_dataset(data)
    text_cache = Path(config["text_embedding_cache_dir"]).expanduser().resolve(strict=True)
    if not text_cache.is_dir():
        raise ValueError("text_embedding_cache_dir must be a prepared directory")
    initializer = Path(config["action_init"]).expanduser().resolve(strict=True)
    if not initializer.is_file() or initializer.suffix != ".pt":
        raise ValueError("action_init must be a native ActionDiT backbone .pt initializer")
    output = Path(config["output_dir"]).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())) and not config.get("resume"):
        raise ValueError("Use a fresh FastWAM SONIC output_dir or an explicit resume checkpoint")
    values = config.get("overrides", {})
    if not isinstance(values, dict) or set(values) - SAFE_OVERRIDES:
        raise ValueError("Unsupported SONIC training overrides; contract/model/data fields are fixed")
    for key, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Training overrides must be finite")
        if key in {"batch_size", "num_epochs", "max_steps", "gradient_accumulation_steps", "save_every", "eval_every", "log_every"}:
            _positive_int(value, key)
        if key == "num_workers" and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError("num_workers must be a nonnegative integer")
        if key in {"learning_rate", "weight_decay"}:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (key == "learning_rate" and value == 0):
                raise ValueError(f"{key} must be a finite {'positive' if key == 'learning_rate' else 'nonnegative'} number")
        if key in {"wandb.enabled", "model.mot_checkpoint_mixed_attn", "model.compile_training_denoise"} and not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
    result = {
        "output_dir": str(output),
        "data.train.dataset_dirs": [str(data)],
        "data.train.text_embedding_cache_dir": str(text_cache),
        "model.action_dit_pretrained_path": str(initializer),
        **values,
    }
    if config.get("resume"):
        result["resume"] = str(Path(config["resume"]).expanduser().resolve(strict=True))
    return result


def build_train_command(config, source_dir):
    """Construct the Accelerate command; no output writes or training happen here."""
    root = _root(source_dir)
    training_overrides(config)
    launcher = Path(config["launcher_config"]).expanduser().resolve(strict=True)
    count = _positive_int(config.get("num_processes", 8), "num_processes")
    package_src = Path(__file__).resolve().parents[2]
    cache = Path(config["model_cache_root"]).expanduser().resolve(strict=True)
    return {
        "argv": [
            config.get("python", "python3"), "-m", "accelerate.commands.launch",
            "--config_file", str(root / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"),
            "--num_processes", str(count), "--module",
            "humanwam.backends.fastwam_sonic_training", "--config", str(launcher),
        ],
        "cwd": str(root),
        "env": {
            "PYTHONPATH": os.pathsep.join([str(package_src), str(root / "src"), os.environ.get("PYTHONPATH", "")]),
            "DIFFSYNTH_MODEL_BASE_PATH": str(cache), "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--compose-only", action="store_true", help="Print the resolved native recipe without loading models")
    args = parser.parse_args(argv)
    from humanwam.config import load_config

    launcher = load_config(args.config)
    if launcher["backend"] != "fastwam_sonic":
        raise ValueError("Use a fastwam_sonic training config")
    options = launcher["options"]
    root = _root(launcher["source_dir"])
    cfg = compose_sonic_config(root, training_overrides(options))
    from omegaconf import OmegaConf

    if args.compose_only:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return 0
    sys.path.insert(0, str(root / "src"))
    native = importlib.import_module("fastwam.runtime")
    if Path(native.__file__).resolve() != root / "src/fastwam/runtime.py":
        raise RuntimeError("A different FastWAM installation shadows the pinned source")
    from fastwam.utils.config_resolvers import register_default_resolvers

    register_default_resolvers()
    # The same offline settings are used by the direct module invocation.
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(Path(options["model_cache_root"]).expanduser().resolve(strict=True))
    os.environ["DIFFSYNTH_SKIP_DOWNLOAD"] = "true"
    native.run_training(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
