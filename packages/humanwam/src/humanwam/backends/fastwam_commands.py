"""Official Accelerate/Hydra training and explicit HumanWAM HTTP serving."""

import json
import math
import os
import re
from pathlib import Path


def _root(source_dir):
    root = Path(source_dir).expanduser().resolve()
    if not (root / "scripts/train.py").is_file():
        raise FileNotFoundError(f"Official FastWAM checkout missing: {root}")
    return root


def _overrides(values):
    if not isinstance(values, dict):
        raise TypeError("overrides must map native Hydra keys to values")
    result = []
    for key, value in values.items():
        if not re.fullmatch(r"\+{0,2}[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", key):
            raise ValueError("Invalid Hydra override key")
        if key.lstrip("+").split(".", 1)[0] in ("task", "output_dir", "resume"):
            raise ValueError("Set task/output_dir/resume explicitly, not through overrides")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Hydra overrides must be finite")
        result.append(key + "=" + json.dumps(value, allow_nan=False))
    return result


def build_train_command(config, source_dir):
    root = _root(source_dir)
    if config.get("method", "gaussian_fm") != "gaussian_fm":
        raise ValueError(
            "FastWAM training maintains native Gaussian FM; Delta-flow requires a separate audited implementation"
        )
    inference_fields = {"profile", "action_schema", "action_dim", "state_dim", "horizon"}
    unsupported = sorted(inference_fields.intersection(config))
    if unsupported:
        raise ValueError(
            "FastWAM training does not accept top-level inference settings: "
            + ", ".join(unsupported)
            + "; select a native task and explicit Hydra overrides instead. "
            "Custom overrides do not establish a supported G1/SONIC training recipe."
        )
    task = config.get("task", "libero_uncond_2cam224_1e-4")
    if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", task):
        raise ValueError("task must name a native configs/task YAML")
    if not (root / "configs/task" / (task + ".yaml")).is_file():
        raise FileNotFoundError(f"Native FastWAM task is not registered: {task}")
    count = int(config.get("num_processes", 8))
    if count < 1:
        raise ValueError("num_processes must be positive")
    output = Path(config["output_dir"]).expanduser().resolve()
    if (
        output.exists()
        and (not output.is_dir() or any(output.iterdir()))
        and not config.get("resume")
    ):
        raise ValueError("Use a fresh FastWAM output_dir or explicitly specify resume")
    argv = [
        config.get("python", "python3"),
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(root / "scripts/accelerate_configs/accelerate_zero1_ds.yaml"),
        "--num_processes",
        str(count),
        str(root / "scripts/train.py"),
        "task=" + task,
        "output_dir=" + json.dumps(str(output)),
    ]
    argv.extend(_overrides(config.get("overrides", {})))
    if config.get("resume"):
        resume = Path(config["resume"]).expanduser().resolve(strict=True)
        argv.append("resume=" + json.dumps(str(resume)))
    cache = Path(config["model_cache_root"]).expanduser().resolve()
    return {
        "argv": argv,
        "cwd": str(root),
        "env": {
            "PYTHONPATH": str(root / "src") + os.pathsep + os.environ.get("PYTHONPATH", ""),
            "DIFFSYNTH_MODEL_BASE_PATH": str(cache),
            "DIFFSYNTH_SKIP_DOWNLOAD": "true",
        },
    }


def build_serve_command(config, source_dir):
    root = _root(source_dir)
    if not config.get("launcher_config"):
        raise ValueError("FastWAM HTTP serve requires the HumanWAM launcher_config path")
    launcher = Path(config["launcher_config"]).expanduser().resolve(strict=True)
    port = int(config.get("port", 8094))
    if not 1 <= port <= 65535:
        raise ValueError("port must be in 1..65535")
    return {
        "argv": [
            config.get("python", "python3"),
            "-m",
            "humanwam.http_server",
            "--config",
            str(launcher),
            "--host",
            config.get("host", "127.0.0.1"),
            "--port",
            str(port),
        ],
        "cwd": str(root),
        "env": {},
    }
