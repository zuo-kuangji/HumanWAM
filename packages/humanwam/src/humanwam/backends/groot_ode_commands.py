"""Pinned ODE-POLICY G1 commands; construction never starts models or hardware."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

GR00T_SOURCE_COMMIT = "23ace64f17aa5015259b8609d371eb61a357c776"
ODE_SOURCE_COMMIT = "fd8bec470faf7793702533358867db02819e8f27"
NATIVE_CAMERAS = ["observation.images.ego_view", "observation.images.chest"]


def pin_source(root, expected_commit, label):
    """Check the exact clean tracked git source before importing runtime code."""
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"Cannot verify the pinned {label} git checkout: {root}") from exc
    if result.stdout.strip() != expected_commit:
        raise ValueError(f"Pinned {label} requires source at {expected_commit}")
    if status.stdout.strip():
        raise ValueError(f"Pinned {label} tracked source has local changes")
    return root


def source_root(source_dir):
    root = Path(source_dir).expanduser().resolve()
    for name in ("groot/src/ode_g1/config.py", "groot/src/ode_g1/cli.py"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"Pinned ODE-POLICY real-g1 source missing: {root / name}")
    return root


def _overrides(config):
    values = config.get("native_overrides", {})
    if not isinstance(values, dict):
        raise ValueError("native_overrides must map native dotted keys to JSON values")
    result = []
    for key, value in values.items():
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", key):
            raise ValueError("native_overrides keys must be native dotted configuration names")
        result.append(key + "=" + json.dumps(value, allow_nan=False))
    return result


def load_native_config(config, source_dir):
    """Run the pinned native validator without importing Torch or ode_g1.

    Loading config.py by filename avoids silently using another installed
    ``ode_g1`` version. The native validator owns YAML defaults, path expansion,
    field names and architecture constraints.
    """
    root = source_root(source_dir)
    pin_source(root, ODE_SOURCE_COMMIT, "ODE-POLICY real-g1")
    path = Path(config["native_config"]).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("native_config must be a native ODE G1 YAML file")
    spec = importlib.util.spec_from_file_location("_humanwam_pinned_ode_g1_config", root / "groot/src/ode_g1/config.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.load_config(path, _overrides(config))
    if cfg["base_model"]["allow_download"]:
        raise ValueError("HumanWAM GR00T ODE uses prepared local assets; set base_model.allow_download=false")
    if cfg["head"]["horizon"] != 16:
        raise ValueError("HumanWAM GR00T ODE requires its H16 training/serving contract")
    if cfg["dataset"]["camera_keys"] != NATIVE_CAMERAS:
        raise ValueError("HumanWAM GR00T ODE requires ego_view then chest camera order")
    if type(config.get("num_processes", 1)) is not int or config.get("num_processes", 1) != 1:
        raise ValueError("The published ODE G1 trainer supports one optimizer process, without DDP")
    return cfg


def gr00t_source_root(config, ode_root):
    """Require the separate published ODE runtime, never the standard N1.7 pin."""
    root = Path(config.get("gr00t_source_dir", ode_root.parent / "Isaac-GR00T-ODE")).expanduser().resolve()
    if not (root / "gr00t/model/gr00t_n1d7/gr00t_n1d7.py").is_file():
        raise FileNotFoundError(f"Pinned ODE Isaac-GR00T source missing: {root}")
    return pin_source(root, GR00T_SOURCE_COMMIT, "separate Isaac-GR00T ODE")


def _environment(root, gr00t_root=None):
    package_src = Path(__file__).resolve().parents[2]
    paths = [str(root / "groot/src"), str(package_src), os.environ.get("PYTHONPATH", "")]
    if gr00t_root is not None:
        paths.insert(0, str(gr00t_root))
    return {
        "PYTHONPATH": os.pathsep.join(paths),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
    }


def build_native_command(config, source_dir, action):
    """Build a published CPU/GPU CLI stage, excluding robot execution commands."""
    if action not in {"inspect", "prepare", "features", "train", "infer", "deploy-check"}:
        raise ValueError("Choose inspect, prepare, features, train, infer or deploy-check")
    root = source_root(source_dir)
    load_native_config(config, root)
    gr00t_root = gr00t_source_root(config, root) if action in {"features", "train", "infer"} else None
    path = Path(config["native_config"]).expanduser().resolve(strict=True)
    argv = [config.get("python", "python3"), "-m", "ode_g1", "deploy" if action == "deploy-check" else action, "--config", str(path)]
    for value in _overrides(config):
        argv += ["--set", value]
    if action == "deploy-check":
        argv.append("--check")
    if action == "infer":
        source = Path(config["input"]).expanduser().resolve(strict=True)
        output = Path(config["output"]).expanduser().resolve()
        if not source.is_file():
            raise ValueError("input must be an observation NPZ file")
        if output.exists():
            raise ValueError("Choose a new inference output path")
        argv += ["--input", str(source), "--output", str(output)]
        episode_start = config.get("episode_start", False)
        if type(episode_start) is not bool:
            raise ValueError("episode_start must be a boolean")
        if episode_start:
            argv.append("--episode-start")
        images = config.get("images", {})
        if not isinstance(images, dict):
            raise ValueError("images must map native camera keys to RGB image paths")
        for camera, image in images.items():
            if not isinstance(camera, str) or "=" in camera or not camera:
                raise ValueError("Invalid native input camera key")
            path = Path(image).expanduser().resolve(strict=True)
            if not path.is_file():
                raise ValueError("Input image must be a file")
            argv += ["--image", camera + "=" + str(path)]
        if config.get("instruction") is not None:
            instruction = config["instruction"]
            if not isinstance(instruction, str) or not instruction.strip():
                raise ValueError("instruction must be a nonempty string")
            argv += ["--instruction", instruction]
    return {"argv": argv, "cwd": str(root), "env": _environment(root, gr00t_root)}


def build_train_command(config, source_dir):
    return build_native_command(config, source_dir, "train")


def build_serve_command(config, source_dir):
    root = source_root(source_dir)
    load_native_config(config, root)
    gr00t_root = gr00t_source_root(config, root)
    launcher = Path(config["launcher_config"]).expanduser().resolve(strict=True)
    port = config.get("port", 5558)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in 1..65535")
    host = config.get("host", "127.0.0.1")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("host must be a nonempty address")
    return {
        "argv": [config.get("python", "python3"), "-m", "humanwam.sonic_policy_server", "--config", str(launcher), "--host", host, "--port", str(port)],
        "cwd": str(root), "env": _environment(root, gr00t_root),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["inspect", "prepare", "features", "train", "infer", "deploy-check"])
    parser.add_argument("--config", required=True, help="HumanWAM JSON wrapper, not the native ODE YAML")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--episode-start", action="store_true")
    parser.add_argument("--image", action="append", default=[], metavar="CAMERA=RGB_PATH")
    parser.add_argument("--instruction")
    args = parser.parse_args(argv)
    from humanwam.config import load_config, repository_root

    config = load_config(args.config)
    if config["backend"] != "groot_ode":
        raise ValueError("Use a groot_ode HumanWAM config")
    root = repository_root()
    source = Path(config.get("source_dir", root / "third_party/ODE-POLICY"))
    if not source.is_absolute():
        source = root / source
    options = {**config["options"], "python": config.get("python", "python3")}
    if args.action == "infer":
        if not args.input or not args.output:
            parser.error("infer requires --input and --output")
        images = {}
        for item in args.image:
            if "=" not in item:
                parser.error("--image must be CAMERA=RGB_PATH")
            key, value = item.split("=", 1)
            if key in images:
                parser.error("Duplicate --image camera")
            images[key] = value
        options.update(input=args.input, output=args.output, episode_start=args.episode_start, images=images, instruction=args.instruction)
    elif any([args.input, args.output, args.episode_start, args.image, args.instruction]):
        parser.error("Observation flags apply only to infer")
    command = build_native_command(options, source, args.action)
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return 0
    return subprocess.call(command["argv"], cwd=command["cwd"], env={**os.environ, **command["env"]})


if __name__ == "__main__":
    raise SystemExit(main())
