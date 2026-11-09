"""Native DiT4DiT command builders. Return argv/cwd; never run robot commands."""

from pathlib import Path


def _root(config):
    root = Path(config["upstream_root"]).expanduser().resolve()
    if not (root / "DiT4DiT/training/train.py").is_file():
        raise FileNotFoundError(f"DiT4DiT checkout missing: {root}")
    return root


def train_command(config: dict) -> dict:
    root = _root(config)
    native_config = Path(config["native_config"]).expanduser().resolve()
    if not native_config.is_file():
        raise FileNotFoundError(native_config)
    processes = int(config.get("num_processes", 8))
    if processes < 1:
        raise ValueError("num_processes must be positive")
    # train.py constructs DeepSpeedPlugin with a relative config path, so cwd
    # MUST be the upstream root, not the HumanWAM root.
    argv = [
        config.get("python", "python"),
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(root / "DiT4DiT/config/deepseeds/deepspeed_zero2.yaml"),
        "--num_processes",
        str(processes),
        str(root / "DiT4DiT/training/train.py"),
        "--config_yaml",
        str(native_config),
    ]
    overrides = config.get("overrides", {})
    if not isinstance(overrides, dict):
        raise TypeError("DiT4DiT overrides must be a mapping of native dot-keys to values")
    for name, value in overrides.items():
        if not name or any(char.isspace() for char in name) or name.startswith("-"):
            raise ValueError("Invalid native dot-key")
        if isinstance(value, (dict, list, tuple)):
            raise ValueError("Use a native YAML config for structured values")
        argv.extend(
            [
                f"--{name}",
                "null"
                if value is None
                else str(value).lower()
                if isinstance(value, bool)
                else str(value),
            ]
        )
    return {"argv": argv, "cwd": str(root), "env": {"PYTHONPATH": str(root)}}


def serve_command(config: dict) -> dict:
    root = _root(config)
    checkpoint = Path(config["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    port = int(config.get("port", 10093))
    if not 1 <= port <= 65535:
        raise ValueError("port must be in1..65535")
    # This is upstream's WebSocket protocol, distinct from HumanWAM's generic
    # service. Its response includes data.normalized_actions in native units.
    argv = [
        config.get("python", "python"),
        "-m",
        "deployment.model_server.server_policy",
        "--ckpt_path",
        str(checkpoint),
        "--port",
        str(port),
        "--idle_timeout",
        str(int(config.get("idle_timeout", 1800))),
    ]
    if config.get("use_bf16", False):
        argv.append("--use_bf16")
    return {"argv": argv, "cwd": str(root), "env": {"PYTHONPATH": str(root)}}


def build_train_command(config: dict, source_dir) -> dict:
    return train_command(dict(config, upstream_root=str(source_dir)))


def build_serve_command(config: dict, source_dir) -> dict:
    return serve_command(dict(config, upstream_root=str(source_dir)))
