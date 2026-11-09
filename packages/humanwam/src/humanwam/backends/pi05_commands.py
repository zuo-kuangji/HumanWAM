"""Argument builders for the actual pinned OpenPI scripts; never use a shell."""

from pathlib import Path

from .pi05 import validate_config


def _source(config, source_dir=None):
    root = Path(source_dir or config["source_dir"]).expanduser().resolve()
    if not (root / "src/openpi/training/config.py").is_file():
        raise FileNotFoundError("source_dir is not the bootstrapped pinned OpenPI checkout")
    return root


def build_train_command(config: dict, source_dir=None, *, python: str | None = None) -> dict:
    c = validate_config(config, require_checkpoint=False)
    root = _source(c, source_dir)
    if not isinstance(c.get("exp_name"), str) or not c["exp_name"].strip():
        raise ValueError("exp_name is required for a new native OpenPI training run")
    argv = [
        python or c.get("python", "python"),
        "scripts/train.py",
        c["config_name"],
        "--exp-name",
        c["exp_name"],
    ]
    for key in ["batch_size", "num_train_steps", "save_interval", "checkpoint_base_dir"]:
        if key in c:
            value = c[key]
            if key != "checkpoint_base_dir" and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"{key} must be a positive integer")
            argv.extend(["--" + key.replace("_", "-"), str(value)])
    if c.get("resume"):
        argv.append("--resume")
    # Data repository/root, normalization and modalities are determined by the
    # named native config and LeRobot cache, never invented generic flags.
    return {"argv": argv, "cwd": str(root), "env": {}}


def build_serve_command(config: dict, source_dir=None, *, python: str | None = None) -> dict:
    c = validate_config(config)
    root = _source(c, source_dir)
    port = c.get("port", 8000)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("port must be in1..65535")
    argv = [
        python or c.get("python", "python"),
        "scripts/serve_policy.py",
        "--port",
        str(port),
        "policy:checkpoint",
        "--policy.config",
        c["config_name"],
        "--policy.dir",
        c["checkpoint_dir"],
        "--policy.num-steps",
        str(c["num_steps"]),
    ]
    if c.get("default_prompt"):
        argv[2:2] = ["--default-prompt", c["default_prompt"]]
    return {"argv": argv, "cwd": str(root), "env": {}}
