"""Native Psi-Zero process commands; no shell, robot launch or weight conversion."""

import json
from pathlib import Path

from .psi0 import validate_config


def _source(source_dir):
    root = Path(source_dir).expanduser().resolve()
    for name in (
        "src/psi/models/psi0.py",
        "src/psi/deploy/serve_psi0_sonic.py",
        "scripts/train.py",
    ):
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return root


def build_train_command(config: dict, source_dir) -> dict:
    root = _source(source_dir)
    arguments = Path(config["native_args_file"]).expanduser().resolve(strict=True)
    flags = json.loads(arguments.read_text())
    if (
        not isinstance(flags, list)
        or not flags
        or any(not isinstance(v, str) or "\0" in v for v in flags)
    ):
        raise ValueError(
            "native_args_file must be a nonempty JSON string list of exact native training flags/values"
        )
    if any(v.replace("_", "-").startswith("--auto-tag-run") for v in flags):
        raise ValueError("Do not auto-commit/tag code from a unified training launcher")
    module = config.get("native_config", "finetune_sonic_psi0_config")
    if not isinstance(module, str) or not module.replace("_", "").isalnum():
        raise ValueError("native_config must name a native training config module")
    if not (root / "src/psi/config/train" / (module + ".py")).is_file():
        raise FileNotFoundError("Native Psi-Zero training config is missing")
    count = config.get("num_processes", 8)
    if type(count) is not int or count < 1:
        raise ValueError("num_processes must be a positive integer")
    # The native training entry requires its .env file. Do not create/copy one
    # containing credentials; callers prepare it in their isolated native env.
    if not (root / ".env").is_file():
        raise FileNotFoundError(
            "Prepare the native Psi-Zero .env locally before training (see .env.sample)"
        )
    argv = [
        config.get("python", "python3"),
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node",
        str(count),
        str(root / "scripts/train.py"),
        module,
        *flags,
    ]
    return dict(argv=argv, cwd=str(root), env={"PYTHONPATH": str(root / "src")})


def build_serve_command(config: dict, source_dir) -> dict:
    root = _source(source_dir)
    c = validate_config(config)
    port = config.get("port", 8014)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("port must be an integer in1..65535")
    argv = [
        config.get("python", "python3"),
        "-m",
        "psi.deploy.serve_psi0_sonic",
        "--policy",
        "psi",
        "--rtc",
        "--run-dir",
        c["run_dir"],
        "--ckpt-step",
        str(c["ckpt_step"]),
        "--device",
        config.get("device", "cuda:0"),
        "--host",
        config.get("host", "127.0.0.1"),
        "--port",
        str(port),
    ]
    if "action_exec_horizon" in c:
        horizon = c["action_exec_horizon"]
        if type(horizon) is not int or not 1 <= horizon <= c["horizon"]:
            raise ValueError(
                "action_exec_horizon must be positive and no larger than the trained chunk"
            )
        argv += ["--action-exec-horizon", str(horizon)]
    return dict(argv=argv, cwd=str(root), env={"PYTHONPATH": str(root / "src")})
