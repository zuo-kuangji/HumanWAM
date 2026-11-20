"""Native Psi-Zero process commands; no shell, robot launch or weight conversion."""

import json
from pathlib import Path

from .psi0 import NECKLESS_PROFILE, validate_config


def _neckless_training_flags(config, flags):
    """Bind the native model, data and statistics dimensions without neck fields."""
    action_keys = config.get("action_keys", ["action.body_token_v1_1", "action[:14]"])
    state_keys = config.get("state_keys", ["observation.state"])
    for values in (action_keys, state_keys):
        if (not isinstance(values, list) or not values
                or any(not isinstance(value, str) or not value or "neck" in value.lower()
                       or value.startswith("-") or "\0" in value for value in values)):
            raise ValueError("The neckless profile requires explicit non-neck action_keys/state_keys")
    horizon = config.get("horizon", 30)
    execution = config.get("action_exec_horizon", horizon)
    if type(execution) is not int or not 1 <= execution <= horizon:
        raise ValueError("Neckless training action_exec_horizon must be in1..horizon")
    required = {
        "--model.action-dim": ["78"],
        "--model.odim": ["43"],
        "--model.action-chunk-size": [str(horizon)],
        "--model.action-exec-horizon": [str(execution)],
        "--data.transform.repack.action-chunk-size": [str(horizon)],
        "--data.transform.repack.pad-action-dim": ["78"],
        "--data.transform.field.pad-action-dim": ["78"],
        "--data.transform.repack.pad-state-dim": ["43"],
        "--data.transform.field.pad-state-dim": ["43"],
        "--data.transform.repack.action-keys": action_keys,
        "--data.transform.field.stat-action-keys": action_keys,
        "--data.transform.repack.state-keys": state_keys,
        "--data.transform.field.stat-state-keys": state_keys,
    }
    seen = set()
    index = 0
    while index < len(flags):
        token = flags[index]
        if not token.startswith("--"):
            raise ValueError("Native training arguments must begin with an option")
        name, equals, inline = token.partition("=")
        name = name.replace("_", "-")
        values = [inline] if equals else []
        index += 1
        while index < len(flags) and not flags[index].startswith("--"):
            values.append(flags[index])
            index += 1
        if name in required:
            if name in seen:
                raise ValueError("Duplicate neckless training option: " + name)
            seen.add(name)
            if values != required[name]:
                raise ValueError("Native flags conflict with the neckless profile: " + name)
    result = list(flags)
    for name, values in required.items():
        if name not in seen:
            result.extend([name, *values])
    return result


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
    config = validate_config(config, checkpoint=False)
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
    if config["embodiment_profile"] == NECKLESS_PROFILE:
        flags = _neckless_training_flags(config, flags)
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
