"""CPU-safe policy registry and explicit native launch commands."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from .config import load_config, repository_root
from .registry import build_command, build_policy, describe_backends


def json_value(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parser():
    p = argparse.ArgumentParser(prog="humanwam")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List backend families without model imports")
    sub.add_parser("doctor", help="Check pinned sources, patches and local imports")
    for name in ("train", "serve"):
        action = sub.add_parser(name, help=f"Run native {name} entry in the configured environment")
        action.add_argument("--config", required=True)
        action.add_argument("--dry-run", action="store_true")
    predict = sub.add_parser(
        "predict", help="Offline prediction; never connects to robot actuators"
    )
    predict.add_argument("--config", required=True)
    predict.add_argument(
        "--observation", required=True, help="NPZ arrays plus optional JSON metadata"
    )
    predict.add_argument("--metadata", help="JSON with prompt/native non-array observation fields")
    predict.add_argument("--output", required=True)
    data = sub.add_parser("audit-data", help="Read-only LeRobot metadata/time audit")
    data.add_argument("dataset_dir")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "list":
            print(json.dumps(describe_backends(), indent=2))
            return 0
        root = repository_root()
        if args.command == "doctor":
            from .doctor import inspect_environment, inspect_sources

            sources = inspect_sources(root)
            print(
                json.dumps(
                    dict(
                        sources=sources, environment=inspect_environment(), gpu_model_loaded=False
                    ),
                    indent=2,
                )
            )
            return int(any(row["status"] != "pass" for row in sources))
        if args.command == "audit-data":
            from .data import audit_metadata

            print(json.dumps(audit_metadata(args.dataset_dir), indent=2))
            return 0
        config = load_config(args.config)
        if args.command in ("train", "serve"):
            command = build_command(config, args.command, root)
            if args.dry_run:
                print(json.dumps(command, indent=2))
                return 0
            return subprocess.call(
                command["argv"], cwd=command["cwd"], env={**os.environ, **command.get("env", {})}
            )
        output = Path(args.output)
        if output.exists():
            raise ValueError("Output already exists; choose a new file")
        from .observations import observation_from_npz

        metadata = json.loads(Path(args.metadata).read_text()) if args.metadata else None
        observation = observation_from_npz(args.observation, metadata)
        result = build_policy(config).predict(observation)
        with output.open("x") as handle:
            json.dump(result, handle, default=json_value, allow_nan=False)
        return 0
    except (ValueError, FileNotFoundError, ImportError, RuntimeError, KeyError) as exc:
        print(f"humanwam: {exc}", file=__import__("sys").stderr)
        return 2
