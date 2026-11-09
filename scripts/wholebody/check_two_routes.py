#!/usr/bin/env python3
"""Run motion-token and joint-reference CPU checks in isolated native processes.

Use the ImageWAM training environment. This never constructs a full FLUX model,
allocates a CUDA tensor, publishes robot commands, or starts training.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


CHECKS = (
    ("policy_runtime", "packages/humanwam/tests", "test_*.py"),
    ("source_and_rotation", "tests", "test_joint_reference_contract.py"),
    ("motion_dataset", "tests", "test_wholebody_dataset.py"),
    ("joint_dataset", "tests", "test_wholebody_joint_reference_dataset.py"),
    ("native_fm_ic", "tests", "test_wholebody_delta.py"),
    ("training_cli", "tests", "test_wholebody_training_cli.py"),
    ("joint_v2_cli", "tests", "test_wholebody_joint_v2_cli.py"),
    ("augmentation_config", "tests", "test_augmentation_config.py"),
    ("visual_augmentation", "tests", "test_wholebody_visual_augmentation.py"),
    ("training_augmentation", "tests", "test_wholebody_training_augmentation.py"),
    ("augmentation_cli", "tests", "test_wholebody_cli.py"),
)


def run_checks(output_dir):
    root = Path(__file__).resolve().parents[2]
    required = (
        "torch",
        "numpy",
        "av",
        "pyarrow",
        "hydra",
        "omegaconf",
        "accelerate",
        "pytest",
    )
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        raise RuntimeError(
            "Use the native ImageWAM environment; missing: " + ", ".join(missing)
        )
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        WANDB_MODE="disabled",
        TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="2",
        PYTHONPATH=os.pathsep.join(
            (str(root / "src"), str(root / "packages/humanwam/src"))
        ),
    )
    checks = []
    for name, directory, pattern in CHECKS:
        target = directory if pattern == "test_*.py" else str(Path(directory) / pattern)
        argv = [sys.executable, "-m", "pytest", "-q", target]
        started = time.monotonic()
        result = subprocess.run(
            argv, cwd=root, env=environment, text=True, capture_output=True
        )
        log = result.stdout + result.stderr
        (output / (name + ".log")).write_text(log)
        count = re.search(r"(\d+) passed", log)
        skipped = re.search(r"(\d+) skipped", log)
        checks.append(
            dict(
                name=name,
                returncode=result.returncode,
                elapsed_seconds=time.monotonic() - started,
                tests=int(count.group(1)) if count else None,
                skipped=int(skipped.group(1)) if skipped else 0,
                log=str(output / (name + ".log")),
                argv=argv,
            )
        )
        print(json.dumps(checks[-1]), flush=True)
    sources = {}
    for directory in ("src/imagewam", "packages/humanwam/src", "scripts/wholebody"):
        for path in sorted((root / directory).rglob("*.py")):
            sources[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    report = dict(
        status="CPU_CHECKS_PASSED"
        if all(c["returncode"] == 0 for c in checks)
        else "CPU_CHECKS_FAILED",
        python=sys.version,
        executable=sys.executable,
        checks=checks,
        source_sha256=sources,
        cuda_visible_devices="",
        full_flux_model_loaded=False,
        optimizer_updates=0,
        robot_actuation_performed=False,
        source_datasets_modified=False,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New report directory; existing paths are refused",
    )
    report = run_checks(parser.parse_args().output_dir)
    print(report["status"])
    raise SystemExit(0 if report["status"] == "CPU_CHECKS_PASSED" else 1)
