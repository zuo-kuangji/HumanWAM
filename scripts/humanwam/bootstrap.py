#!/usr/bin/env python3
"""Fetch exact code pins and apply auditable patches; never installs model weights."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def run(*args, cwd=None):
    subprocess.run(args, cwd=cwd, check=True)


def bootstrap(only=None):
    lock = json.loads((ROOT / "backends.lock.json").read_text())
    selected = set(only or [row["name"] for row in lock["sources"]])
    if selected - {row["name"] for row in lock["sources"]}:
        raise ValueError("Unknown source selected")
    for row in lock["sources"]:
        if row["name"] not in selected:
            continue
        source = ROOT / row["path"]
        if not (source / ".git").exists():
            run("git", "submodule", "update", "--init", "--", row["path"], cwd=ROOT)
        actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
        if actual != row["commit"]:
            raise RuntimeError(f"{row['name']} revision differs; preserve local work and inspect before switching")
        for patch in row["patches"]:
            path = ROOT / patch["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]:
                raise RuntimeError("Patch checksum differs: " + str(path))
            applied = subprocess.run(["git", "apply", "--reverse", "--check", str(path)], cwd=source, capture_output=True)
            if applied.returncode == 0:
                print(row["name"], path.name, "already applied")
                continue
            run("git", "apply", "--check", str(path), cwd=source)
            run("git", "apply", str(path), cwd=source)
            print(row["name"], path.name, "applied")
        print(row["name"], actual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=["groot_n17", "sonic", "pi05", "dit4dit", "psi0", "fastwam"])
    bootstrap(parser.parse_args().only)
