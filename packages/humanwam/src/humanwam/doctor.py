"""Check source pins and the local runtime without loading GPU models."""

import hashlib
import importlib.util
import json
import os
import subprocess
import tempfile
from pathlib import Path


def _git(source, *args):
    result = subprocess.run(["git", "-C", str(source), *args], text=True, capture_output=True)
    return result.stdout.strip() if result.returncode == 0 else None


def inspect_sources(root):
    lock = json.loads((root / "backends.lock.json").read_text())
    results = []
    for item in lock["sources"]:
        source = root / item["path"]
        actual = _git(source, "rev-parse", "HEAD") if (source / ".git").exists() else None
        row = dict(name=item["name"], path=str(source), expected=item["commit"], actual=actual)
        row["status"] = "pass" if actual == item["commit"] else "missing_or_wrong_revision"
        for patch in item.get("patches", []):
            path = root / patch["path"]
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != patch["sha256"]
            ):
                row["status"] = "patch_hash_mismatch"
            elif source.exists():
                check = subprocess.run(
                    ["git", "-C", str(source), "apply", "--reverse", "--check", str(path)],
                    capture_output=True,
                )
                if check.returncode:
                    row["status"] = "patch_not_applied"
        if row["status"] == "pass":
            # Compute the exact permitted worktree from HEAD plus locked patches
            # using a temporary index. The real submodule index is never changed.
            with tempfile.TemporaryDirectory(prefix="humanwam-index-") as folder:
                env = dict(os.environ, GIT_INDEX_FILE=str(Path(folder) / "index"))

                def git_index(*args):
                    return subprocess.check_output(
                        ["git", "-C", str(source), *args], env=env, text=True
                    ).strip()

                git_index("read-tree", "HEAD")
                for patch in item.get("patches", []):
                    git_index("apply", "--cached", "--whitespace=nowarn", str(root / patch["path"]))
                allowed = set(git_index("diff", "--cached", "--name-only", "HEAD").splitlines())
                changed = set((_git(source, "diff", "HEAD", "--name-only") or "").splitlines())
                unexpected = changed - allowed
                for path in allowed:
                    expected = git_index("rev-parse", ":" + path)
                    actual_blob = _git(source, "hash-object", "--", path)
                    if actual_blob != expected:
                        unexpected.add(path)
                untracked = (
                    _git(source, "ls-files", "--others", "--exclude-standard") or ""
                ).splitlines()
                row["unexpected_tracked_changes"] = sorted(unexpected)
                row["untracked_files"] = sorted(set(untracked) - allowed)
                row["patch_added_files"] = sorted(set(untracked) & allowed)
                if unexpected or row["untracked_files"]:
                    row["status"] = "unlocked_local_changes"
        results.append(row)
    return results


def inspect_environment():
    return {
        module: importlib.util.find_spec(module) is not None
        for module in ("numpy", "torch", "jax", "gr00t", "openpi", "imagewam")
    }
