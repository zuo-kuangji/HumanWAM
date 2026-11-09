"""Small JSON/YAML configs; paths expand only explicit environment variables."""

import json
import os
import re
from pathlib import Path


def repository_root():
    override = os.environ.get("HUMANWAM_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
        if not (root / "configs").is_dir():
            raise ValueError("HUMANWAM_ROOT must point to a HumanWAM checkout")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / "configs").is_dir() and (parent / "src/imagewam").is_dir():
            return parent
    raise ValueError("Set HUMANWAM_ROOT when using a wheel outside a checkout")


def _expand(value):
    if isinstance(value, str):
        result = os.path.expanduser(os.path.expandvars(value))
        missing = re.findall(r"\$\{?[A-Za-z_][A-Za-z0-9_]*\}?", result)
        if missing:
            raise ValueError(f"Unset environment variable in configuration: {', '.join(missing)}")
        return result
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    return value


def load_config(path):
    from .registry import backend

    path = Path(path).expanduser().resolve(strict=True)
    if path.suffix == ".json":
        value = json.loads(path.read_text())
    else:
        import yaml

        value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("A HumanWAM config must be a mapping with version: 1")
    backend(value.get("backend"))
    if not isinstance(value.get("options", {}), dict):
        raise ValueError("options must be a mapping")
    root = repository_root()
    os.environ.setdefault("HUMANWAM_ROOT", str(root))
    value = _expand(value)
    if "source_dir" in value:
        source = Path(value["source_dir"])
        value["source_dir"] = str((root / source if not source.is_absolute() else source).resolve())
    value["_config_path"] = str(path)
    return value
