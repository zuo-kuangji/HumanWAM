"""Metadata-only views of already downloaded native model assets."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

_COSMOS_ID = "nvidia/Cosmos-Reason2-2B"
_RECEIPT_NAME = "humanwam_asset_relocation.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_gr00t_offline_view(checkpoint, backbone, output):
    """Create a fresh symlink view and relocate exactly two asset locators.

    This performs no downloads, model imports, parameter edits or cache writes.
    Existing checkpoint files, including weights/statistics, remain symlink
    targets. Both metadata fields must name the expected native backbone.
    """
    checkpoint = Path(checkpoint).expanduser().resolve(strict=True)
    backbone = Path(backbone).expanduser().resolve(strict=True)
    requested_output = Path(output).expanduser()
    if os.path.lexists(requested_output):
        raise FileExistsError(f"Choose a fresh output directory: {requested_output}")
    output = requested_output.resolve()
    if not checkpoint.is_dir() or not backbone.is_dir():
        raise ValueError("checkpoint and backbone must be existing local directories")
    if output.is_relative_to(checkpoint) or output.is_relative_to(backbone):
        raise ValueError("The output view must remain outside the checkpoint and backbone")
    if (checkpoint / _RECEIPT_NAME).exists():
        raise ValueError("Use an original checkpoint, not another HumanWAM asset view")
    processor = Path("processor_config.json")
    if not (checkpoint / processor).is_file():
        processor = Path("processor/processor_config.json")
    definitions = (
        (Path("config.json"), ("model_name",)),
        (processor, ("processor_kwargs", "model_name")),
    )
    replacements = {}
    local_alias = output / "assets/nvidia/Cosmos-Reason2-2B"
    for relative, field_path in definitions:
        source = checkpoint / relative
        raw = json.loads(source.read_text())
        parent = raw
        try:
            for key in field_path[:-1]:
                parent = parent[key]
            previous = parent[field_path[-1]]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"Missing native asset locator {relative}:{'.'.join(field_path)}"
            ) from exc
        if previous != _COSMOS_ID:
            raise ValueError(f"Expected {relative}:{'.'.join(field_path)} to name {_COSMOS_ID!r}")
        parent[field_path[-1]] = str(local_alias)
        replacements[relative] = (raw, field_path, _sha(source))

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.stage-", dir=output.parent))
    linked = []
    changed = {}
    try:

        def populate(source, target, relative=Path()):
            for path in source.iterdir():
                name = relative / path.name
                destination = target / path.name
                if name in replacements:
                    continue
                if path.is_dir():
                    destination.mkdir()
                    populate(path, destination, name)
                else:
                    destination.symlink_to(path.resolve(strict=True))
                    linked.append(str(name))

        populate(checkpoint, stage)
        alias = stage / "assets/nvidia/Cosmos-Reason2-2B"
        if os.path.lexists(alias):
            raise ValueError("The checkpoint already contains the reserved Cosmos asset alias")
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(backbone, target_is_directory=True)
        for relative, (raw, field_path, original_sha) in replacements.items():
            path = stage / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x") as stream:
                json.dump(raw, stream, indent=2)
                stream.write("\n")
            changed[str(relative)] = {
                "field_path": ".".join(field_path),
                "old_model_name": _COSMOS_ID,
                "new_model_name": str(local_alias),
                "original_sha256": original_sha,
                "relocated_sha256": _sha(path),
            }
        for relative, (_, _, original_sha) in replacements.items():
            if _sha(checkpoint / relative) != original_sha:
                raise RuntimeError("Source metadata changed while preparing the view")
        receipt = {
            "original_checkpoint": str(checkpoint),
            "view": str(output),
            "backbone": str(backbone),
            "local_alias": str(local_alias),
            "metadata_only_asset_locator_overrides": changed,
            "linked_original_files": sorted(linked),
            "original_cache_modified": False,
            "weights_symlink_original_unchanged": True,
            "model_weights_loaded": False,
        }
        with (stage / _RECEIPT_NAME).open("x") as stream:
            json.dump(receipt, stream, indent=2)
            stream.write("\n")
        if os.path.lexists(output):
            raise FileExistsError(f"Output appeared while preparing the view: {output}")
        stage.rename(output)
    except BaseException:
        shutil.rmtree(stage)
        raise
    return receipt
