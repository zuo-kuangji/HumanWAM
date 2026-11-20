"""Prepare a separate LeRobot v2.1 copy for FastWAM feedback76/H16 training.

Parquet conversion requires pyarrow in the data-preparation environment. Source
parquets and media are never modified. Media are linked to avoid duplicating
large videos; keep the original dataset available while using the prepared copy.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np

PROFILE = "feedback76_h16"
RECEIPT = "humanwam_feedback76.json"
MAPPING = {
    "action": ["action.motion_token[0:64]", "observation.inspire_left_hand_state[0:6]", "observation.inspire_right_hand_state[0:6]"],
    "state": ["observation.state[0:22]", "observation.state[29:36]", "observation.inspire_left_hand_state[0:6]", "observation.inspire_right_hand_state[0:6]", "observation.projected_gravity[0:3]"],
    "camera_order": ["ego_view", "chest"],
    "hand_semantics": "measured_inspire6_per_hand",
    "sampling": "same_row_labels_no_resampling",
}
RAW_DIMS = {
    "observation.state": 43,
    "action.motion_token": 64,
    "observation.inspire_left_hand_state": 6,
    "observation.inspire_right_hand_state": 6,
    "observation.projected_gravity": 3,
}


def compact_feedback76(columns):
    """Map raw columns exactly as GR00T's feedback76 modality metadata does."""
    arrays = {}
    for key, width in RAW_DIMS.items():
        value = np.asarray(columns[key], dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != width or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite [N,{width}]")
        arrays[key] = value
    sizes = {len(value) for value in arrays.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("All feedback76 columns must have the same nonzero row count")
    left = arrays["observation.inspire_left_hand_state"]
    right = arrays["observation.inspire_right_hand_state"]
    if any(np.any((value < 0) | (value > 1)) for value in (left, right)):
        raise ValueError("Measured Inspire states must already be [0,1]; no implicit 1000 scaling")
    state = arrays["observation.state"]
    action76 = np.concatenate([arrays["action.motion_token"], left, right], axis=-1)
    state44 = np.concatenate([state[:, :22], state[:, 29:36], left, right, arrays["observation.projected_gravity"]], axis=-1)
    return action76, state44


def _info(path):
    info = json.loads((path / "meta/info.json").read_text())
    fps = info.get("fps")
    if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(fps) or fps <= 0:
        raise ValueError("Dataset fps must be a positive finite number")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("This converter supports LeRobot v2.1 episode parquets only")
    for camera in MAPPING["camera_order"]:
        entry = info.get("features", {}).get("observation.images." + camera, {})
        if entry.get("dtype") not in {"video", "image"}:
            raise ValueError(f"Missing RGB camera observation.images.{camera}")
    return info


def audit_compact_dataset(dataset_dir):
    root = Path(dataset_dir).expanduser().resolve(strict=True)
    info = _info(root)
    for key, width in (("action", 76), ("observation.state", 44)):
        entry = info.get("features", {}).get(key, {})
        if entry.get("shape") != [width] or entry.get("dtype") != "float32":
            raise ValueError(f"Prepared dataset {key} must be float32 [{width}]")
    receipt = json.loads((root / "meta" / RECEIPT).read_text())
    if receipt.get("version") != 1 or receipt.get("profile") != PROFILE or receipt.get("mapping") != MAPPING:
        raise ValueError("Prepared dataset is missing the exact feedback76 mapping receipt")
    if receipt.get("action_frequency_hz") != info["fps"]:
        raise ValueError("Prepared dataset sampling frequency differs from its receipt")
    return receipt


def _feature_stats(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def _replace_vector(table, key, values, pa):
    array = pa.array(values.tolist(), type=pa.list_(pa.float32(), values.shape[1]))
    index = table.schema.get_field_index(key)
    return table.append_column(key, array) if index < 0 else table.set_column(index, key, array)


def prepare_dataset(source_dir, output_dir):
    """Convert episode parquets into a fresh directory with a mapping receipt."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = Path(source_dir).expanduser().resolve(strict=True)
    output = Path(output_dir).expanduser().resolve()
    if output.exists() or source in output.parents or output in source.parents:
        raise ValueError("Choose a fresh output directory outside the source dataset")
    info = _info(source)
    for key, width in RAW_DIMS.items():
        if info.get("features", {}).get(key, {}).get("shape") != [width]:
            raise ValueError(f"Raw dataset metadata requires {key}[{width}]")
    stats_path = source / "meta/episodes_stats.jsonl"
    old_stats = {
        record["episode_index"]: record
        for line in stats_path.read_text().splitlines() if line.strip()
        for record in [json.loads(line)]
    }
    episodes = [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
    if not episodes or len({row["episode_index"] for row in episodes}) != len(episodes):
        raise ValueError("Expected nonempty unique LeRobot episode metadata")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".feedback76-", dir=output.parent))
    receipt = {
        "version": 1, "profile": PROFILE, "mapping": MAPPING,
        "source_dataset": str(source), "action_frequency_hz": info["fps"],
        "source_modified": False, "media_storage": "symlink_to_source",
        "action_dim": 76, "state_dim": 44, "horizon": 16,
    }
    try:
        shutil.copytree(source / "meta", staging / "meta")
        for name in ("videos", "images", "annotations"):
            if (source / name).exists():
                (staging / name).symlink_to(source / name, target_is_directory=True)
        new_stats = []
        total_rows = 0
        for episode in episodes:
            index = episode["episode_index"]
            relative = Path(info["data_path"].format(episode_chunk=index // info["chunks_size"], episode_index=index))
            if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "data":
                raise ValueError("LeRobot data_path must stay under data/")
            table = pq.read_table(source / relative)
            actions, states = compact_feedback76({key: table[key].to_pylist() for key in RAW_DIMS})
            if len(actions) != episode["length"]:
                raise ValueError("Parquet row count differs from episode metadata")
            total_rows += len(actions)
            table = _replace_vector(table, "action", actions, pa)
            table = _replace_vector(table, "observation.state", states, pa)
            # Preserve image feature metadata and correct the two changed vectors.
            metadata = dict(table.schema.metadata or {})
            if b"huggingface" in metadata:
                hf = json.loads(metadata[b"huggingface"])
                features = hf.get("info", {}).get("features")
                if not isinstance(features, dict):
                    raise ValueError("Invalid HuggingFace parquet feature metadata")
                for key, width in (("action", 76), ("observation.state", 44)):
                    features[key] = {"feature": {"dtype": "float32", "_type": "Value"}, "length": width, "_type": "Sequence"}
                metadata[b"huggingface"] = json.dumps(hf).encode()
                table = table.replace_schema_metadata(metadata)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, target)
            record = copy.deepcopy(old_stats[index])
            record["stats"]["action"] = _feature_stats(actions)
            record["stats"]["observation.state"] = _feature_stats(states)
            new_stats.append(record)
        if total_rows != info["total_frames"]:
            raise ValueError("Converted frame count differs from info.json")
        for key, width in (("action", 76), ("observation.state", 44)):
            info["features"][key] = {"dtype": "float32", "shape": [width], "names": None}
        (staging / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
        (staging / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(row) + "\n" for row in new_stats))
        # The native processor computes its own H16 statistics during training;
        # stale legacy aggregate/cache files must not claim the old vector layout.
        for stale in (staging / "meta").glob("*stats*.json"):
            stale.unlink()
        (staging / "meta" / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
        audit_compact_dataset(staging)
        if output.exists():
            raise ValueError("Output appeared during conversion; refusing to replace it")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare_dataset(args.source, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
