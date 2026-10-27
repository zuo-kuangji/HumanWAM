"""Read-only G1 whole-body adapter for native ImageWAM samples.

Fresh normalization is fit on every episode of the supplied collection.
Action76 uses motion64 and same-row measured Inspire hands. Current state44
is separate from the instruction-only text cache; no future state is returned.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F


REQUIRED_METADATA = ("meta/info.json", "meta/episodes.jsonl", "meta/tasks.jsonl", "meta/episodes_stats.jsonl")
DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)
CAMERAS = ("observation.images.ego_view", "observation.images.chest")
ACTION_KEY = "wholebody_action76"
STATE_KEY = "wholebody_state44"
STATS_SCHEMA = "wholebody-state44-action76-sameframe-all-episodes-v1"
NORMALIZATION_STRATEGY = "fit_wholebody_motion64_body29_gravity3_fixed_inspire_hands"
HAND_FIELD_ORDER = ("pinky", "ring", "middle", "index", "thumb_bend", "thumb_rotation")
NUMERIC_COLUMNS = (
    "observation.state", "observation.inspire_left_hand_state",
    "observation.inspire_right_hand_state", "observation.projected_gravity",
    "action.motion_token",
    "timestamp", "frame_index", "episode_index", "index", "task_index",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metadata_paths(root: Path) -> list[str]:
    """Hash all recorded JSON metadata, including optional modality.json."""
    for name in REQUIRED_METADATA:
        if not (root / name).is_file():
            raise FileNotFoundError(root / name)
    return sorted(str(p.relative_to(root)) for p in (root / "meta").iterdir()
                  if p.is_file() and p.suffix in {".json", ".jsonl"})


def load_metadata(dataset_dir: str | Path) -> dict:
    root = Path(dataset_dir).resolve()
    metadata_paths(root)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [json.loads(x) for x in (root / "meta/episodes.jsonl").read_text().splitlines() if x.strip()]
    tasks = [json.loads(x) for x in (root / "meta/tasks.jsonl").read_text().splitlines() if x.strip()]
    episode_stats = [json.loads(x) for x in (root / "meta/episodes_stats.jsonl").read_text().splitlines() if x.strip()]
    ids = [int(e["episode_index"]) for e in episodes]
    count, total_frames = int(info["total_episodes"]), int(info["total_frames"])
    if (info["codebase_version"] != "v2.1" or float(info["fps"]) != 30.0
            or count < 1 or total_frames < 1 or ids != list(range(count))
            or info.get("splits") != {"train": f"0:{count}"}):
        raise ValueError("Expected an all-training LeRobot v2.1 / 30 FPS collection with contiguous episode IDs")
    if int(info["chunks_size"]) != 1000:
        raise ValueError("Expected the audited 1000-episode chunk layout")
    if (len(tasks) != 1 or tasks[0]["task_index"] != 0
            or not isinstance(tasks[0].get("task"), str) or not tasks[0]["task"].strip()):
        raise ValueError("Expected one nonempty recorded instruction with task_index=0")
    if [int(s["episode_index"]) for s in episode_stats] != ids:
        raise ValueError("Episode statistics IDs differ from the curated episode list")
    features = info["features"]
    dimensions = {"observation.state": 43, "action.motion_token": 64,
                  "observation.inspire_left_hand_state": 6,
                  "observation.inspire_right_hand_state": 6,
                  "observation.projected_gravity": 3}
    for key, dim in dimensions.items():
        if features[key]["shape"] != [dim]:
            raise ValueError(f"Wrong raw feature dimension for {key}")
        if "inspire_" in key and "names" in features[key]:
            side = "left" if key == "observation.inspire_left_hand_state" else "right"
            names = tuple(features[key]["names"])
            side_prefixed = tuple(f"{side}_{name}" for name in HAND_FIELD_ORDER)
            if names not in (HAND_FIELD_ORDER, side_prefixed):
                raise ValueError(f"Wrong native Inspire hand field order for {key}")
    for camera in CAMERAS:
        shape = features[camera]["shape"]
        if (len(shape) != 3 or shape[-1] != 3
                or any(not isinstance(size, int) or isinstance(size, bool) or size < 1 for size in shape)
                or features[camera]["dtype"] != "video"):
            raise ValueError(f"Wrong RGB camera specification for {camera}")
    if info.get("total_videos") != count * len(CAMERAS):
        raise ValueError("Wrong recorded video count")
    if any(int(e["length"]) < 1 for e in episodes) or sum(int(e["length"]) for e in episodes) != total_frames:
        raise ValueError("Metadata frame count mismatch")
    for e, stats in zip(episodes, episode_stats):
        for key, stat in stats["stats"].items():
            if "count" in stat and np.any(np.asarray(stat["count"]) != int(e["length"])):
                raise ValueError(f"Wrong metadata statistics count for episode {e['episode_index']}: {key}")
    return {"root": root, "info": info, "episodes": episodes, "tasks": tasks,
            "train": ids, "validation": [], "split": {"train_episode_ids": ids, "validation_episode_ids": []}}


def map_numeric(columns: dict) -> tuple[np.ndarray, np.ndarray]:
    """State44 plus action76=(motion64,left measured6,right measured6).

    Both output parts use the same parquet row. Hand labels are measured
    observations, not actuator command labels and not teleop's binary7D targets.
    """
    n = len(columns["observation.state"])
    dimensions = {"observation.state": 43, "action.motion_token": 64,
                  "observation.inspire_left_hand_state": 6,
                  "observation.inspire_right_hand_state": 6,
                  "observation.projected_gravity": 3}
    arrays = {}
    for key, dim in dimensions.items():
        array = np.asarray(columns[key], dtype=np.float32)
        if array.shape != (n, dim) or not np.isfinite(array).all():
            raise ValueError(f"Invalid shape/non-finite numeric feature {key}: {array.shape}")
        if "inspire_" in key and np.any((array < 0.0) | (array > 1.0)):
            raise ValueError(f"Measured hand values outside physical [0,1]: {key}")
        arrays[key] = array
    body = arrays["observation.state"]
    left = arrays["observation.inspire_left_hand_state"]
    right = arrays["observation.inspire_right_hand_state"]
    state = np.concatenate((body[:, :22], body[:, 29:36], left, right,
                            arrays["observation.projected_gravity"]), axis=-1)
    action = np.concatenate((arrays["action.motion_token"], left, right), axis=-1)
    return state, action


def read_episode(root: Path, episode_id: int, expected_length: int,
                 expected_global_start: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    import pyarrow.parquet as pq
    path = root / f"data/chunk-{episode_id // 1000:03d}/episode_{episode_id:06d}.parquet"
    columns = pq.read_table(path, columns=list(NUMERIC_COLUMNS), use_threads=False).to_pydict()
    n = len(columns["frame_index"])
    if n != expected_length or not np.array_equal(columns["frame_index"], np.arange(n)):
        raise ValueError(f"Nonsequential/mismatched frame indices in episode {episode_id}")
    if not np.all(np.asarray(columns["episode_index"]) == episode_id):
        raise ValueError(f"Cross-episode numeric data in episode {episode_id}")
    if not np.all(np.asarray(columns["task_index"]) == 0):
        raise ValueError(f"Unexpected instruction ID in episode {episode_id}")
    if not np.allclose(columns["timestamp"], np.arange(n) / 30.0, rtol=0, atol=3e-6):
        raise ValueError(f"Timestamp/frame alignment mismatch in episode {episode_id}")
    if expected_global_start is not None and not np.array_equal(columns["index"], np.arange(expected_global_start, expected_global_start + n)):
        raise ValueError(f"Global index mismatch in episode {episode_id}")
    return map_numeric(columns)


def window_indices(anchor: int, length: int, horizon: int = 40, history: int = 40) -> dict:
    if not 0 <= anchor < length or min(horizon, history) < 1:
        raise ValueError("Invalid episode anchor/window")
    future = np.arange(anchor, anchor + horizon, dtype=np.int64)
    past = np.arange(anchor - history, anchor, dtype=np.int64)
    return {"action": np.minimum(future, length - 1), "action_is_pad": future >= length,
            "past_action": np.maximum(past, 0), "past_action_is_pad": past < 0,
            "image": np.asarray([anchor, min(anchor + horizon, length - 1)]),
            "image_is_pad": np.asarray([False, anchor + horizon >= length])}


def _field_stats(values: np.ndarray) -> dict:
    a = values.astype(np.float64)
    return {f"global_{name}": result.tolist() for name, result in {
        "min": a.min(axis=0), "max": a.max(axis=0), "mean": a.mean(axis=0),
        "std": a.std(axis=0), "q01": np.quantile(a, 0.01, axis=0),
        "q99": np.quantile(a, 0.99, axis=0)}.items()}


def all_training_episode_ids(meta: dict, train_split_mode: str) -> list[int]:
    if train_split_mode != "all_episodes":
        raise ValueError("This version trains all episodes with no held-out set")
    return [int(e["episode_index"]) for e in meta["episodes"]]


def fit_statistics(episode_arrays: dict[int, tuple[np.ndarray, np.ndarray]], train_ids: list[int]) -> dict:
    """Fit this collection's motion/body/gravity quantiles; hands stay physical."""
    if train_ids != sorted(episode_arrays) or len(set(train_ids)) != len(train_ids) or not train_ids:
        raise ValueError("Statistics must use all episodes exactly once, in sorted order")
    state = np.concatenate([episode_arrays[i][0] for i in train_ids])
    action = np.concatenate([episode_arrays[i][1] for i in train_ids])
    if (state.shape != (len(action), 44) or action.shape[1:] != (76,)
            or not np.isfinite(state).all() or not np.isfinite(action).all()):
        raise ValueError("Invalid state44/action76 arrays")
    if not np.array_equal(state[:, 29:41], action[:, 64:76]):
        raise ValueError("State and action measured hands must use the same source rows")
    if np.any((action[:, 64:76] < 0) | (action[:, 64:76] > 1)):
        raise ValueError("Measured hand feedback must already be in physical [0,1]")
    return {"schema": STATS_SCHEMA,
            "action": {ACTION_KEY: _field_stats(action)},
            "state": {STATE_KEY: _field_stats(state)},
            "normalization": {"mode": "q01/q99", "clamp": [-5.0, 5.0],
                              "state_fixed_bounds": {"start": 29, "end": 41, "min": 0.0, "max": 1.0},
                              "action_fixed_bounds": {"start": 64, "end": 76, "min": 0.0, "max": 1.0},
                              "constant_range_tolerance": 1e-4},
            "train_episode_ids": list(train_ids), "train_frames": int(len(action)),
            "train_split_mode": "all_episodes", "heldout_episode_ids": [],
            "normalization_reference_episode_ids": list(train_ids),
            "normalization_reference_train_frames": int(len(action)),
            "normalization_strategy": NORMALIZATION_STRATEGY,
            "action_order": "motion64,left_measured_hand6_native_order,right_measured_hand6_native_order",
            "state_order": "body29,left_measured_hand6_native_order,right_measured_hand6_native_order,projected_gravity3",
            "hand_target_time_offset": 0, "hand_target_type": "same-row measured feedback, not issued command",
            "hand_field_order": list(HAND_FIELD_ORDER),
            "hand_field_order_source": "meta/info.json features.observation.inspire_{left,right}_hand_state.names",
            "hand_dataset_units": "[0,1]; do not divide recorded feedback by 1000 again",
            "hand_deployment_raw1000_rule": "if the runtime reports raw Inspire units [0,1000], divide by 1000 exactly once before normalization",
            "action_dim": 76, "proprio_dim": 44,
            "body_state_indices": list(range(22)) + list(range(29, 36)),
            "constant_action_dims": np.flatnonzero(np.std(action.astype(np.float64), axis=0) < 1e-8).tolist()}


class WholeBodyNormalizer:
    """Matches native q01/q99 affine + [-5,5] clamp; constant dimensions stay safe."""
    def __init__(self, stats: dict):
        if stats.get("schema") != STATS_SCHEMA:
            raise ValueError("Unsupported whole-body normalization schema")
        if (stats.get("action_dim") != 76 or stats.get("proprio_dim") != 44
                or stats.get("hand_target_time_offset") != 0
                or stats.get("train_split_mode") != "all_episodes"
                or stats.get("normalization_strategy") != NORMALIZATION_STRATEGY):
            raise ValueError("Incompatible whole-body dimensions, hand offset, or normalization strategy")
        self.stats = stats
        self.transforms = {}
        for field, key, dimension in (("action", ACTION_KEY, 76), ("state", STATE_KEY, 44)):
            values = stats[field][key]
            lower = np.asarray(values["global_q01"], dtype=np.float32).copy()
            upper = np.asarray(values["global_q99"], dtype=np.float32).copy()
            if (lower.shape != (dimension,) or upper.shape != (dimension,)
                    or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                    or np.any(upper < lower)):
                raise ValueError(f"Incorrect normalization dimension for {field}")
            if field == "action":
                lower[64:76] = 0.0
                upper[64:76] = 1.0
            if field == "state":
                lower[29:41] = 0.0
                upper[29:41] = 1.0
            span = upper - lower
            constant = span < 1e-4
            span[constant] = 2.0
            scale = 2.0 / span
            offset = -1.0 - scale * lower
            offset[constant] = -lower[constant]
            if not np.isfinite(scale).all() or not np.isfinite(offset).all():
                raise ValueError("Non-finite normalization statistics")
            self.transforms[field] = (scale, offset)

    def forward(self, values: np.ndarray, field: str) -> np.ndarray:
        scale, offset = self.transforms[field]
        return np.clip(values * scale + offset, -5.0, 5.0).astype(np.float32)

    def backward(self, values: np.ndarray, field: str = "action") -> np.ndarray:
        scale, offset = self.transforms[field]
        return ((values - offset) / scale).astype(np.float32)


def prepare_statistics(dataset_dir: str, output_dir: str,
                       train_split_mode: str = "all_episodes") -> dict:
    """CPU preparation writes only outside the immutable dataset, never inherits."""
    meta = load_metadata(dataset_dir)
    root, output = meta["root"], Path(output_dir).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Statistics/cache outputs must be independent of the read-only dataset")
    if (output / "dataset_stats.json").exists() or (output / "dataset_audit.json").exists():
        raise FileExistsError("Refusing to overwrite existing statistics/audit")
    meta_files = metadata_paths(root)
    before_meta = {name: sha256(root / name) for name in meta_files}
    train_ids = all_training_episode_ids(meta, train_split_mode)
    arrays, data_hashes = {}, {}
    global_start = 0
    for episode in meta["episodes"]:
        ep, n = int(episode["episode_index"]), int(episode["length"])
        path = root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        digest = sha256(path)
        arrays[ep] = read_episode(root, ep, n, global_start)
        if sha256(path) != digest:
            raise ValueError(f"Parquet changed during statistics preparation: {path}")
        data_hashes[str(path.relative_to(root))] = digest
        global_start += n
        for camera in CAMERAS:
            video = root / f"videos/chunk-{ep // 1000:03d}/{camera}/episode_{ep:06d}.mp4"
            if not video.is_file() or video.stat().st_size == 0:
                raise FileNotFoundError(video)
    stats = fit_statistics(arrays, train_ids)
    stats.update(dataset_root=str(root), instruction=meta["tasks"][0]["task"],
                 metadata_sha256=before_meta, parquet_sha256=data_hashes)
    after_meta = {name: sha256(root / name) for name in metadata_paths(root)}
    after_data = {name: sha256(root / name) for name in data_hashes}
    if before_meta != after_meta or data_hashes != after_data:
        raise ValueError("Dataset changed during statistics preparation")
    audit = {"schema": "wholebody-state44-action76-all-training-audit-v1",
             "dataset_root": str(root), "raw_dataset_read_only": True,
             "source_unchanged_during_preparation": True, "episodes": len(train_ids),
             "total_frames": global_start, "fps": 30, "train_episodes": train_ids,
             "validation_episodes": [], "train_split_mode": "all_episodes",
             "normalization_strategy": NORMALIZATION_STRATEGY, "normalization_inherited_from_other_task": False,
             "train_frames": global_start, "validation_frames": 0,
             "metadata_sha256": before_meta, "parquet_sha256": data_hashes,
             "numeric_dimensions_time_and_finite_checks": "PASS",
             "instruction": meta["tasks"][0]["task"], "cameras": list(CAMERAS),
             "raw_video_resolutions": {camera: meta["info"]["features"][camera]["shape"][:2] for camera in CAMERAS},
             "output_video_resolution": [224, 448],
             "state_condition_time": "anchor only, repeated across proprio time axis",
             "future_state_in_model_input": False, "future_image_offset": 40,
             "future_action_offsets": [0, 39], "past_action_offsets": [-40, -1],
             "episode_boundary_policy": "repeat nearest same-episode value; anchor0 uses zero normalized history; padding masks retained",
             "motion_target_preserved": True, "hand_target_type": "same-row measured feedback",
             "hand_target_time_offset": 0, "hand_command_reconstruction_claimed": False,
             "per_channel_hardware_synchronization_verified": False,
             "action_source_noise_mask": [True] * 76,
             "constant_action_dims": stats["constant_action_dims"], "video_predecode": False}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "dataset_stats.json").open("x") as stream:
        json.dump(stats, stream, indent=2, allow_nan=False)
    audit["statistics_sha256"] = sha256(output / "dataset_stats.json")
    with (output / "dataset_audit.json").open("x") as stream:
        json.dump(audit, stream, indent=2, allow_nan=False)
    return audit


def resize_rgb_frame_uint8(array: np.ndarray) -> torch.Tensor:
    """Resize RGB while removing only possible interpolation-boundary roundoff."""
    if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError("Expected uint8 HWC RGB frame")
    rgb = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
    resized = F.interpolate(rgb.unsqueeze(0), size=(224, 224), mode="bilinear",
                            align_corners=False, antialias=True)[0]
    # Float32 antialias weight sums can put white pixels a few ULP above1.
    # Preserve the RGB [0,1] contract before converting to model-space [-1,1].
    return resized.clamp_(0.0, 1.0)


def decode_resized_frames(path: Path, indices: list[int], fps: int = 30) -> dict[int, torch.Tensor]:
    """Decode only the requested span; never cache or predecode whole episodes."""
    import av
    wanted = set(int(x) for x in indices)
    result = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "SLICE"
        stream.codec_context.thread_count = 1
        if stream.start_time not in (None, 0):
            raise ValueError(f"Expected zero-start synchronized video: {path}")
        first, last = min(wanted), max(wanted)
        container.seek(int((first / fps) / stream.time_base), backward=True, any_frame=False, stream=stream)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Video lacks frame timestamps: {path}")
            position = float(frame.pts * stream.time_base) * fps
            index = round(position)
            if abs(position - index) > 0.1:
                raise ValueError(f"Non-frame-aligned decoded timestamp: {path}")
            if index in wanted:
                result[index] = resize_rgb_frame_uint8(frame.to_ndarray(format="rgb24"))
            if index >= last:
                break
    if set(result) != wanted:
        raise RuntimeError(f"Missing exact video frames {wanted - set(result)} in {path}")
    return result


class WholeBodyLeRobotDataset(torch.utils.data.Dataset):
    """Native ImageWAM-ready tensors; numeric data small/in-memory, RGB on demand."""
    def __init__(self, dataset_dir: str, stats_path: str, qwen_text_cache_dir: str,
                 split: str = "train", action_horizon: int = 40, past_action_size: int = 40,
                 qwen_context_len: int = 128, qwen_text_dim: int = 7680,
                 video_cache_frames: int = 64, decoder: Callable | None = None,
                 train_split_mode: str = "all_episodes"):
        if action_horizon != 40 or past_action_size != 40:
            raise ValueError("This audited protocol uses action/history horizon 40")
        if split != "train":
            raise ValueError("This all-episode version has no validation dataset; split must be train")
        self.meta = load_metadata(dataset_dir)
        self.root = self.meta["root"]
        self.stats = json.loads(Path(stats_path).read_text())
        expected_ids = all_training_episode_ids(self.meta, train_split_mode)
        if self.stats["train_episode_ids"] != expected_ids or self.stats.get("heldout_episode_ids") != []:
            raise ValueError("Statistics must cover all recorded episodes with no held-out subset")
        if self.stats.get("train_frames") != sum(int(e["length"]) for e in self.meta["episodes"]):
            raise ValueError("All-episode statistics frame count does not match the collection")
        if (self.stats.get("normalization_reference_episode_ids") != expected_ids
                or self.stats.get("instruction") != self.meta["tasks"][0]["task"]
                or self.stats.get("normalization_reference_train_frames") != sum(int(e["length"]) for e in self.meta["episodes"])):
            raise ValueError("Statistics must be fit from this complete collection")
        if set(self.stats["metadata_sha256"]) != set(metadata_paths(self.root)):
            raise ValueError("Recorded metadata file set changed after preparation")
        for path, digest in self.stats["metadata_sha256"].items():
            if sha256(self.root / path) != digest:
                raise ValueError(f"Dataset metadata changed after preparation: {path}")
        self.normalizer = WholeBodyNormalizer(self.stats)
        self.episode_ids = expected_ids
        self.episodes = {}
        lengths = []
        by_id = {int(e["episode_index"]): e for e in self.meta["episodes"]}
        expected_data_files = {f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet" for ep in self.episode_ids}
        if set(self.stats.get("parquet_sha256", {})) != expected_data_files:
            raise ValueError("Statistics lack the exact all-episode parquet hash manifest")
        global_start = 0
        for ep in self.episode_ids:
            n = int(by_id[ep]["length"])
            relative = f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
            digest = self.stats["parquet_sha256"][relative]
            if sha256(self.root / relative) != digest:
                raise ValueError(f"Parquet changed after statistics preparation: {relative}")
            self.episodes[ep] = read_episode(self.root, ep, n, global_start)
            if sha256(self.root / relative) != digest:
                raise ValueError(f"Parquet changed during dataset loading: {relative}")
            global_start += n
            lengths.append(n)
        self.ends = np.cumsum(lengths)
        self.action_horizon, self.past_action_size = action_horizon, past_action_size
        self.prompt = DEFAULT_PROMPT.format(task=self.meta["tasks"][0]["task"])
        suffix = f"{hashlib.sha256(self.prompt.encode()).hexdigest()}.qwen3_flux2_len{qwen_context_len}.pt"
        cache_path = Path(qwen_text_cache_dir) / suffix
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        self.text_hidden_states = payload["text_hidden_states"]
        self.text_attention_mask = payload["text_attention_mask"].bool()
        if self.text_hidden_states.shape != (qwen_context_len, qwen_text_dim) or self.text_attention_mask.shape != (qwen_context_len,):
            raise ValueError(f"Invalid instruction-only text cache shape: {cache_path}")
        if not torch.isfinite(self.text_hidden_states).all() or not self.text_attention_mask.any():
            raise ValueError("Invalid/empty cached instruction embeddings")
        self.video_cache_frames = int(video_cache_frames)
        if self.video_cache_frames < 0:
            raise ValueError("video_cache_frames must be nonnegative")
        self.frame_cache: OrderedDict = OrderedDict()
        self.decoder = decoder or decode_resized_frames

    def __len__(self):
        return int(self.ends[-1])

    def _frames(self, episode: int, camera: str, indices: list[int]) -> torch.Tensor:
        keys = [(episode, camera, int(i)) for i in indices]
        # Snapshot every hit before touching the LRU. A preceding miss in this
        # same request must not evict a later hit before its value is collected.
        hits = {key: self.frame_cache[key] for key in keys
                if self.video_cache_frames and key in self.frame_cache}
        missing = list(dict.fromkeys(key[2] for key in keys if key not in hits))
        fresh = {}
        if missing:
            path = self.root / f"videos/chunk-{episode // 1000:03d}/{camera}/episode_{episode:06d}.mp4"
            fresh = self.decoder(path, missing)
        values = []
        for key in keys:
            value = hits[key] if key in hits else fresh[key[2]]
            if value.shape != (3, 224, 224) or not torch.isfinite(value).all():
                raise ValueError("Decoder returned invalid RGB image")
            values.append(value)

        # Commit cache changes only after all request values are held locally.
        # This also preserves the cache if decoding/validation fails partway.
        if self.video_cache_frames:
            for key, value in zip(keys, values):
                self.frame_cache[key] = value
                self.frame_cache.move_to_end(key)
            while len(self.frame_cache) > self.video_cache_frames:
                self.frame_cache.popitem(last=False)
        return torch.stack(values)

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ep_position = int(np.searchsorted(self.ends, index, side="right"))
        episode = self.episode_ids[ep_position]
        anchor = int(index - (self.ends[ep_position - 1] if ep_position else 0))
        states, actions = self.episodes[episode]
        window = window_indices(anchor, len(actions), self.action_horizon, self.past_action_size)
        video = torch.cat([self._frames(episode, cam, window["image"].tolist()) for cam in CAMERAS], dim=-1)
        video = (video * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous()
        action = self.normalizer.forward(actions[window["action"]], "action")
        past = self.normalizer.forward(actions[window["past_action"]], "action")
        if bool(window["past_action_is_pad"].all()):
            # At episode start no logged action is causally available. Using
            # actions[0] here would leak the first target through its source.
            past = np.zeros_like(past)
        current_state = self.normalizer.forward(states[anchor], "state")
        # Never return future-state values through the proprio time dimension.
        proprio = np.repeat(current_state[None], self.action_horizon, axis=0)
        return {"video": video, "action": torch.from_numpy(action),
                "past_action": torch.from_numpy(past), "proprio": torch.from_numpy(proprio),
                "action_is_pad": torch.from_numpy(window["action_is_pad"]),
                "past_action_is_pad": torch.from_numpy(window["past_action_is_pad"]),
                "image_is_pad": torch.from_numpy(window["image_is_pad"]),
                "proprio_is_pad": torch.zeros(self.action_horizon, dtype=torch.bool),
                "action_dim_is_pad": torch.zeros(76, dtype=torch.bool),
                "proprio_dim_is_pad": torch.zeros(44, dtype=torch.bool),
                "text_hidden_states": self.text_hidden_states,
                "text_attention_mask": self.text_attention_mask,
                "prompt": self.prompt, "instruction": self.prompt,
                "idx": int(index), "episode_index": int(episode), "frame_index": anchor,
                "task_index": 0, "dataset_index": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-split-mode", choices=["all_episodes"], default="all_episodes")
    options = parser.parse_args()
    audit = prepare_statistics(options.dataset_dir, options.output_dir,
                               train_split_mode=options.train_split_mode)
    print(json.dumps({key: audit[key] for key in ("episodes", "total_frames", "train_frames", "validation_frames", "statistics_sha256")}, indent=2))
