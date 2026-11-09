"""Read-only common LeRobot metadata checks; never changes or renumbers recordings."""

import json
import math
from pathlib import Path


def audit_metadata(dataset_dir):
    root = Path(dataset_dir).expanduser().resolve(strict=True)
    info = json.loads((root / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if isinstance(info["fps"], bool):
        raise ValueError("FPS must be numeric, not boolean")
    fps = float(info["fps"])
    if not math.isfinite(fps) or fps <= 0 or not episodes:
        raise ValueError("Expected positive FPS and nonempty episodes")
    ids = [row["episode_index"] for row in episodes]
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Episode IDs must be nonnegative exact integers")
    if any(
        type(info[key]) is not int or info[key] < 1 for key in ("total_frames", "total_episodes")
    ):
        raise ValueError("Dataset totals must be positive exact integers")
    if len(set(ids)) != len(ids) or ids != sorted(ids):
        raise ValueError("Episode IDs must be unique and sorted")
    if any(type(row["length"]) is not int for row in episodes):
        raise ValueError("Episode lengths must be exact integers")
    lengths = [row["length"] for row in episodes]
    if (
        min(lengths) <= 0
        or sum(lengths) != info["total_frames"]
        or len(ids) != info["total_episodes"]
    ):
        raise ValueError("Episode lengths/counts differ from info.json")
    durations = [n / fps for n in lengths]
    return dict(
        dataset=str(root),
        episodes=len(ids),
        frames=sum(lengths),
        fps=fps,
        min_seconds=min(durations),
        max_seconds=max(durations),
        mean_seconds=sum(durations) / len(durations),
        per_episode=[
            dict(episode_id=i, frames=n, seconds=s) for i, n, s in zip(ids, lengths, durations)
        ],
        source_modified=False,
        raw_numeric_video_integrity_checked=False,
        note="Metadata audit only. Run the selected backend's field mapping and numeric/video checks before training.",
    )
