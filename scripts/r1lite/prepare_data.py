"""Build a two-camera metadata view without modifying downloaded data."""
import argparse
import json
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from imagewam.utils import misc
from imagewam.utils.config_resolvers import register_default_resolvers

CAMERAS = {"observation.images.head_rgb", "observation.images.right_wrist_rgb"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--task", default="r1lite_cups_a2ai2i_20ep")
    args = parser.parse_args()
    manifest = json.loads((args.source / "imagewam_training_meta/selected_views_download_manifest.json").read_text())
    args.view.mkdir(parents=True, exist_ok=True)
    meta = args.view / "meta"
    meta.mkdir(exist_ok=True)
    info = json.loads((args.source / "meta/info.json").read_text())
    discarded = {k for k, v in info["features"].items() if v["dtype"] == "video" and k not in CAMERAS}
    info["features"] = {k: v for k, v in info["features"].items() if k not in discarded}
    info["total_videos"] = info["total_episodes"] * len(CAMERAS)
    expected = json.dumps(info, indent=2) + "\n"
    info_path = meta / "info.json"
    if info_path.exists() and info_path.read_text() != expected:
        raise RuntimeError("Refusing to overwrite a different dataset metadata view")
    info_path.write_text(expected)
    for name in ("tasks.jsonl", "episodes.jsonl"):
        target = meta / name
        if not target.exists():
            target.symlink_to(args.source / "meta" / name)
    rows = []
    for line in (args.source / "meta/episodes_stats.jsonl").read_text().splitlines():
        row = json.loads(line)
        row["stats"] = {k: v for k, v in row["stats"].items() if k not in discarded}
        rows.append(json.dumps(row))
    (meta / "episodes_stats.jsonl").write_text("\n".join(rows) + "\n")
    for name in ("data", "videos"):
        target = args.view / name
        if not target.exists():
            target.symlink_to(args.source / name)
    training_meta = args.view / "imagewam_training_meta"
    training_meta.mkdir(exist_ok=True)
    (training_meta / "view_provenance.json").write_text(json.dumps({
        "source": str(args.source), "download_manifest": manifest,
        "selected_cameras": sorted(CAMERAS), "changes": "metadata view only; original parquet/video bytes unchanged",
    }, indent=2) + "\n")

    register_default_resolvers()
    with initialize_config_dir(config_dir=str(args.repo / "configs"), version_base="1.3"):
        cfg = compose(config_name="train", overrides=[
            f"task={args.task}",
            f"data.train.dataset_dirs=[{args.view}]",
            "data.train.qwen_text_cache_dir=null",
        ])
    misc.register_work_dir(training_meta)
    stats_path = training_meta / "dataset_stats.json"
    if stats_path.exists():
        cfg.data.train.pretrained_norm_stats = str(stats_path)
    dataset = instantiate(cfg.data.train)
    samples = []
    for idx in (0, 100, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[idx]
        shapes = {k: list(sample[k].shape) for k in ("video", "past_action", "action", "proprio")}
        assert shapes == {"video": [3, 2, 224, 448], "past_action": [16, 7], "action": [16, 7], "proprio": [16, 7]}, shapes
        assert all(torch.isfinite(sample[k]).all() for k in shapes)
        samples.append({"index": idx, "shapes": shapes, "prompt": sample["prompt"]})
    result = {"status": "VALIDATED", "dataset_len": len(dataset), "episodes": info["total_episodes"],
              "samples": samples, "stats_path": str(stats_path), "view": str(args.view)}
    (training_meta / "dataset_validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
