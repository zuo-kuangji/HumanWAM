"""Build the pi0.5 training view of the G1/H2 tasks: decoded 224px frame cache, state / action arrays, norm stats.

Usage (openpi env, CPU only, one node):  python hb_prepare.py b6 basket a3 a7 a18 [--workers 16] [--overwrite]
Output per task in $HB_PI05_DATA/<task>/:
  frames_<cam>.npy  uint8 [N, 224, 224, 3] for the layout's two cameras; openpi_client.image_tools.resize_with_pad
                    (= OpenPI's ResizeImages(224, 224) transform, so the transform is a no-op in training)
  lowdim.npz        state [N,S], action [N,A] (raw), ep_end [N] (exclusive end row of the row's episode), episode_index, frame_index
  meta.json         provenance, layout, fps/control_hz, prompt, content box; assets/<layout>_<task>/norm_stats.json; READY
  g1 (b6 basket a3 a7): state 44 = body29 by name | inspire L6 | inspire R6 | gravity3, action 76 = motion64 | L6 | R6
  h2 (a18)            : state 78 = observation.state75 | gravity3,             action 75 = body_target31 | hands44
Why not OpenPI's LeRobotDataset path: the pinned lerobot (0cf8648, CODEBASE_VERSION v2.1) can read these v2.1 folders from
a local root, but openpi.training.data_loader.create_torch_dataset passes neither `root` nor `episodes` (needed for the B6
first-100 subset) and would fall back to the HF hub on any miss; and random-access h264 decoding per sample is slow.
Norm stats follow scripts/compute_norm_stats.py: RunningStats over the pre-Normalize "state" [N,S] and the "actions"
chunks [N,16,A] (same chunk/clamp as training), raw values (no augmentation). For h2 the actions stats are 75-D; the 76th
(padding) dim is added after Normalize by OpenPI's PadStatesAndActions, so it is exactly 0 and never normalized.
"""
import argparse
import json
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import hb_g1_data as D

COMMON = ["observation.projected_gravity", "timestamp", "frame_index", "episode_index", "task_index"]
FEATURES = {  # feature -> required shape, per layout
    "g1": {"action.motion_token": 64, "observation.inspire_left_hand_state": 6, "observation.inspire_right_hand_state": 6,
           "observation.projected_gravity": 3},
    "h2": {"observation.state": 75, "observation.projected_gravity": 3, "action.body_target": 31, "action.hands": 44},
}


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def col2d(table, name):
    arr = table.column(name).combine_chunks()
    vals = np.asarray(arr.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
    return vals.reshape(len(arr), -1)


def feature_names(info, key):
    names = info["features"][key].get("names")
    if isinstance(names, dict):
        names = next(iter(names.values()))
    return list(names) if names else None


def state_names(info):
    return feature_names(info, "observation.state")


def content_box(h, w, size=D.IMG):
    """Same arithmetic as openpi_client.image_tools._resize_with_pad_pil -> (top, bottom, left, right)."""
    ratio = max(w / size, h / size)
    rh, rw = int(h / ratio), int(w / ratio)
    top, left = max(0, int((size - rh) / 2)), max(0, int((size - rw) / 2))
    return top, top + rh, left, left + rw


def read_episode(src, info, ep, layout):
    """One episode -> raw state/action rows for the layout (training and verify_cpu use this)."""
    rel = info["data_path"].format(episode_chunk=ep // int(info["chunks_size"]), episode_index=ep)
    hand_diff = None
    if layout == "g1":
        cols = ["observation.state", "action.motion_token", "observation.inspire_left_hand_state",
                "observation.inspire_right_hand_state", *COMMON]
        t = pq.read_table(src / rel, columns=cols)
        st = col2d(t, "observation.state")
        body_idx = [state_names(info).index(n) for n in D.G1_BODY]
        left, right = col2d(t, "observation.inspire_left_hand_state"), col2d(t, "observation.inspire_right_hand_state")
        grav, motion = col2d(t, "observation.projected_gravity"), col2d(t, "action.motion_token")
        state = np.concatenate([st[:, body_idx], left, right, grav], axis=1)
        action = np.concatenate([motion, left, right], axis=1)
        if st.shape[1] >= 41:
            hand_diff = float(np.abs(st[:, 29:41] - np.concatenate([left, right], 1)).max())
    elif layout == "h2":
        t = pq.read_table(src / rel, columns=["observation.state", "action.body_target", "action.hands", *COMMON])
        st = col2d(t, "observation.state")
        state = np.concatenate([st, col2d(t, "observation.projected_gravity")], axis=1)
        action = np.concatenate([col2d(t, "action.body_target"), col2d(t, "action.hands")], axis=1)
    else:
        raise ValueError(layout)
    n = t.num_rows
    L = D.LAYOUTS[layout]
    assert state.shape == (n, L.state_dim) and action.shape == (n, L.action_dim), (ep, state.shape, action.shape)
    fi = t.column("frame_index").to_numpy()
    assert (fi == np.arange(n)).all(), f"episode {ep}: frame_index not 0..n-1"
    assert (t.column("episode_index").to_numpy() == ep).all(), ep
    return dict(state=state, action=action, ts=t.column("timestamp").to_numpy().astype(np.float64), fi=fi,
                task_index=np.unique(t.column("task_index").to_numpy()), hand_diff=hand_diff, raw_state_dim=st.shape[1])


def decode_into(job):
    """Decode one episode video sequentially (PyAV), pick the frame nearest to each parquet timestamp, resize_with_pad."""
    import av
    from openpi_client import image_tools

    video, npy, start, ts, fps, hw = job
    frames, times = [], []
    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for f in container.decode(stream):
            arr = f.to_ndarray(format="rgb24")
            if arr.shape[:2] != tuple(hw):
                raise ValueError(f"{video}: frame {arr.shape} != {hw}")
            frames.append(image_tools.resize_with_pad(arr[None], D.IMG, D.IMG)[0])
            times.append(float(f.pts * f.time_base))
    times = np.asarray(times)
    j = np.clip(np.searchsorted(times, ts), 1, len(times) - 1)
    j = np.where(np.abs(times[j - 1] - ts) <= np.abs(times[j] - ts), j - 1, j)
    err = float(np.abs(times[j] - ts).max())
    if err > 0.5 / fps:
        raise ValueError(f"{video}: timestamp mismatch {err:.4f}s (decoded {len(times)} frames, need {len(ts)})")
    out = np.load(npy, mmap_mode="r+")
    out[start:start + len(ts)] = np.stack(frames)[j]
    out.flush()
    return len(times), len(ts), err


def guard_quantiles(stats, raw, key, log):
    """pi0.5 uses quantile norm (q01->-1, q99->+1, no clipping). If a dim's q01..q99 range is degenerate relative to its
    full range (e.g. a hand that moves in <1% of frames), outliers would map to huge values; use min/max for that dim."""
    flat = raw.reshape(-1, raw.shape[-1])
    lo, hi = flat.min(0), flat.max(0)
    q01, q99 = np.asarray(stats.q01, np.float64).copy(), np.asarray(stats.q99, np.float64).copy()
    bad = (q99 - q01) < 0.05 * (hi - lo)
    q01[bad], q99[bad] = lo[bad], hi[bad]
    log[key] = {"dims_minmax_fallback": np.flatnonzero(bad).tolist(),
                "constant_dims": np.flatnonzero((hi - lo) < 1e-9).tolist()}
    stats.q01, stats.q99 = q01, q99
    return stats


def check_features(task, info, layout):
    feats = info["features"]
    for key, dim in FEATURES[layout].items():
        assert key in feats and list(feats[key]["shape"]) == [dim], (task, key, feats.get(key, {}).get("shape"))
    notes = {}
    if layout == "g1":
        missing = [n for n in D.G1_BODY if n not in (state_names(info) or [])]
        assert not missing, (task, "observation.state names lack", missing)
    else:  # h2: action.body_target / action.hands are documented to carry the names of state[0:31] / state[31:75]
        sn = state_names(info)
        for key, sl in (("action.body_target", slice(0, 31)), ("action.hands", slice(31, 75))):
            an = feature_names(info, key)
            if sn and an and list(an) != list(sn[sl]):
                diff = [(a, b) for a, b in zip(an, sn[sl], strict=False) if a != b][:5]
                print(f"WARNING {task}: {key} names differ from observation.state[{sl.start}:{sl.stop}] e.g. {diff}")
                notes[key] = diff
    return notes


def prepare(task, *, out_root=None, max_episodes=None, check_expected=True, workers=16, overwrite=False):
    import openpi.shared.normalize as normalize  # lightweight (pydantic/numpy)

    spec = D.TASKS[task]
    src, layout = spec.root, spec.layout
    L = D.LAYOUTS[layout]
    out_root = Path(out_root or D.data_root())
    dst = out_root / task
    if (dst / "READY").exists() and not overwrite:
        print(f"{task}: {dst} already READY (use --overwrite)")
        return dst
    info = json.loads((src / "meta/info.json").read_text())
    assert str(info.get("codebase_version", "")).startswith("v2"), info.get("codebase_version")
    fps = float(info["fps"])
    name_notes = check_features(task, info, layout)
    feats = info["features"]
    hw = {}
    for cam in L.cameras:
        f = feats[f"observation.images.{cam}"]
        assert f["dtype"] == "video", (cam, f["dtype"])
        shp = list(f["shape"])
        hw[cam] = (shp[0], shp[1]) if shp[-1] == 3 else (shp[1], shp[2])
    episodes = jl(src / "meta/episodes.jsonl")
    n_eps = len(episodes) if spec.limit is None else spec.limit
    if max_episodes:
        n_eps = min(n_eps, max_episodes)
    episodes = episodes[:n_eps]
    assert [e["episode_index"] for e in episodes] == list(range(n_eps)), "episodes.jsonl not 0..n-1"
    if check_expected:
        assert n_eps == spec.episodes, (task, n_eps, spec.episodes)
    tasks = {r["task_index"]: r["task"] for r in jl(src / "meta/tasks.jsonl")}

    rows, task_ids, hand_diff, raw_dim = [], set(), [], None
    for e in episodes:
        r = read_episode(src, info, e["episode_index"], layout)
        assert len(r["state"]) == int(e["length"]), (e, len(r["state"]))
        rows.append(r)
        task_ids.update(int(x) for x in r["task_index"])
        hand_diff.append(r["hand_diff"])
        raw_dim = r["raw_state_dim"]
    assert len(task_ids) == 1, f"expected a single task, got {task_ids}"
    prompt = tasks[next(iter(task_ids))]
    lengths = np.array([len(r["state"]) for r in rows])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    n = int(lengths.sum())
    if check_expected and spec.frames is not None:
        assert n == spec.frames, (task, n, spec.frames)
    state = np.concatenate([r["state"] for r in rows]).astype(np.float32)
    action = np.concatenate([r["action"] for r in rows]).astype(np.float32)
    ep_end = np.repeat(starts + lengths, lengths)
    assert np.isfinite(state).all() and np.isfinite(action).all()

    tmp = out_root / f".{task}.tmp-{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    jobs = []
    for cam in L.cameras:
        npy = tmp / f"frames_{cam}.npy"
        np.lib.format.open_memmap(npy, mode="w+", dtype=np.uint8, shape=(n, D.IMG, D.IMG, 3)).flush()
        for e, s, r in zip(episodes, starts, rows, strict=True):
            ep = e["episode_index"]
            rel = info["video_path"].format(episode_chunk=ep // int(info["chunks_size"]),
                                            video_key=f"observation.images.{cam}", episode_index=ep)
            jobs.append((src / rel, npy, int(s), r["ts"], fps, hw[cam]))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        res = list(ex.map(decode_into, jobs, chunksize=1))
    decoded_minus_rows = sorted({a - b for a, b, _ in res})
    print(f"{task}: decoded {len(jobs)} videos in {time.time() - t0:.0f}s; decoded-rows diffs {decoded_minus_rows}; "
          f"max |dt| {max(x[2] for x in res):.4f}s")

    np.savez(tmp / "lowdim.npz", state=state, action=action, ep_end=ep_end,
             episode_index=np.repeat(np.arange(n_eps), lengths), frame_index=np.concatenate([r["fi"] for r in rows]))
    chunk_idx = np.minimum(np.arange(n)[:, None] + np.arange(D.HORIZON)[None], ep_end[:, None] - 1)
    chunks = action[chunk_idx]  # [N, 16, A] exactly as the training sampler builds it
    norm_log = {}
    stats = {}
    for key, arr in (("state", state), ("actions", chunks)):
        rs = normalize.RunningStats()
        rs.update(arr)
        stats[key] = guard_quantiles(rs.get_statistics(), arr, key, norm_log)
    normalize.save(tmp / "assets" / D.asset_id(task), stats)
    cam_hw = hw[L.cameras[0]]
    assert all(hw[c] == cam_hw for c in L.cameras), hw
    meta = dict(task=task, layout=layout, source=str(src), codebase_version=info.get("codebase_version"),
                fps=fps, control_hz=spec.control_hz,
                control_hz_note="assumed controller rate (hb_g1_data.TASKS); state delay = round(k*fps/control_hz) rows, k in {1,2}",
                episodes=n_eps, frames=n, episode_lengths=lengths.tolist(), prompt=prompt, raw_state_dim=raw_dim,
                max_abs_diff_state29_41_vs_inspire=max([h for h in hand_diff if h is not None], default=None),
                name_notes=name_notes, state_layout=L.state_desc, action_layout=L.action_desc + ", rows t..t+15",
                cameras={"base_0_rgb": f"observation.images.{L.cameras[0]}",
                         "left_wrist_0_rgb": f"observation.images.{L.cameras[1]}",
                         "right_wrist_0_rgb": "zeros, image_mask False"},
                raw_hw=list(cam_hw), content_box=list(content_box(*cam_hw)), norm_guard=norm_log,
                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
    (tmp / "READY").write_text("ok\n")
    if dst.exists():
        shutil.rmtree(dst)
    tmp.rename(dst)
    print(f"PREPARE_OK {task} ({layout}): {n_eps} episodes, {n} frames, fps {fps}, prompt={prompt!r}, norm guard={norm_log}")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+", choices=sorted(D.TASKS))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    for t in a.tasks:
        prepare(t, workers=a.workers, overwrite=a.overwrite)


if __name__ == "__main__":
    sys.exit(main())
