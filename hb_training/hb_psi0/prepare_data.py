"""Build Psi0 dataset views of the LeRobot v2.1 recordings (raw data is never modified).

Usage: <psi0 python> prepare_data.py [--cams <cam0>,<cam1>] [--force] <task> [<task> ...]   (b6 basket a3 a7 a18)

View  = $G/psi0-data/<task>/ (LeRobot v2.1 layout read by psi/data/lerobot/lerobot_ext.py):
  data/chunk-*/episode_*.parquet  NEW small parquet: timestamp, frame_index, episode_index, index, task_index,
                                  <state_key> (float32[S]), <action_key> (float32[A])
  videos/...                      symlinks to the raw mp4s of the selected cameras only
                                  (LeRobot decodes every video key listed in info.json)
  meta/info.json                  rewritten features / totals
  meta/episodes.jsonl, tasks.jsonl  copied (B6: first 100 episodes only)
  meta/episodes_stats.jsonl       per-episode stats of the two new columns (lerobot aggregate_stats input)
  meta/stats_psi0.json            min/max/mean/std/q01/q99 used by SonicActionStateTransform (bounds)
  hb-view-receipt.json            robot layout (keys, dims, body_dims, cameras) read by hb_args.py + diagnostics

Per-robot layouts (same row t; labels are the values of row t):
  g1 (b6 basket a3 a7)  observation.hb44 = observation.state[0:29] (29 body joints, checked by name)
                                           + inspire_left_hand_state 6 + inspire_right_hand_state 6 + projected_gravity 3
                        action.hb76      = action.motion_token 64 + inspire_left 6 + inspire_right 6 (measured hands)
                        cameras ego_view (cam0) + chest; joint-perturbed dims 29
  h2 (a18)              observation.hb78 = observation.state[0:75] (legs 12, waist 3, arms 14, head 2, hands 22+22)
                                           + projected_gravity 3
                        action.hb75      = action.body_target 31 (names == state[0:31]) + action.hands 44 (names == state[31:75])
                        cameras head (cam0) + chest; joint-perturbed dims 31 (29 body + head 2)
Why derived columns: SonicRepackTransform.delta_timestamps keys windows by column name, so one column cannot be
both a 1-frame state and a 30-frame action chunk (G1 hands), and Psi0 takes one state / one action key here.
The CLIP pooled text cache is NOT precomputed here: FinetuneTrainer._warm_pooled_cache_from_tasks
builds it at start from meta/tasks.jsonl and saves <run>/clip_pooled_cache.pt (shipped with the run).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

H = Path(__file__).resolve().parent
G = Path(os.environ.get("G", H.parent))
P = G.parent
OUT = Path(os.environ.get("HB_DATA_ROOT", G / "psi0-data"))
TASKS = {  # task -> (source dataset, episode limit, robot layout)
    "b6": (P / "b6-v5t7-ready-start-60ep-20261109/data/deliveries/g1/b6-v5t7-ready-start-20261109", 100, "g1"),
    "basket": (P / "basket-current-dynamics-100-v21-b0ff522/data", None, "g1"),  # HF basket-transfer-expert-v1 @current-dynamics-100-v21 b0ff522 (old ed77719 retired)
    "trolley": (P / "trolley-push-expert-v1-20261110/data/trolley_push", 100, "g1"),  # HF trolley-push-expert-v1 @ef9024a: fixed first 100 of 103
    "b7": (P / "b7-black-clothes-hamper-v1-1bb002a/data", 100, "g1"),  # HF b7-black-clothes-hamper-expert-v1 @1bb002a, first 100 of 107
    "a6": (P / "a6-door-pull-v1-cf3ee9b/data/door_pull", None, "g1"),  # HF door-pull-expert-v1 @cf3ee9b (A6)
    "c6t": (P / "c6-test8-06c231d/data", None, "g1"),  # c6 8-episode adaptation test only
    "c6": (P / "c6-bathroom-door-fetch-v1-3afba69/data", None, "g1"),  # HF c6-bathroom-door-fetch-expert-v1 @3afba69 (C6)
    "a3": (P / "a3-a7-20261109/a3-fridge-open-expert-v1/780fcf1e4bd7db947a8c2d8f7b92e61f24f71b45", None, "g1"),
    "a7": (P / "a3-a7-20261109/a7-cabinet-close-expert-v1/1935a98e8f55d4d1d2d1615ddb97c2679713595c", None, "g1"),
    "a18": (P / "a18-h2-dataset-audit-8e956a10/dataset", None, "h2"),
    "a17": (P / "a17-unscrew-bottle-v1-407e6e2/data", None, "h2"),  # HF a17-unscrew-bottle-expert-v1 @407e6e2
}
BODY = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint",
        "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
STATE, TOKEN, LH, RH, GRAV = ("observation.state", "action.motion_token", "observation.inspire_left_hand_state",
                              "observation.inspire_right_hand_state", "observation.projected_gravity")
BT, HANDS = "action.body_target", "action.hands"
# parts = (column, start, end); a part's width is end - start
LAYOUTS = {
    "g1": dict(state_key="observation.hb44", action_key="action.hb76", body_dims=29,
               state_parts=[(STATE, 0, 29), (LH, 0, 6), (RH, 0, 6), (GRAV, 0, 3)],
               action_parts=[(TOKEN, 0, 64), (LH, 0, 6), (RH, 0, 6)],
               exact_dims={TOKEN: 64, LH: 6, RH: 6, GRAV: 3},
               cams={"ego": "observation.images.ego_view", "chest": "observation.images.chest"}, default_cams="ego,chest"),
    "h2": dict(state_key="observation.hb78", action_key="action.hb75", body_dims=31,
               state_parts=[(STATE, 0, 75), (GRAV, 0, 3)],
               action_parts=[(BT, 0, 31), (HANDS, 0, 44)],
               exact_dims={STATE: 75, GRAV: 3, BT: 31, HANDS: 44},
               cams={"head": "observation.images.head", "chest": "observation.images.chest",
                     "wrist": "observation.images.wrist", "third_person": "observation.images.third_person"},
               default_cams="head,chest"),
}
SCALARS = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
TS_TOL = 1e-4  # LeRobotDataset tolerance_s default; check_timestamps_sync raises beyond it


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def wl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def names_of(ft, dim, prefix):
    n = ft.get("names")
    if isinstance(n, dict):
        n = next(iter(n.values()), None)
    return list(n) if isinstance(n, list) and len(n) == dim else [f"{prefix}_{i}" for i in range(dim)]


def raw_names(ft):
    n = ft.get("names")
    if isinstance(n, dict):
        n = next(iter(n.values()), None)
    return list(n) if isinstance(n, list) else None


def col2d(t, key, dim):
    c = t.column(key).combine_chunks()
    v = np.asarray(c.flatten().to_numpy(zero_copy_only=False), dtype=np.float64)
    if v.size != len(t) * dim:
        raise ValueError(f"{key}: {v.size} values for {len(t)} rows x {dim}")
    return v.reshape(len(t), dim)


def fsl(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)), x.shape[1])


def ep_stats(x):
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(),
            "std": x.std(0).tolist(), "count": [int(len(x))]}


def full_stats(x, names):
    return {"min": x.min(0).tolist(), "max": x.max(0).tolist(), "mean": x.mean(0).tolist(), "std": x.std(0).tolist(),
            "q01": np.quantile(x, 0.01, axis=0).tolist(), "q99": np.quantile(x, 0.99, axis=0).tolist(),
            "count": [int(len(x))], "names": names}


def video_frames(path):
    try:
        import av
        with av.open(str(path)) as c:
            s = c.streams.video[0]
            return int(s.frames), float(s.average_rate or 0)
    except Exception as e:  # noqa: BLE001 - diagnostic only
        return None, str(e)[:200]


def part_names(feats, parts):
    out = []
    for key, a, b in parts:
        nm = names_of(feats[key], feats[key]["shape"][0], key.split(".")[-1])
        out += nm[a:b] if key == STATE else [f"{key.split('.')[-1]}.{x}" for x in nm[a:b]]
    return out


def check_layout(task, robot, L, feats, snames):
    for key, dim in L["exact_dims"].items():
        assert key in feats and list(feats[key]["shape"]) == [dim], (task, key, feats.get(key))
    for key, a, b in L["state_parts"] + L["action_parts"]:
        assert key in feats and feats[key]["shape"][0] >= b, (task, key, feats.get(key))
    if robot == "g1":
        body_idx = [snames.index(n) for n in BODY]
        assert body_idx == list(range(29)), f"{task}: body joints are not observation.state[0:29]: {body_idx}"
    else:  # h2: action names must mirror the state names (GR00T a18 audit)
        for key, a, b in ((BT, 0, 31), (HANDS, 31, 75)):
            n = raw_names(feats[key])
            if n is None:
                print(f"WARN {task}: {key} has no names; cannot check it mirrors observation.state[{a}:{b}]")
            else:
                assert n == snames[a:b], f"{task}: {key} names != observation.state[{a}:{b}] names"
        print(f"INFO {task}: perturbed joints (state[0:31]) = {snames[:31]}")


def build(task, cams, force, keep_ts):
    src, limit, robot = TASKS[task]
    L = LAYOUTS[robot]
    SK, AK, NB = L["state_key"], L["action_key"], L["body_dims"]
    info = json.loads((src / "meta/info.json").read_text())
    feats = info["features"]
    if info.get("codebase_version") != "v2.1":
        print(f"WARN {task}: codebase_version={info.get('codebase_version')} (expected v2.1)")
    fps = int(info["fps"])
    assert fps == 30, (task, fps)
    snames = names_of(feats[STATE], feats[STATE]["shape"][0], "state")
    check_layout(task, robot, L, feats, snames)
    vkeys = [L["cams"][c] for c in cams]
    for k in vkeys:
        assert feats[k]["dtype"] == "video", (task, k)
    dims = {k: feats[k]["shape"][0] for k, _, _ in L["state_parts"] + L["action_parts"]}

    episodes = jl(src / "meta/episodes.jsonl")
    total_src = len(episodes)
    n = total_src if limit is None else limit
    assert total_src >= n, (task, total_src, n)
    episodes = episodes[:n]
    assert [e["episode_index"] for e in episodes] == list(range(n)), f"{task}: episodes not contiguous 0..{n-1}"
    tasks = jl(src / "meta/tasks.jsonl")

    dst = OUT / task
    tmp = OUT / f".{task}.partial"
    if dst.exists():
        if not force:
            raise SystemExit(f"{dst} exists (use --force to rebuild the view; raw data is untouched)")
        shutil.rmtree(dst)
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "meta").mkdir(parents=True)
    s_names, a_names = part_names(feats, L["state_parts"]), part_names(feats, L["action_parts"])

    S, A, ep_rows = [], [], []
    diag = {"ts_max_dev_s": 0.0, "ts_rewritten_episodes": [], "basket_hand_diff": None,
            "h2_target_minus_state_absmax": None}
    chunks = int(info["chunks_size"])
    for e in episodes:
        i = e["episode_index"]
        chunk = i // chunks
        rel = info["data_path"].format(episode_chunk=chunk, episode_index=i)
        t = pq.read_table(src / rel)
        T = len(t)
        assert T == int(e["length"]), (task, i, T, e["length"])
        c = {k: col2d(t, k, d) for k, d in dims.items()}
        st = np.concatenate([c[k][:, a:b] for k, a, b in L["state_parts"]], 1).astype(np.float32)
        ac = np.concatenate([c[k][:, a:b] for k, a, b in L["action_parts"]], 1).astype(np.float32)
        assert np.isfinite(st).all() and np.isfinite(ac).all(), (task, i)
        if task == "basket" and c[STATE].shape[1] >= 41:  # info only: does state[29:41] equal the hand columns?
            d = float(np.abs(c[STATE][:, 29:41] - np.concatenate([c[LH], c[RH]], 1)).max())
            diag["basket_hand_diff"] = max(diag["basket_hand_diff"] or 0.0, d)
        if robot == "h2":  # info only: how far the commanded targets are from the measured state
            d = np.abs(np.concatenate([c[BT], c[HANDS]], 1) - c[STATE][:, :75]).max(0)
            prev = diag["h2_target_minus_state_absmax"]
            diag["h2_target_minus_state_absmax"] = d if prev is None else np.maximum(prev, d)
        cols = {k: t.column(k).combine_chunks() for k in SCALARS}
        fi = np.asarray(cols["frame_index"].to_numpy(zero_copy_only=False))
        assert (fi == np.arange(T)).all(), (task, i, "frame_index")
        assert (np.asarray(cols["episode_index"].to_numpy(zero_copy_only=False)) == i).all(), (task, i)
        ts = np.asarray(cols["timestamp"].to_numpy(zero_copy_only=False), dtype=np.float64)
        dev = float(np.abs(ts - fi / fps).max())
        diag["ts_max_dev_s"] = max(diag["ts_max_dev_s"], dev)
        if dev > TS_TOL and not keep_ts:  # constant-rate h264: frame k is at k/fps
            cols["timestamp"] = pa.array((fi / fps).astype(np.float32))
            diag["ts_rewritten_episodes"].append(i)
        out = pa.table({**cols, SK: fsl(st), AK: fsl(ac)})
        (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(out, tmp / rel)
        for k in vkeys:
            relv = info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=i)
            (tmp / relv).parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src / relv, tmp / relv)
        ep_rows.append({"episode_index": i, "stats": {SK: ep_stats(st), AK: ep_stats(ac)}})
        S.append(st)
        A.append(ac)
    S, A = np.concatenate(S), np.concatenate(A)
    frames = int(len(S))

    new_feats = {k: feats[k] for k in SCALARS}
    new_feats.update({k: feats[k] for k in vkeys})
    new_feats[SK] = {"dtype": "float32", "shape": [S.shape[1]], "names": s_names}
    new_feats[AK] = {"dtype": "float32", "shape": [A.shape[1]], "names": a_names}
    info2 = dict(info, features=new_feats, total_episodes=n, total_frames=frames, total_videos=n * len(vkeys),
                 total_chunks=(n - 1) // chunks + 1, splits={"train": f"0:{n}"})
    (tmp / "meta/info.json").write_text(json.dumps(info2, indent=2))
    wl(tmp / "meta/episodes.jsonl", episodes)
    wl(tmp / "meta/episodes_stats.jsonl", ep_rows)
    wl(tmp / "meta/tasks.jsonl", tasks)
    stats = {AK: full_stats(A, a_names), SK: full_stats(S, s_names)}
    (tmp / "meta/stats_psi0.json").write_text(json.dumps(stats, indent=1))

    vf = {k: [{"episode": j, "expected": int(episodes[j]["length"]),
               "frames_rate": video_frames(src / info["video_path"].format(
                   episode_chunk=j // chunks, video_key=k, episode_index=j))}
              for j in (0, n - 1)] for k in vkeys}
    if diag["h2_target_minus_state_absmax"] is not None:
        d = diag["h2_target_minus_state_absmax"]
        diag["h2_target_minus_state_absmax"] = {"body31_max": float(d[:31].max()), "hands44_max": float(d[31:].max())}
    rec = dict(
        task=task, robot=robot, source=str(src), source_episodes=total_src, episodes=n, frames=frames, fps=fps,
        image_keys=vkeys, state_key=SK, action_key=AK, state_dim=int(S.shape[1]), action_dim=int(A.shape[1]),
        body_dims=NB, perturbed_joint_names=s_names[:NB],
        state_layout=" + ".join(f"{k}[{a}:{b}]" for k, a, b in L["state_parts"]),
        action_layout=" + ".join(f"{k}[{a}:{b}]" for k, a, b in L["action_parts"]),
        raw_state_dim=len(snames), tasks=[x["task"] for x in tasks],
        constant_dims={"action": [a_names[j] for j in np.where(A.max(0) - A.min(0) < 1e-6)[0]],
                       "state": [s_names[j] for j in np.where(S.max(0) - S.min(0) < 1e-6)[0]]},
        timestamps=diag, video_frames_first_last_episode=vf,
        stats_sha256=hashlib.sha256((tmp / "meta/stats_psi0.json").read_bytes()).hexdigest(),
    )
    if robot == "g1":
        tok = A[:, :64]
        grid = np.abs(tok * 16 - np.round(tok * 16)) < 1e-3
        rec["token_diag"] = {"min": float(tok.min()), "max": float(tok.max()),
                             "per_dim_min_range": [float(tok.min(0).min()), float(tok.min(0).max())],
                             "per_dim_max_range": [float(tok.max(0).min()), float(tok.max(0).max())],
                             "frac_on_1_16_grid": float(grid.mean()), "frac_abs_le_1": float((np.abs(tok) <= 1).mean())}
        rec["hand_range"] = {"left": [float(A[:, 64:70].min()), float(A[:, 64:70].max())],
                             "right": [float(A[:, 70:76].min()), float(A[:, 70:76].max())]}
    else:
        rec["action_range"] = {"body_target31": [float(A[:, :31].min()), float(A[:, :31].max())],
                               "hands44": [float(A[:, 31:].min()), float(A[:, 31:].max())]}
    (tmp / "hb-view-receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    os.replace(tmp, dst)
    keys = ("robot", "episodes", "frames", "state_dim", "action_dim", "image_keys", "constant_dims",
            "token_diag", "hand_range", "action_range")
    print("VIEW_OK", task, json.dumps({k: rec[k] for k in keys if k in rec}), flush=True)
    if diag["ts_rewritten_episodes"]:
        print(f"NOTE {task}: timestamps of {len(diag['ts_rewritten_episodes'])} episodes deviated up to "
              f"{diag['ts_max_dev_s']:.2e}s from frame_index/fps and were rewritten in the view")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+")
    ap.add_argument("--cams", default=os.environ.get("HB_CAMS"),
                    help="comma list, cam0 first (g1: ego[,chest]; h2: head[,chest,...]); default per robot")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-timestamps", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for t in a.tasks:
        if t not in TASKS:
            raise SystemExit(f"unknown task {t}; known {list(TASKS)}")
        L = LAYOUTS[TASKS[t][2]]
        cams = [c for c in (a.cams or L["default_cams"]).split(",") if c]
        cam0 = L["default_cams"].split(",")[0]
        assert cams and cams[0] == cam0 and set(cams) <= set(L["cams"]), f"{t}: cameras {cams}, cam0 must be {cam0}"
        build(t, cams, a.force, a.keep_timestamps)


if __name__ == "__main__":
    sys.exit(main())
