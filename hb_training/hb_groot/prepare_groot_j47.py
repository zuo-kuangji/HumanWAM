"""2026-11-18 user: GR00T joint47 experiment on B6 v5 (all 150 episodes of HF humanoid-benchmark/b6-pedal-bin v5/).

Aligned with HumanWAM joint47 v2 (docs/G1_JOINT_REFERENCE_DELTA.md, HumanWAM-joint @ bai-joint):
  action47 = body29 (measured observation.state, by joint name) | left Inspire6 | right Inspire6 (measured)
             | root rotation6D relative to the CURRENT heading (computed per sample in hb_j47.py).
The parquet files get one extra column, action.root_rot6d_world = world-frame rotation6D of
observation.root_orientation (wxyz), columns [R00,R10,R20,R01,R11,R21] (HumanWAM's own conversion with an identity
anchor). Its normalization bounds are fixed to [-1, 1] (as in HumanWAM). Videos are symlinked; source data untouched.
Usage: <groot python> prepare_groot_j47.py <src v5 dir> <dst dir>
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

H = Path(__file__).resolve().parent
G = H.parent
P = G.parent
spec = importlib.util.spec_from_file_location("jrr", P / "HumanWAM-joint/src/imagewam/joint_reference_rotation.py")
jrr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(jrr)
BODY = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint",
        "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
GROUPS = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22), "right_arm": (22, 29)}
ROT = "action.root_rot6d_world"
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def wl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def main(src, dst):
    src, dst = Path(src), Path(dst)
    info = json.loads((src / "meta/info.json").read_text())
    f = info["features"]
    names = f["observation.state"]["names"]
    idx = [names.index(n) for n in BODY]
    assert idx == list(range(idx[0], idx[0] + 29)), idx  # body29 contiguous in our G1 exports
    assert f["observation.root_orientation"]["shape"] == [4]
    for k in ("observation.inspire_left_hand_state", "observation.inspire_right_hand_state"):
        assert f[k]["shape"] == [6], k
    episodes = jl(src / "meta/episodes.jsonl")
    stats = jl(src / "meta/episodes_stats.jsonl")
    n = len(episodes)
    assert [e["episode_index"] for e in episodes] == list(range(n)) == [s["episode_index"] for s in stats]
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    (dst / "meta").mkdir(parents=True)
    vkeys = [k for k, v in f.items() if v.get("dtype") == "video"]
    frames = 0
    for e, s in zip(episodes, stats):
        i = e["episode_index"]
        chunk = i // int(info["chunks_size"])
        rel = info["data_path"].format(episode_chunk=chunk, episode_index=i)
        t = pq.read_table(src / rel)
        q = np.stack(t.column("observation.root_orientation").to_numpy(zero_copy_only=False)).astype(np.float64)
        r6 = jrr.world_quaternion_to_heading_relative_rotation6d(q, IDENT)  # identity anchor = world frame
        assert r6.shape == (len(q), 6) and np.isfinite(r6).all()
        t = t.append_column(ROT, pa.array(list(r6.astype(np.float32)), type=pa.list_(pa.float32())))
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, dst / rel)
        frames += len(q)
        s["stats"][ROT] = {"min": r6.min(0).tolist(), "max": r6.max(0).tolist(), "mean": r6.mean(0).tolist(),
                           "std": r6.std(0).tolist(), "count": [len(q)]}
        for k in vkeys:
            relv = info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=i)
            (dst / relv).parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src / relv, dst / relv)
    assert frames == sum(int(e["length"]) for e in episodes)
    f[ROT] = {"dtype": "float32", "shape": [6], "names": ["r00", "r10", "r20", "r01", "r11", "r21"]}
    info.update(total_episodes=n, total_frames=frames, total_videos=n * len(vkeys), splits={"train": f"0:{n}"})
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))
    wl(dst / "meta/episodes.jsonl", episodes)
    wl(dst / "meta/episodes_stats.jsonl", stats)
    (dst / "meta/tasks.jsonl").write_text((src / "meta/tasks.jsonl").read_text())
    state = {g: {"original_key": "observation.state", "start": idx[a], "end": idx[a] + (b - a)} for g, (a, b) in GROUPS.items()}
    state.update(left_hand={"original_key": "observation.inspire_left_hand_state", "start": 0, "end": 6},
                 right_hand={"original_key": "observation.inspire_right_hand_state", "start": 0, "end": 6},
                 projected_gravity={"original_key": "observation.projected_gravity", "start": 0, "end": 3})
    modality = {"state": state,
                "action": {"body29": {"original_key": "observation.state", "start": idx[0], "end": idx[0] + 29},
                           "left_hand_target": {"original_key": "observation.inspire_left_hand_state", "start": 0, "end": 6},
                           "right_hand_target": {"original_key": "observation.inspire_right_hand_state", "start": 0, "end": 6},
                           "root_rot6d": {"original_key": ROT, "start": 0, "end": 6}},
                "video": {"ego_view": {"original_key": "observation.images.ego_view"},
                          "chest": {"original_key": "observation.images.chest"}},
                "annotation": {"human.task_description": {"original_key": "task_index"}}}
    (dst / "meta/modality.json").write_text(json.dumps(modality, indent=2))
    src_g = G / "HumanWAM/third_party/Isaac-GR00T"
    env = dict(os.environ, PYTHONPATH=f"{src_g}:{H}")
    r = subprocess.run([sys.executable, str(src_g / "gr00t/data/stats.py"), "--dataset-path", str(dst), "--embodiment-tag",
                        "NEW_EMBODIMENT", "--modality-config-path", str(H / "g1_joint47_h40.py")], env=env, cwd=src_g,
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("stats failed:\n" + r.stdout[-2000:] + r.stderr[-2000:])
    st = json.loads((dst / "meta/stats.json").read_text())
    fixed = {"mean": [0.0] * 6, "std": [1.0] * 6, "min": [-1.0] * 6, "max": [1.0] * 6, "q01": [-1.0] * 6, "q99": [1.0] * 6}
    st[ROT] = {k: fixed[k] for k in st.get(ROT, fixed)} if ROT in st else fixed
    (dst / "meta/stats.json").write_text(json.dumps(st, indent=2))
    receipt = dict(task="b6v5j47", source=str(src), episodes=n, frames=frames, video_keys=vkeys, body_index=idx,
                   action47="body29 | inspire_left6 | inspire_right6 | root_rot6d (current-heading relative, hb_j47.py)",
                   action_delta_indices=list(range(0, 80, 2)), rot6d_bounds=[-1, 1],
                   humanwam_rotation="HumanWAM-joint src/imagewam/joint_reference_rotation.py",
                   modality_sha256=hashlib.sha256((dst / "meta/modality.json").read_bytes()).hexdigest(),
                   stats_sha256=hashlib.sha256((dst / "meta/stats.json").read_bytes()).hexdigest())
    (dst / "hb-dataset-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("DATASET_OK", json.dumps(receipt)[:600], flush=True)


main(sys.argv[1], sys.argv[2])
