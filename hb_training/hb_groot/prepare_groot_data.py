"""Build GR00T dataset views (symlinked episodes + rewritten meta + stats) for the G1 tasks.

Usage: <groot python> prepare_groot_data.py <task> [<task> ...]   (tasks: b6 basket a3 a7)
B6 always uses the fixed first 100 episodes (0-99) of b6-v5t7-ready-start-20261109.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
G = H.parent
P = G.parent
OUT = G / "groot-data"
TASKS = {
    "b6": (P / "b6-v5t7-ready-start-60ep-20261109/data/deliveries/g1/b6-v5t7-ready-start-20261109", 100),
    "basket": (P / "basket-current-dynamics-100-v21-b0ff522/data", None),  # HF basket-transfer-expert-v1 @current-dynamics-100-v21 b0ff522 (old ed77719 retired)
    "basketv2": (P / "basket-trimv2-c4df4d4/data", None),  # HF a20-basket-transfer-pre-reach-trim-expert-v2 @c4df4d40 (user 2026-11-14)
    "trolley": (P / "trolley-push-expert-v1-20261110/data/trolley_push", 100),  # HF trolley-push-expert-v1 @ef9024a: fixed first 100 of 103
    "b7": (P / "b7-black-clothes-hamper-v1-1bb002a/data", 100),  # HF b7-black-clothes-hamper-expert-v1 @1bb002a, first 100 of 107
    "a6": (P / "a6-door-pull-v1-cf3ee9b/data/door_pull", None),  # HF door-pull-expert-v1 @cf3ee9b (A6)
    "b7green": (P / "b7-green-clothes-hamper-v1-ada288c/data", None),  # HF b7-green-clothes-hamper-expert-v1 @ada288c, 100 episodes (user 2026-11-13)
    "a7v2": (P / "a7-cabinet-close-v2-aecf070/data", None),  # HF a7-cabinet-close-expert-v2 @aecf070, 100 episodes (user 2026-11-14)
    "b8": (P / "b8-table-wipe-v1-3b2e618/data", None),  # HF b8-table-wipe-expert-v1 @3b2e618, 100 episodes (user 2026-11-14)
    "b1": (P / "b1-riceball-microwave-v1-36da298/data", None),  # HF b1-riceball-microwave-expert-v1 @36da298, all 115 episodes (user 2026-11-14)
    "a10v3": (P / "a10-trolley-push-v3-6e410c5/data", None),  # HF a10-trolley-push-expert-v3 @6e410c5, all 148 episodes (user 2026-11-15)
    "a3v3": (P / "a3-fridge-open-v3-24882d5/data", None),  # HF a3-fridge-open-expert-v3 @24882d5, all 125 episodes, measured hands (user 2026-11-15)
    "a7v3": (P / "a7-cabinet-close-v3-c2e15b0/data", None),  # HF a7-cabinet-close-expert-v3 @c2e15b0, all 100 episodes, measured hands (user 2026-11-15)
    "b6v6": (P / "b6-pedal-bin-v6-ad2f5bf/data", None),  # HF b6-pedal-bin @ad2f5bf0 (v6 at the repo root), all 542 episodes (user 2026-11-15)
    "b5": (P / "b5-chair-push-v1-b85b7e9/data", None),  # HF b5-chair-push-expert-v1 @b85b7e9, all 100 episodes (user 2026-11-15)
    "a3v4": (P / "a3-fridge-open-v4-6393d4a/data", None),  # HF a3-fridge-open-expert-v4 @6393d4a, all 125 episodes (user 2026-11-15: v4 replaces v3)
    "a7v4": (P / "a7-cabinet-close-v4-90cc2b5/data", None),  # HF a7-cabinet-close-expert-v4 @90cc2b5, all 100 episodes (user 2026-11-15: v4 replaces v3)
    "b7greenv2": (P / "b7-green-clothes-hamper-v2-c12757e/data", None),  # HF b7-green-clothes-hamper @c12757e (v2 at the repo root), all 120 episodes (user 2026-11-14)
    "a10v2": (P / "a10-trolley-push-v2-d3abb4c/data", None),  # HF a10-trolley-push-expert-v2 @d3abb4c, all 150 episodes (user 2026-11-13)
    "a1": (P / "a1-ground-object-pickup-v1-f47f07e/data", None),  # HF a1-ground-object-pickup-expert-v1 @f47f07e (A1)
    "c6t": (P / "c6-test8-06c231d/data", None),  # c6 8-episode adaptation test only
    "c6": (P / "c6-bathroom-door-fetch-v1-3afba69/data", None),  # HF c6-bathroom-door-fetch-expert-v1 @3afba69 (C6)
    "a3": (P / "a3-a7-20261109/a3-fridge-open-expert-v1/780fcf1e4bd7db947a8c2d8f7b92e61f24f71b45", None),
    "a7": (P / "a3-a7-20261109/a7-cabinet-close-expert-v1/1935a98e8f55d4d1d2d1615ddb97c2679713595c", None),
    "a18": (P / "a18-h2-dataset-audit-8e956a10/dataset", None),
    "a17": (P / "a17-unscrew-bottle-v1-407e6e2/data", None),  # HF a17-unscrew-bottle-expert-v1 @407e6e2
    "a18c2": (P / "a18-h2-dataset-audit-8e956a10/dataset", None),  # A18, 2 cameras head + chest (user 2026-11-14 18:2x)
    "a17c2": (P / "a17-unscrew-bottle-v1-407e6e2/data", None),  # A17, 2 cameras head + chest (user 2026-11-14 18:2x)
    "a18cr2": (P / "a18-pour-water-cropped-v2-44cdd57/data", None),  # HF a18-pour-water-expert-cropped-v2 @44cdd57, head + chest (user 2026-11-15)
    "a17cr2": (P / "a17-unscrew-bottle-cropped-v2-3520b70/data", None),  # HF a17-unscrew-bottle-expert-cropped-v2 @3520b70, head + chest (user 2026-11-15)
    "a14": (P / "a14-unscrew-bulb-v1-b4390fc/data", None),  # HF a14-unscrew-bulb-expert-v1 @b4390fc (H2, 3 cameras)
    "a14v2": (P / "a14-unscrew-bulb-v2-8411a6d/data", None),  # HF a14-unscrew-bulb-expert-v2 @8411a6da: + action.planner (H2 guide 2026-11-14)
    "a14cr3": (P / "a14-screw-bulb-cropped-v3-79fe08a/data", None),  # HF a14-screw-bulb-expert-cropped-v3 @79fe08a (trimmed from v2 @8411a6d; user 2026-11-20: newest A14)
    "b7t": (P / "b7-black-clothes-hamper-trim-949da46/data", 100),  # HF b7-black-clothes-hamper @949da46 trimmed, first 100 of 107 as FastWAM / DF / pi0.5 (user 2026-11-22)
    "b7gt": (P / "b7-green-clothes-hamper-v2trim-22e3b83/data", None),  # HF b7-green-clothes-hamper @22e3b83 trimmed v2, all 421 (user 2026-11-22)
    "b5ss": (P / "b5-chair-push-standing-start-v1-913f5a1/data", None),  # HF b5-chair-push-standing-start-expert-v1 @913f5a1, all 100 (user 2026-11-22)
}
CONFIG = {"a18": "h2w3_body31_hand44_h16.py", "a17": "h2w3_body31_hand44_h16.py", "a14": "h2w3_body31_hand44_h16.py",
          "a18c2": "h2_body31_hand44_h16.py", "a17c2": "h2_body31_hand44_h16.py",
          "a18cr2": "h2_body31_hand44_h16.py", "a17cr2": "h2_body31_hand44_h16.py",
          "a14v2": "h2_body31_hand44_planner7_h16.py", "a14cr3": "h2_body31_hand44_planner7_h16.py"}  # 2026-11-14 H2 guide: action82 incl. the planner command
# model-input cameras per H2 task (user 2026-11-12 22:0x: EVERY H2 task uses head + chest + wrist; third_person never)
CAMS = {"a18": ("head", "chest", "wrist"), "a17": ("head", "chest", "wrist"), "a14": ("head", "chest", "wrist"),
        "a18c2": ("head", "chest"), "a17c2": ("head", "chest"), "a18cr2": ("head", "chest"), "a17cr2": ("head", "chest"), "a14v2": ("head", "chest"), "a14cr3": ("head", "chest")}  # 2026-11-14: 2-camera re-train (user)
PLANNER = {"a14v2", "a14cr3"}  # tasks whose action adds action.planner (7) after body_target + hands (H2 guide a14 2.2)
H2_GROUPS = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22), "right_arm": (22, 29),
             "head": (29, 31), "left_hand": (31, 53), "right_hand": (53, 75)}


def h2_modality(info, cams=("head", "chest")):
    f = info["features"]
    names = f["observation.state"]["names"]
    assert len(names) == 75 and names[:31] == f["action.body_target"]["names"] and names[31:] == f["action.hands"]["names"]
    assert f["observation.projected_gravity"]["shape"] == [3]
    state = {g: {"original_key": "observation.state", "start": a, "end": b} for g, (a, b) in H2_GROUPS.items()}
    state["projected_gravity"] = {"original_key": "observation.projected_gravity", "start": 0, "end": 3}
    return {"state": state,
            "action": {"body_target": {"original_key": "action.body_target", "start": 0, "end": 31},
                       "left_hand_target": {"original_key": "action.hands", "start": 0, "end": 22},
                       "right_hand_target": {"original_key": "action.hands", "start": 22, "end": 44}},
            "video": {c: {"original_key": f"observation.images.{c}"} for c in cams},
            "annotation": {"human.task_description": {"original_key": "task_index"}}}
BODY = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
        "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint",
        "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
        "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
GROUPS = {"left_leg": (0, 6), "right_leg": (6, 12), "waist": (12, 15), "left_arm": (15, 22), "right_arm": (22, 29)}


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def wl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


def build(task):
    src, limit = TASKS[task]
    info = json.loads((src / "meta/info.json").read_text())
    names = info["features"]["observation.state"]["names"]
    if task in CONFIG:
        idx = list(range(31))
    else:
        idx = [names.index(n) for n in BODY]
        for g, (a, b) in GROUPS.items():
            assert idx[a:b] == list(range(idx[a], idx[a] + (b - a))), (task, g, idx[a:b])
        for key, dim in (("action.motion_token", 64), ("observation.inspire_left_hand_state", 6),
                         ("observation.inspire_right_hand_state", 6), ("observation.projected_gravity", 3),
                         ("observation.images.ego_view", None), ("observation.images.chest", None)):
            assert key in info["features"], (task, key)
            if dim:
                assert info["features"][key]["shape"] == [dim], (task, key)
    episodes = jl(src / "meta/episodes.jsonl")
    stats = jl(src / "meta/episodes_stats.jsonl")
    n = len(episodes) if limit is None else limit
    episodes, stats = episodes[:n], stats[:n]
    assert [e["episode_index"] for e in episodes] == list(range(n)) == [s["episode_index"] for s in stats]
    frames = sum(int(e["length"]) for e in episodes)
    dst = OUT / task
    if dst.exists():
        raise SystemExit(f"{dst} exists")
    (dst / "meta").mkdir(parents=True)
    vkeys = [k for k, f in info["features"].items() if f.get("dtype") == "video"]
    for e in episodes:
        i = e["episode_index"]
        chunk = i // int(info["chunks_size"])
        rel = info["data_path"].format(episode_chunk=chunk, episode_index=i)
        (dst / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(src / rel, dst / rel)
        for k in vkeys:
            relv = info["video_path"].format(episode_chunk=chunk, video_key=k, episode_index=i)
            (dst / relv).parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src / relv, dst / relv)
    if task in CONFIG:
        # stats.py computes stats for every low-dim feature in info.json and cannot handle e.g. observation.tactile
        # [10,24,24]; keep only what the H2 modality uses (parquet columns stay untouched behind the symlinks).
        keep = {"observation.state", "observation.projected_gravity", "action.body_target", "action.hands",
                "timestamp", "frame_index", "episode_index", "index", "task_index"}
        keep |= {f"observation.images.{c}" for c in CAMS[task]}
        keep |= {"action.planner"} if task in PLANNER else set()
        info["features"] = {k: v for k, v in info["features"].items() if k in keep}
        vkeys = [k for k in vkeys if k in keep]
        stats = [{**s, "stats": {k: v for k, v in s["stats"].items() if k in keep}} if "stats" in s else s
                 for s in stats]
    info.update(total_episodes=n, total_frames=frames, total_videos=n * len(vkeys),
                total_chunks=(n - 1) // int(info["chunks_size"]) + 1, splits={"train": f"0:{n}"})
    (dst / "meta/info.json").write_text(json.dumps(info, indent=2))
    wl(dst / "meta/episodes.jsonl", episodes)
    wl(dst / "meta/episodes_stats.jsonl", stats)
    (dst / "meta/tasks.jsonl").write_text((src / "meta/tasks.jsonl").read_text())
    if task in CONFIG:
        modality = h2_modality(info, CAMS[task])
        if task in PLANNER:
            assert info["features"]["action.planner"]["shape"] == [7]
            modality["action"]["planner"] = {"original_key": "action.planner", "start": 0, "end": 7}
    else:
        state = {g: {"original_key": "observation.state", "start": idx[a], "end": idx[a] + (b - a)}
                 for g, (a, b) in GROUPS.items()}
        state.update(left_hand={"original_key": "observation.inspire_left_hand_state", "start": 0, "end": 6},
                     right_hand={"original_key": "observation.inspire_right_hand_state", "start": 0, "end": 6},
                     projected_gravity={"original_key": "observation.projected_gravity", "start": 0, "end": 3})
        modality = {"state": state,
                    "action": {"motion_token": {"original_key": "action.motion_token", "start": 0, "end": 64},
                               "left_hand_joints": {"original_key": "observation.inspire_left_hand_state", "start": 0, "end": 6},
                               "right_hand_joints": {"original_key": "observation.inspire_right_hand_state", "start": 0, "end": 6}},
                    "video": {"ego_view": {"original_key": "observation.images.ego_view"},
                              "chest": {"original_key": "observation.images.chest"}},
                    "annotation": {"human.task_description": {"original_key": "task_index"}}}
    (dst / "meta/modality.json").write_text(json.dumps(modality, indent=2))
    py = sys.executable
    src_g = G / "HumanWAM/third_party/Isaac-GR00T"
    env = dict(os.environ, PYTHONPATH=f"{src_g}:{H}")
    done = None
    for tag in ("NEW_EMBODIMENT", "new_embodiment"):
        r = subprocess.run([py, str(src_g / "gr00t/data/stats.py"), "--dataset-path", str(dst), "--embodiment-tag", tag,
                            "--modality-config-path", str(H / CONFIG.get(task, "g1_feedback76_h16.py"))], env=env, cwd=src_g,
                           capture_output=True, text=True)
        if r.returncode == 0:
            done = tag
            break
        last = r.stdout[-1500:] + r.stderr[-1500:]
    if done is None:
        raise SystemExit(f"stats failed for {task}:\n{last}")
    assert (dst / "meta/stats.json").is_file()
    receipt = dict(task=task, source=str(src), episodes=n, frames=frames, video_keys=vkeys,
                   state_dim_raw=len(names), body_index=idx, stats_tag=done,
                   modality_sha256=hashlib.sha256((dst / "meta/modality.json").read_bytes()).hexdigest(),
                   stats_sha256=hashlib.sha256((dst / "meta/stats.json").read_bytes()).hexdigest())
    (dst / "hb-dataset-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("DATASET_OK", json.dumps(receipt)[:400], flush=True)


for t in sys.argv[1:]:
    build(t)
