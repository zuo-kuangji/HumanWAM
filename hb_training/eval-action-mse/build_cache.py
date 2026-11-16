"""Build the model-agnostic evaluation cache for one task/split (CPU only; GR00T env: pyarrow + torchcodec).

Usage: python build_cache.py <task> <split> [--workers 8] [--force]
Writes $G/eval-action-mse/cache/<task>/stats.npz (training-frame action/state mean/std, once per task) and
       $G/eval-action-mse/cache/<task>/<split>/{meta.json, ep_XXXXXX.npz}
ep npz: state (T,Ds) f32, action (T,Da) f32, query_t (Q,), task_index, img_<cam> (Q,H,W,3) uint8 (native resolution).
The raw recordings are only read.
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hb_eval_common as E  # noqa: E402


def info_of(task):
    root = E.TASKS[task][0]
    return root, json.loads((root / "meta/info.json").read_text())


def columns_for(task, info):
    lay = E.TASKS[task][2]
    f = info["features"]
    if lay == "g1":
        names = f["observation.state"]["names"]
        body = [names.index(n) for n in E.G1_BODY]
        for k, d in (("action.motion_token", 64), ("observation.inspire_left_hand_state", 6),
                     ("observation.inspire_right_hand_state", 6), ("observation.projected_gravity", 3)):
            assert f[k]["shape"] == [d], (task, k, f[k]["shape"])
        return body, ["observation.state", "observation.inspire_left_hand_state", "observation.inspire_right_hand_state",
                      "observation.projected_gravity", "action.motion_token", "task_index", "frame_index"]
    names = f["observation.state"]["names"]
    assert len(names) == 75 and names[:31] == f["action.body_target"]["names"] and names[31:] == f["action.hands"]["names"]
    cols = ["observation.state", "observation.projected_gravity", "action.body_target", "action.hands",
            "task_index", "frame_index"]
    if lay == "h2p":  # + the 7-D SONIC planner command (action82)
        assert f["action.planner"]["shape"] == [7], (task, f["action.planner"]["shape"])
        cols.insert(4, "action.planner")
    return None, cols


def read_numeric(task, ep):
    root, info = info_of(task)
    body, cols = columns_for(task, info)
    rel = info["data_path"].format(episode_chunk=ep // int(info["chunks_size"]), episode_index=ep)
    t = pq.read_table(root / rel, columns=cols).to_pydict()
    a = {k: np.asarray(v, dtype=np.float64) for k, v in t.items() if k not in ("task_index", "frame_index")}
    assert np.array_equal(np.asarray(t["frame_index"]), np.arange(len(t["frame_index"]))), (task, ep)
    if body is not None:
        L, R = a["observation.inspire_left_hand_state"], a["observation.inspire_right_hand_state"]
        state = np.concatenate([a["observation.state"][:, body], L, R, a["observation.projected_gravity"]], -1)
        action = np.concatenate([a["action.motion_token"], L, R], -1)
    else:
        state = np.concatenate([a["observation.state"], a["observation.projected_gravity"]], -1)
        action = np.concatenate([a["action.body_target"], a["action.hands"]]
                                + ([a["action.planner"]] if "action.planner" in a else []), -1)
    ti = np.unique(np.asarray(t["task_index"]))
    assert len(ti) == 1, (task, ep, ti)
    return state.astype(np.float32), action.astype(np.float32), int(ti[0])


def decode(path, idx):
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(str(path))
    fr = dec.get_frames_at(indices=[int(i) for i in idx]).data  # (N, C, H, W) uint8
    return fr.permute(0, 2, 3, 1).contiguous().numpy()


def build_episode(args):
    task, split, ep, cams = args
    out = E.split_dir(task, split) / f"ep_{ep:06d}.npz"
    if out.exists():
        return ep, "exists"
    root, info = info_of(task)
    state, action, ti = read_numeric(task, ep)
    q = np.asarray(E.queries(len(action)), dtype=np.int64)
    arrs = {"state": state, "action": action, "query_t": q, "task_index": np.int64(ti)}
    for c in cams:
        rel = info["video_path"].format(episode_chunk=ep // int(info["chunks_size"]), video_key=f"observation.images.{c}",
                                        episode_index=ep)
        img = decode(root / rel, q)
        assert img.shape[0] == len(q) and img.dtype == np.uint8, (task, ep, c, img.shape)
        arrs[f"img_{c}"] = img
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrs)
    tmp.replace(out)
    return ep, "ok"


def train_stats(task, n_recorded):
    p = E.CACHE / task / "stats.npz"
    if p.exists():
        return
    n = E.TASKS[task][1] or n_recorded
    S, A = [], []
    for ep in range(n):
        s, a, _ = read_numeric(task, ep)
        S.append(s)
        A.append(a)
    S, A = np.concatenate(S).astype(np.float64), np.concatenate(A).astype(np.float64)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, action_mean=A.mean(0), action_std=A.std(0), state_mean=S.mean(0), state_std=S.std(0),
             train_episodes=np.int64(n), train_frames=np.int64(len(A)))


def baselines(task, split, eps):
    st = E.load_stats(task)
    lay = E.layout(task)
    res = {}
    for name in ("train_mean", "hold_last_action"):
        acc = E.Accumulator(E.norm_std(st), lay["groups"])
        for ep in eps:
            with np.load(E.split_dir(task, split) / f"ep_{ep:06d}.npz") as z:
                a = z["action"].astype(np.float64)
                for t in z["query_t"]:
                    gt, v = E.gt_chunk(a, int(t))
                    if name == "train_mean" or t == 0:
                        pred = np.repeat(st["action_mean"][None], E.HORIZON, 0)
                    else:
                        pred = np.repeat(a[t - 1][None], E.HORIZON, 0)
                    acc.add(ep, pred, gt, v)
        r = acc.result()
        res[name] = {"nmse": r["nmse"], "mse_raw": r["mse_raw"], "groups": r["groups"]}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("split")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cams", default=None, help="comma list (default: layout cameras present in the recording)")
    a = ap.parse_args()
    root, info = info_of(a.task)
    episodes = E.jl(root / "meta/episodes.jsonl")
    n_rec = len(episodes)
    assert [e["episode_index"] for e in episodes] == list(range(n_rec)), "non-contiguous episode ids"
    eps = E.split_episodes(a.task, a.split, n_rec)
    if not eps:
        raise SystemExit(f"{a.task}/{a.split}: no episodes (no held-out recordings for this task)")
    have = [k[len("observation.images."):] for k, v in info["features"].items() if v.get("dtype") == "video"]
    cams = a.cams.split(",") if a.cams else [c for c in E.layout(a.task)["cams"] if c in have]
    d = E.split_dir(a.task, a.split)
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    train_stats(a.task, n_rec)
    with Pool(a.workers) as pool:
        for ep, s in pool.imap_unordered(build_episode, [(a.task, a.split, ep, cams) for ep in eps]):
            print(f"[cache] {a.task}/{a.split} ep {ep} {s}", flush=True)
    prompts = {str(r["task_index"]): r["task"] for r in E.jl(root / "meta/tasks.jsonl")}
    meta = {"task": a.task, "split": a.split, "layout": E.TASKS[a.task][2], "source": str(root),
            "episodes": eps, "n_recorded": n_rec, "n_train": E.TASKS[a.task][1] or n_rec, "cams": cams,
            "fps": info.get("fps"), "horizon": E.HORIZON, "stride": E.STRIDE, "prompts": prompts,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    meta["baselines"] = baselines(a.task, a.split, eps)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    print(f"[cache] {a.task}/{a.split} done: {len(eps)} episodes, cams {cams}, {time.time() - t0:.0f}s, "
          f"baselines {json.dumps({k: round(v['nmse'], 4) for k, v in meta['baselines'].items()})}", flush=True)


if __name__ == "__main__":
    main()
