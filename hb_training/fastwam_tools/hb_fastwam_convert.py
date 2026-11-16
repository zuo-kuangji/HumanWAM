"""2026-11-16: LeRobot v2.1 copy of one of our datasets for FastWAM on OUR benchmark interface (CLAUDE.md section 4):
  g1 : state44 = observation.state[0:29] (29 body joints) + measured Inspire 6 + 6 + projected gravity 3;
       action76 = action.motion_token 64 + measured Inspire 6 + 6 (same-row labels, as GR00T / Psi-Zero / Delta Flow)
  h2 : state78 = observation.state[0:75] + projected gravity 3; action75 = action.body_target 31 + action.hands 44
  h2p: as h2 with action82 = action75 + action.planner 7 (A14 v2)
HumanWAM's fastwam_sonic_data.py expects a 43-D G1 state and takes state[0:22] + state[29:36]; on our 41-D state that
would put left-hand values into the right arm, so it is not used. Source files are never modified; media are symlinked.
--episodes N keeps the first N episodes (B7: first 100 of 107, as for every other model).
Usage: hb_fastwam_convert.py <profile> <source data dir> <fresh output dir> [--episodes N]"""
import copy, json, shutil, sys, tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

G1 = {"observation.state": 41, "action.motion_token": 64, "observation.inspire_left_hand_state": 6,
      "observation.inspire_right_hand_state": 6, "observation.projected_gravity": 3}
H2 = {"observation.state": 75, "observation.projected_gravity": 3, "action.body_target": 31, "action.hands": 44}
H2P = dict(H2, **{"action.planner": 7})
SPEC = {"g1": (G1, ("ego_view", "chest")), "h2": (H2, ("head", "chest")), "h2p": (H2P, ("head", "chest"))}


def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return {"min": v.min(0).tolist(), "max": v.max(0).tolist(), "mean": v.mean(0).tolist(), "std": v.std(0).tolist(), "count": [len(v)]}


def vec(table, key, values):
    arr = pa.array(values.tolist(), type=pa.list_(pa.float32(), values.shape[1]))
    i = table.schema.get_field_index(key)
    return table.append_column(key, arr) if i < 0 else table.set_column(i, key, arr)


def map_rows(profile, c):
    if profile == "g1":
        L, R = c["observation.inspire_left_hand_state"], c["observation.inspire_right_hand_state"]
        dev = float(np.abs(c["observation.state"][:, 29:41] - np.concatenate([L, R], 1)).max())
        assert dev < 1e-4, f"observation.state[29:41] differs from the Inspire columns by {dev}"
        return (np.concatenate([c["action.motion_token"], L, R], 1),
                np.concatenate([c["observation.state"][:, :29], L, R, c["observation.projected_gravity"]], 1))
    a = [c["action.body_target"], c["action.hands"]] + ([c["action.planner"]] if profile == "h2p" else [])
    return np.concatenate(a, 1), np.concatenate([c["observation.state"], c["observation.projected_gravity"]], 1)


def main(profile, src, out, limit=None):
    raw, cams = SPEC[profile]
    src, out = Path(src).resolve(strict=True), Path(out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists")
    info = json.loads((src / "meta/info.json").read_text())
    assert info.get("codebase_version") == "v2.1", info.get("codebase_version")
    for k, w in raw.items():
        assert info["features"][k]["shape"] == [w], (k, info["features"][k]["shape"])
    for c in cams:
        assert info["features"]["observation.images." + c]["dtype"] in ("video", "image"), c
    eps = [json.loads(l) for l in (src / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
    assert [e["episode_index"] for e in eps] == list(range(len(eps)))
    if limit is not None:
        eps = eps[:limit]
    old = {r["episode_index"]: r for r in (json.loads(l) for l in (src / "meta/episodes_stats.jsonl").read_text().splitlines() if l.strip())}
    out.parent.mkdir(parents=True, exist_ok=True)
    stg = Path(tempfile.mkdtemp(prefix=".hbfw-", dir=out.parent))
    shutil.copytree(src / "meta", stg / "meta")
    for n in ("videos", "images"):
        if (src / n).exists():
            (stg / n).symlink_to(src / n, target_is_directory=True)
    new, total = [], 0
    for e in eps:
        i = e["episode_index"]
        rel = Path(info["data_path"].format(episode_chunk=i // info["chunks_size"], episode_index=i))
        t = pq.read_table(src / rel)
        c = {k: np.asarray(t[k].to_pylist(), dtype=np.float32) for k in raw}
        act, st = map_rows(profile, c)
        assert len(act) == e["length"] and np.isfinite(act).all() and np.isfinite(st).all()
        total += len(act)
        t = vec(vec(t, "action", act), "observation.state", st)
        md = dict(t.schema.metadata or {})
        if b"huggingface" in md:
            hf = json.loads(md[b"huggingface"])
            for k, w in (("action", act.shape[1]), ("observation.state", st.shape[1])):
                hf["info"]["features"][k] = {"feature": {"dtype": "float32", "_type": "Value"}, "length": w, "_type": "Sequence"}
            md[b"huggingface"] = json.dumps(hf).encode()
            t = t.replace_schema_metadata(md)
        (stg / rel).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(t, stg / rel)
        r = copy.deepcopy(old[i])
        r["stats"]["action"], r["stats"]["observation.state"] = stats(act), stats(st)
        new.append(r)
    if limit is None:
        assert total == info["total_frames"], (total, info["total_frames"])
    else:
        info["total_episodes"], info["total_frames"] = len(eps), total
        info["total_videos"] = len(eps) * len(cams) if "total_videos" in info else info.get("total_videos")
        info["splits"] = {"train": f"0:{len(eps)}"}
        (stg / "meta/episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in eps))
    info["features"]["action"] = {"dtype": "float32", "shape": [int(act.shape[1])], "names": None}
    info["features"]["observation.state"] = {"dtype": "float32", "shape": [int(st.shape[1])], "names": None}
    drop = [k for k, v in info["features"].items() if v["dtype"] in ("video", "image") and k not in ["observation.images." + c for c in cams]]
    for k in drop:  # 2-camera rule: unused cameras are not downloaded, FastWAM checks every listed video file
        del info["features"][k]
        for r in new:
            r["stats"].pop(k, None)
    if drop and info.get("total_videos"):
        info["total_videos"] = len(eps) * len(cams)
    (stg / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n")
    (stg / "meta/episodes_stats.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new))
    for s in (stg / "meta").glob("*stats*.json"):
        s.unlink()
    (stg / "meta/hb_fastwam_interface.json").write_text(json.dumps(dict(
        profile=profile, source=str(src), cameras=list(cams), state_dim=int(st.shape[1]), action_dim=int(act.shape[1]),
        episodes=len(eps), frames=total, fps=info["fps"], first_episodes_only=limit), indent=2) + "\n")
    stg.rename(out)
    print("HBFW_CONVERT_OK", profile, out, "episodes", len(eps), "frames", total, "state", st.shape[1], "action", act.shape[1], flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    lim = None
    if "--episodes" in a:
        j = a.index("--episodes"); lim = int(a[j + 1]); del a[j:j + 2]
    main(a[0], a[1], a[2], lim)
