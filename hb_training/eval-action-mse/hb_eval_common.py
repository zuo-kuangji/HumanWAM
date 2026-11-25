"""Humanoid Benchmark open-loop action-reconstruction evaluation: shared task table, cache layout and metrics.

Protocol (identical for every model; mirrors Isaac-GR00T gr00t/eval/open_loop_eval.py):
  * query frames t = 0, 16, 32, ... over the whole episode (non-overlapping 16-step chunks);
  * the model sees the observation at t exactly as at deployment (cameras, raw benchmark state 44/78, instruction;
    Delta Flow also its native 16-step past-action history), all augmentation off;
  * the predicted chunk (first 16 steps, raw action units) is compared with the recorded actions t..t+15;
    rows beyond the episode end are excluded (no clamped / padded targets);
  * metrics: raw MSE / MAE, and nMSE = MSE after dividing every action dim by its std over the task's TRAINING
    frames (floored, see STD_FLOOR_FRAC), overall, per action group and per horizon step.
Splits: "heldout" = recorded episodes that were never trained on (B6 100-148, trolley 100-102, B7 100-106);
        "train10" = 10 fixed, evenly spaced training episodes (fit on seen data, NOT generalisation).
Only numpy here: imported by every model environment.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

G = Path(os.environ.get("G", "/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/"
                             "preflight-20261104T1648Z/policies-20261109"))
P = G.parent
ROOT = G / "eval-action-mse"
CACHE = Path(os.environ.get("HB_EVAL_CACHE", ROOT / "cache"))
HORIZON = 16
STRIDE = 16
N_TRAIN_EVAL = 10
STD_FLOOR_FRAC = 0.05  # per-dim std floor = 5 % of the median std of the action vector
# 2026-11-20 user: a dim whose std over all training frames is ~0 (never moves in this task, e.g. A18 left Sharpa
# hand) has no meaningful normalised error -> excluded from every nMSE; its raw error is reported in "const_dims"
CONST_STD_EPS = 1e-6
METRIC = "nmse-v2: dims constant over training excluded"

# task -> (source LeRobot v2.1 root, number of TRAINING episodes = fixed first N (None = all recorded), layout)
TASKS = {
    "b6": (P / "b6-v5t7-ready-start-60ep-20261109/data/deliveries/g1/b6-v5t7-ready-start-20261109", 100, "g1"),
    "basket": (P / "basket-current-dynamics-100-v21-b0ff522/data", None, "g1"),
    "trolley": (P / "trolley-push-expert-v1-20261110/data/trolley_push", 100, "g1"),
    "b7": (P / "b7-black-clothes-hamper-v1-1bb002a/data", 100, "g1"),
    "b7green": (P / "b7-green-clothes-hamper-v1-ada288c/data", None, "g1"),
    "a6": (P / "a6-door-pull-v1-cf3ee9b/data/door_pull", None, "g1"),
    "a10v2": (P / "a10-trolley-push-v2-d3abb4c/data", None, "g1"),
    "a1": (P / "a1-ground-object-pickup-v1-f47f07e/data", None, "g1"),
    "c6": (P / "c6-bathroom-door-fetch-v1-3afba69/data", None, "g1"),
    "a3": (P / "a3-a7-20261109/a3-fridge-open-expert-v1/780fcf1e4bd7db947a8c2d8f7b92e61f24f71b45", None, "g1"),
    "a7": (P / "a3-a7-20261109/a7-cabinet-close-expert-v1/1935a98e8f55d4d1d2d1615ddb97c2679713595c", None, "g1"),
    "a18": (P / "a18-h2-dataset-audit-8e956a10/dataset", None, "h2"),
    "a17": (P / "a17-unscrew-bottle-v1-407e6e2/data", None, "h2"),
    "a14": (P / "a14-unscrew-bulb-v1-b4390fc/data", None, "h2"),
    # 2026-11-15: newer datasets (GR00T in-training eval needs a cache for every task it trains on)
    "a10v3": (P / "a10-trolley-push-v3-6e410c5/data", None, "g1"),
    "a7v2": (P / "a7-cabinet-close-v2-aecf070/data", None, "g1"),
    "b1": (P / "b1-riceball-microwave-v1-36da298/data", None, "g1"),
    "b7greenv2": (P / "b7-green-clothes-hamper-v2-c12757e/data", None, "g1"),
    "b8": (P / "b8-table-wipe-v1-3b2e618/data", None, "g1"),
    "basketv2": (P / "basket-trimv2-c4df4d4/data", None, "g1"),
    "a14v2": (P / "a14-unscrew-bulb-v2-8411a6d/data", None, "h2p"),  # action 82 = hb75 + SONIC planner 7
    # 2026-11-15 evening: v6/v4 datasets (a3v3, a7v3 retired, not registered)
    "b6v6": (P / "b6-pedal-bin-v6-ad2f5bf/data", None, "g1"),  # 542 ep, all used (B6 first-100 rule is v5t7 only)
    "b5": (P / "b5-chair-push-v1-b85b7e9/data", None, "g1"),
    "a3v4": (P / "a3-fridge-open-v4-6393d4a/data", None, "g1"),
    "a7v4": (P / "a7-cabinet-close-v4-90cc2b5/data", None, "g1"),
    # 2026-11-16: B5 standing start (HF b5-chair-push-standing-start-expert-v1 @913f5a16, 100 ep; FastWAM b5ss)
    "b5ss": (P / "b5-chair-push-standing-start-v1-913f5a1/data", None, "g1"),
    "a9": (P / "a9-pedal-open-8d43206/data", None, "g1"),  # 2026-11-24 user: A9 pedal bin lid open (HF a9-pedal-open-expert-v1 @8d432063, all 100 ep, G1, 30 Hz)
    "b1v2": (P / "b1-riceball-microwave-v2-30hz-b58f043/data", None, "g1"),  # 2026-11-25 user: B1 v2 (HF b1-riceball-microwave-expert-v2-30hz @b58f0434, all 200 ep, G1, 30 Hz)
    "b4": (P / "b4-cushion-194e7cd/data", None, "g1"),  # 2026-11-25 user: B4 cushion bed -> sofa (HF b4-cushion @194e7cde, v2 at the repo root, all 302 ep, G1, 30 Hz)
    "a20kosmo": (P / "a20-basket-transfer-kosmo-30hz-v1-7324dbe/data", None, "g1"),  # 2026-11-25 user: A20 Kosmo replay (same states / actions as basketv2, Kosmo-scanned scene videos)
    # 2026-11-17: B7 black clothes trimmed (HF b7-black-clothes-hamper @949da46, 107 ep; train = first 100, heldout 100-106)
    "b7t": (P / "b7-black-clothes-hamper-trim-949da46/data", 100, "g1"),
    # 2026-11-15 night: H2 cropped v2 (GR00T h32 relay item 71); head+chest only -> build_cache --cams head,chest
    "a18cr2": (P / "a18-pour-water-cropped-v2-44cdd57/data", None, "h2"),  # HF a18-pour-water-expert-cropped-v2 @44cdd571
    "a17cr2": (P / "a17-unscrew-bottle-cropped-v2-3520b70/data", None, "h2"),  # HF a17-unscrew-bottle-expert-cropped-v2 @3520b70f
    # 2026-11-22 user: every training must run the in-training MSE -> register every dataset before it is trained on
    "b7gt": (P / "b7-green-clothes-hamper-v2trim-22e3b83/data", None, "g1"),  # HF b7-green-clothes-hamper @22e3b83 trimmed v2, 421 ep
    # 2026-11-24 (owner): B7 with the 10 standing frames before dock dropped (every episode starts at dock;
    # debug/tasks/b7/tools/trim/b7_drop_lead.py, bai faddba64): black HF @619f3427 (107 ep, train = first 100,
    # heldout 100-106), green HF @4d2c613c (421 ep)
    "b7td": (P / "b7-black-clothes-hamper-leaddrop-619f342/data", 100, "g1"),
    "b7gtd": (P / "b7-green-clothes-hamper-v2leaddrop-4d2c613/data", None, "g1"),
    "a14cr3": (P / "a14-screw-bulb-cropped-v3-79fe08a/data", None, "h2p"),  # HF a14-screw-bulb @79fe08a cropped-v3, head + chest, action82
}


def task_for_source(source):
    """benchmark task whose recording root is <source> (groot-data/<view>/hb-dataset-receipt.json "source")"""
    src = Path(source).resolve()
    hits = [t for t, (root, _, _) in TASKS.items() if Path(root).resolve() == src]
    return hits[0] if hits else None
G1_BODY = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
           "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
           "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
           "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
           "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
           "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
           "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
           "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
LAYOUTS = {
    # state: body29 | inspire L6 | inspire R6 | projected_gravity3 ; action: motion_token64 | inspire L6 | R6 (same row)
    "g1": dict(state_dim=44, action_dim=76, cams=("ego_view", "chest"),
               groups={"motion_token": (0, 64), "hands": (64, 76)}),
    # state: observation.state75 (legs12 waist3 arms14 head2 hands22+22) | gravity3 ; action: body_target31 | hands44
    "h2": dict(state_dim=78, action_dim=75, cams=("head", "chest", "wrist"),
               groups={"body_target": (0, 31), "hands": (31, 75)}),
    # 2026-11-15 user: A14 v2 is trained on action82 = body_target31 | hands44 | action.planner7 (SONIC planner
    # command); all 82 dims are scored (planner reported as its own group)
    "h2p": dict(state_dim=78, action_dim=82, cams=("head", "chest"),
                groups={"body_target": (0, 31), "hands": (31, 75), "planner": (75, 82)}),
}


def jl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def layout(task):
    return LAYOUTS[TASKS[task][2]]


def split_episodes(task, split, n_recorded):
    _, n_train, _ = TASKS[task]
    n_train = n_recorded if n_train is None else n_train
    if split == "heldout":
        return list(range(n_train, n_recorded))
    if split == "train10":
        return sorted({int(round(x)) for x in np.linspace(0, n_train - 1, N_TRAIN_EVAL)})
    if split.startswith("ep:"):  # explicit list, e.g. ep:100,101
        return [int(x) for x in split[3:].split(",")]
    raise ValueError(f"unknown split {split!r}")


def split_dir(task, split):
    return CACHE / task / split


def load_meta(task, split):
    return json.loads((split_dir(task, split) / "meta.json").read_text())


def load_stats(task):
    z = np.load(CACHE / task / "stats.npz")
    return {k: z[k] for k in z.files}


def norm_std(stats):
    """per-dim divisor; np.inf marks a dim that is constant over training (excluded from nMSE, see CONST_STD_EPS)"""
    std = stats["action_std"].astype(np.float64)
    out = np.maximum(std, STD_FLOOR_FRAC * float(np.median(std)))
    out[std < CONST_STD_EPS] = np.inf
    return out


def iter_episodes(task, split, max_episodes=None):
    meta = load_meta(task, split)
    eps = meta["episodes"][: max_episodes or None]
    for ep in eps:
        with np.load(split_dir(task, split) / f"ep_{ep:06d}.npz") as z:
            yield ep, {k: z[k] for k in z.files}


class Accumulator:
    """Sums of squared / absolute errors; everything is computed in float64 over valid (row, dim) entries."""

    def __init__(self, std, groups, horizon=HORIZON):
        self.std = np.asarray(std, dtype=np.float64)
        self.groups = groups
        self.live = np.isfinite(self.std)  # False = constant over training -> not part of nMSE
        self.n_live = int(self.live.sum())
        D = len(self.std)
        self.sse_raw = np.zeros(D)
        self.sae_raw = np.zeros(D)
        self.sse_n = np.zeros(D)
        self.rows = 0
        self.step_sse_n = np.zeros(horizon)
        self.step_rows = np.zeros(horizon)
        self.per_episode = {}

    def add(self, ep, pred, gt, valid):
        pred = np.asarray(pred, dtype=np.float64)
        gt = np.asarray(gt, dtype=np.float64)
        v = np.asarray(valid, dtype=bool)
        if pred.shape != gt.shape:
            raise ValueError(f"pred {pred.shape} vs gt {gt.shape}")
        if not np.isfinite(pred[v]).all():
            raise FloatingPointError("non-finite prediction")
        e = (pred - gt)[v]
        en = np.where(self.live, e / np.where(self.live, self.std, 1.0), 0.0)
        self.sse_raw += (e ** 2).sum(0)
        self.sae_raw += np.abs(e).sum(0)
        self.sse_n += (en ** 2).sum(0)
        self.rows += int(v.sum())
        k = np.nonzero(v)[0]
        self.step_sse_n[k] += (en ** 2).sum(1)
        self.step_rows[k] += 1
        s = self.per_episode.setdefault(int(ep), [0.0, 0])
        s[0] += float((en ** 2).sum())
        s[1] += int(v.sum()) * self.n_live

    def result(self):
        L = self.live
        D = max(self.n_live, 1)
        n = max(self.rows, 1)
        out = {"rows": self.rows, "metric": METRIC, "dims_scored": self.n_live,
               "mse_raw": float(self.sse_raw[L].sum() / (n * D)),
               "mae_raw": float(self.sae_raw[L].sum() / (n * D)), "nmse": float(self.sse_n[L].sum() / (n * D)),
               "groups": {}, "nmse_by_step": (self.step_sse_n / np.maximum(self.step_rows, 1) / D).tolist(),
               "nmse_by_dim": (self.sse_n / n).tolist()}
        for g, (a, b) in self.groups.items():
            k = int(L[a:b].sum())
            if k:
                out["groups"][g] = {"mse_raw": float(self.sse_raw[a:b][L[a:b]].sum() / (n * k)),
                                    "nmse": float(self.sse_n[a:b][L[a:b]].sum() / (n * k))}
        c = ~L
        if c.any():
            k = int(c.sum())
            m = float(self.sse_raw[c].sum() / (n * k))
            out["const_dims"] = {"n": k, "idx": np.nonzero(c)[0].tolist(), "mse_raw": m, "rms_raw": float(np.sqrt(m)),
                                 "mae_raw": float(self.sae_raw[c].sum() / (n * k))}
        pe = {ep: s / max(c, 1) for ep, (s, c) in sorted(self.per_episode.items())}
        out["nmse_per_episode"] = pe
        vals = np.array(list(pe.values()))
        if len(vals) > 1:  # bootstrap-free spread: std error over episodes
            out["nmse_episode_sem"] = float(vals.std(ddof=1) / np.sqrt(len(vals)))
        return out


def queries(T):
    return list(range(0, T, STRIDE))


def gt_chunk(action, t, horizon=HORIZON):
    T = len(action)
    idx = np.arange(t, t + horizon)
    valid = idx < T
    return action[np.minimum(idx, T - 1)], valid


def run(adapter, task, split, out_path, max_episodes=None, max_queries=None, save_pred=True, log_every=50):
    """Evaluate one adapter on one cached split; writes <out_path> (json) and <out_path>.pred.npz."""
    meta = load_meta(task, split)
    lay = LAYOUTS[meta["layout"]]
    stats = load_stats(task)
    acc = Accumulator(norm_std(stats), lay["groups"])
    preds, keys = [], []
    t0 = time.time()
    infer_s = 0.0
    nq = 0
    for ep, z in iter_episodes(task, split, max_episodes):
        state, action = z["state"], z["action"]
        qts = list(z["query_t"])
        for qi, t in enumerate(qts):
            obs = {"episode": ep, "t": int(t), "T": len(action), "state": state[t], "prompt": meta["prompts"][str(int(z["task_index"]))],
                   "images": {c: z[f"img_{c}"][qi] for c in meta["cams"]},
                   "past_action": action[max(0, t - HORIZON):t],
                   "_gt_check": action[t:t + HORIZON]}  # only for adapters' input-pipeline consistency checks
            s = time.time()
            pred = np.asarray(adapter.predict(obs), dtype=np.float64)
            infer_s += time.time() - s
            if pred.ndim != 2 or pred.shape[0] < HORIZON or pred.shape[1] != lay["action_dim"]:
                raise ValueError(f"adapter returned {pred.shape}, need (>={HORIZON}, {lay['action_dim']})")
            pred = pred[:HORIZON]
            gt, valid = gt_chunk(action, t)
            acc.add(ep, pred, gt, valid)
            if save_pred:
                preds.append(pred.astype(np.float32))
                keys.append((ep, int(t)))
            nq += 1
            if log_every and nq % log_every == 0:
                print(f"[eval] {task}/{split} {nq} queries, running nMSE {acc.result()['nmse']:.4f}, "
                      f"{infer_s / nq:.3f} s/query", flush=True)
            if max_queries and nq >= max_queries:
                break
        if max_queries and nq >= max_queries:
            break
    res = acc.result()
    res.update(task=task, split=split, layout=meta["layout"], episodes=sorted(acc.per_episode), queries=nq,
               horizon=HORIZON, stride=STRIDE, std_floor_frac=STD_FLOOR_FRAC, wall_s=round(time.time() - t0, 1),
               infer_s_per_query=round(infer_s / max(nq, 1), 4), adapter=adapter.info(),
               baselines=meta.get("baselines", {}), cache_created=meta.get("created"),
               partial=bool(max_episodes or max_queries), finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1))
    tmp.replace(out_path)
    if save_pred and preds:
        np.savez_compressed(str(out_path) + ".pred.npz", pred=np.stack(preds).astype(np.float16),
                            keys=np.asarray(keys, dtype=np.int64))
    return res
