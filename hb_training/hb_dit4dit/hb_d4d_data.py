"""Robot layouts, task table, cached dataset, per-pass distributed batch stream, state/action encoding and HB
augmentation v2 for the DiT4DiT (Mondo-Robotics/DiT4DiT @ 66a6f3a) adaptation.

One training sample = exactly the dict the native LeRobotMixtureDataset.__getitem__
(DiT4DiT/dataloader/gr00t_lerobot/datasets.py:1640-1816) hands to DiT4DiT.forward:
  image        list of 9 float16 tensors [3, 224, 448] in [0, 1]: rows t, t+2, ..., t+16 (video_delta_indices 0..16 with
               action_video_freq_ratio 2, indices clamped to the episode's last row like the native get_video).
               Each frame is [cam0 | cam1] side by side: the native multi-view layout (datasets.py:1757-1766 concatenates the
               primary view and the extra views along width per timestep). frame 0 = conditioning observation,
               frames 1..8 = future-video flow-matching targets.
               Resize: 480x640 -> 224x224 bilinear (F.interpolate, align_corners=False, no antialias, = native
               _to_img_tensor), done once in dit4dit_prepare.py and cached as uint8.
  lang         str (meta/tasks.jsonl)
  state        float16 [1, S]: state of row t (native observation_indices [0]), S = raw dims = G1 44 / H2 78.
               Every dim -> native min_max to [-1, 1] (StateActionTransform 'min_max'). No sin/cos expansion and no
               joint velocity: user rule 2026-11-10, every model gets exactly the 44 (G1) / 78 (H2) state dims of HF
               humanoid-benchmark/b6-imagewam-deltaflow76 real30-from-sim60-20261108. (The native G1 recipe
               UnitreeG1DataConfig applied StateActionSinCosTransform, which would give 73 / 109 dims - not used.)
  action       float16 [16, A]: rows t..t+15 (clamped to the episode's last row = native "first_last" padding),
               native min_max normalisation per dim to [-1, 1] (constant dims -> 0), statistics of the training view.
  action_mask  bool [16, A] all True;  state_mask bool [1, S_enc] all True  (no padding: max dims = real dims)
Augmentation (training only, CLAUDE.md v2):
  visual   hb_aug_v2.augment on x[T=9, V=2, 3, 224, 224] with obs_mask = frame 0 only: colour shared over cameras and
           time, geometry / light patches / white balance per camera fixed over time, sensor/optics effects only on the
           observation frame (not on the future-image targets), one random effect boosted x1.5.
  proprio  p=0.15 the whole state is taken from 1-2 control periods earlier (round(k*fps/control_hz) rows, not before
           the episode start); then the n_body joints (g1 29 body, h2 31 = body + head) get a per-trajectory bias
           U(+-2 deg) drawn from (HB_AUG_SEED, loader pass, episode) and per-sample N(0, 0.25 deg) noise clipped at
           3 sigma, BEFORE the min_max normalisation. History drop / joint velocity: DiT4DiT has neither input.
  Labels (actions) and statistics always use the raw values.
The loader pass reaches the workers through the index: HBLoader yields i + pass * N.
"""
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parent
G = Path(os.environ.get("HB_G", H.parent))  # $P/policies-20261109
P = Path(os.environ.get("HB_P", G.parent))  # .../preflight-20261104T1648Z (dataset parent)
try:
    import hb_aug_v2 as aug
except ImportError:  # fallback: GR00T folder next to us
    sys.path.append(str(G / "hb_groot"))
    import hb_aug_v2 as aug

IMG, HORIZON = 224, 16
VIDEO_DELTA = list(range(17))  # native dit4dit_g1.yaml video_delta_indices
VIDEO_RATIO = 2  # native action_video_freq_ratio (dit4dit_g1.yaml, run_real_robot.sh)
VIDEO_OFFSETS = VIDEO_DELTA[::VIDEO_RATIO]  # [0, 2, ..., 16] -> 9 frames
BIAS_DEG, NOISE_DEG, DELAY_P = 2.0, 0.25, 0.15
TAG = "new_embodiment"  # native stats tag for unknown robot types (EmbodimentTag.NEW_EMBODIMENT)


@dataclass(frozen=True)
class Layout:
    name: str
    raw_state_dim: int
    action_dim: int
    cameras: tuple  # (cam0 = left half, cam1 = right half)
    n_body: int  # leading raw state dims that get bias + noise
    segments: tuple  # ((kind, start, end), ...) over the raw state, kind in {"sincos", "minmax"}
    state_desc: str
    action_desc: str

    @property
    def state_dim(self) -> int:  # encoded
        return sum((2 if k == "sincos" else 1) * (b - a) for k, a, b in self.segments)


STATE_DIMS = {"g1": 44, "h2": 78}  # user rule 2026-11-10: model state input == these raw dims, no velocity, no sin/cos
LAYOUTS = {
    "g1": Layout("g1", 44, 76, ("ego_view", "chest"), 29, (("minmax", 0, 44),),
                 "body29(observation.state by name) | inspire_left6 | inspire_right6 | projected_gravity3",
                 "action.motion_token64 | inspire_left6 | inspire_right6 (same-row measured hands)"),
    "h2": Layout("h2", 78, 75, ("head", "chest"), 31, (("minmax", 0, 78),),
                 "observation.state75 (legs12|waist3|arms14|head2|hands44) | projected_gravity3",
                 "action.body_target31 | action.hands44"),
}
for _k, _L in LAYOUTS.items():
    assert _L.raw_state_dim == _L.state_dim == STATE_DIMS[_k] and all(s[0] == "minmax" for s in _L.segments), (
        f"{_k}: model state must be the raw {STATE_DIMS[_k]} dims (no sin/cos, no velocity)")
G1_BODY = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
           "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
           "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint",
           "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
           "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
           "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
           "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]


@dataclass(frozen=True)
class Task:
    root: Path  # LeRobot v2.1 root
    limit: int | None  # episode limit (B6: always the fixed first 100)
    episodes: int  # expected episodes in the view
    frames: int | None  # expected frames
    layout: str
    control_hz: float  # controller rate (state delay in rows = round(k * fps / control_hz))


TASKS = {
    "b6": Task(P / "b6-v5t7-ready-start-60ep-20261109/data/deliveries/g1/b6-v5t7-ready-start-20261109", 100, 100, 65127, "g1", 50.0),
    "a3": Task(P / "a3-a7-20261109/a3-fridge-open-expert-v1/780fcf1e4bd7db947a8c2d8f7b92e61f24f71b45", None, 100, 28180, "g1", 30.0),
    "a7": Task(P / "a3-a7-20261109/a7-cabinet-close-expert-v1/1935a98e8f55d4d1d2d1615ddb97c2679713595c", None, 100, 30664, "g1", 30.0),
    "a18": Task(P / "a18-h2-dataset-audit-8e956a10/dataset", None, 100, 72752, "h2", 30.0),
}


def layout_of(task: str) -> Layout:
    return LAYOUTS[TASKS[task].layout]


def data_root() -> Path:
    return Path(os.environ.get("HB_D4D_DATA", G / "dit4dit-data"))


def task_dir(task: str) -> Path:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    return data_root() / task


def load_meta(task_or_root) -> dict:
    root = Path(task_or_root) if ("/" in str(task_or_root) or "\\" in str(task_or_root)) else task_dir(str(task_or_root))
    if not (root / "READY").exists():
        raise FileNotFoundError(f"{root}/READY missing; run dit4dit_prepare.py first")
    return json.loads((root / "meta.json").read_text())


# ---------------------------------------------------------------------------------------------------------------- encoding
def minmax(x, lo, hi):
    """Native StateActionTransform 'min_max' (transform/state_action.py:153-173): 2(x-min)/(max-min)-1, constant dims -> 0."""
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    ok = hi != lo
    out = np.zeros_like(x)
    out[..., ok] = 2.0 * (x[..., ok] - lo[ok]) / (hi[ok] - lo[ok]) - 1.0
    return out


def unminmax(y, lo, hi):
    """Inverse of minmax for deployment (constant dims return min)."""
    y = np.asarray(y, dtype=np.float64)
    lo, hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
    return (y + 1.0) / 2.0 * (hi - lo) + lo


def encode_state(raw, layout: Layout, smin, smax):
    """raw [..., raw_state_dim] -> [..., state_dim]; per segment: sincos -> [sin | cos], minmax -> native min_max."""
    raw = np.asarray(raw, dtype=np.float64)
    parts = []
    for kind, a, b in layout.segments:
        x = raw[..., a:b]
        if kind == "sincos":
            parts += [np.sin(x), np.cos(x)]
        else:
            parts.append(minmax(x, np.asarray(smin)[a:b], np.asarray(smax)[a:b]))
    out = np.concatenate(parts, axis=-1)
    assert out.shape[-1] == layout.state_dim == STATE_DIMS[layout.name], (out.shape, layout.state_dim)
    return out


# ------------------------------------------------------------------------------------------------------------ proprio aug
def trajectory_bias(seed: int, epoch: int, episode: int, n: int) -> np.ndarray:
    """Per-joint +-2 deg bias, fixed for (seed, loader pass, episode) -> identical on every rank/worker."""
    rng = np.random.default_rng([int(seed), int(epoch), int(episode)])
    return rng.uniform(-math.radians(BIAS_DEG), math.radians(BIAS_DEG), size=n)


def joint_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    s = math.radians(NOISE_DEG)
    return np.clip(rng.normal(0.0, s, size=n), -3 * s, 3 * s)


def delay_rows(rng: np.random.Generator, fps: float, control_hz: float) -> int:
    if rng.random() >= DELAY_P:
        return 0
    k = int(rng.integers(1, 3))  # 1 or 2 control periods
    return max(1, round(k * fps / control_hz))


# ---------------------------------------------------------------------------------------------------------------- dataset
class HBD4DDataset(torch.utils.data.Dataset):
    def __init__(self, task_or_root, *, augment: bool = False, seed: int | None = None):
        root = Path(task_or_root) if ("/" in str(task_or_root) or "\\" in str(task_or_root)) else task_dir(str(task_or_root))
        meta = load_meta(root)
        self.meta = meta
        self.layout = LAYOUTS[meta["layout"]]
        low = np.load(root / "lowdim.npz")
        self.state = low["state"].astype(np.float32)  # raw
        self.action = low["action"].astype(np.float32)  # raw
        self.ep_end = low["ep_end"].astype(np.int64)
        self.episode = low["episode_index"].astype(np.int64)
        _, first = np.unique(self.episode, return_index=True)
        self.ep_start = first[self.episode].astype(np.int64)  # episodes are contiguous and indexed 0..n-1
        L = self.layout
        assert self.state.shape[1:] == (L.raw_state_dim,) and self.action.shape[1:] == (L.action_dim,), (
            self.state.shape, self.action.shape)
        nrm = meta["norm"]
        self.amin, self.amax = np.asarray(nrm["action_min"]), np.asarray(nrm["action_max"])
        self.smin, self.smax = np.asarray(nrm["state_min"]), np.asarray(nrm["state_max"])
        self.n = len(self.state)
        self.root, self.augment = str(root), bool(augment)
        self.prompt = meta["prompt"]
        self.fps = float(meta["fps"])
        self.control_hz = float(meta.get("control_hz", self.fps))
        self.seed = int(os.environ.get("HB_AUG_SEED", "42")) if seed is None else int(seed)
        self._frames = self._rng = self._pid = None
        self._aug_log = None

    def __len__(self):
        return self.n

    def __getstate__(self):  # never pickle memmaps / rng into DataLoader workers
        d = dict(self.__dict__)
        d["_frames"] = d["_rng"] = d["_pid"] = d["_aug_log"] = None
        return d

    def _lazy_init(self):
        if self._pid == os.getpid():
            return
        self._frames = {c: np.load(f"{self.root}/frames_{c}.npy", mmap_mode="r") for c in self.layout.cameras}
        if torch.utils.data.get_worker_info() is not None:
            torch.set_num_threads(1)
        # torch.initial_seed() differs per worker and per loader iterator; mix in the global rank as well.
        self._rng = np.random.default_rng([torch.initial_seed() % (2**63), int(os.environ.get("RANK", "0"))])
        self._pid = os.getpid()
        log_dir = os.environ.get("HB_AUG_LOG")
        self._aug_log = [Path(log_dir) / f"aug-rank{os.environ.get('RANK', '0')}-pid{os.getpid()}.jsonl", 0] if log_dir else None

    # rows used by sample i
    def action_rows(self, i: int) -> np.ndarray:
        return np.minimum(np.arange(i, i + HORIZON), self.ep_end[i] - 1)

    def video_rows(self, i: int) -> np.ndarray:
        return np.minimum(i + np.asarray(VIDEO_OFFSETS), self.ep_end[i] - 1)

    def proprio(self, i: int, epoch: int) -> tuple[np.ndarray, int]:
        """Augmented RAW state for row i (training): optional delay, then trajectory bias + per-sample noise."""
        d = delay_rows(self._rng, self.fps, self.control_hz)
        j = max(i - d, int(self.ep_start[i]))
        state = self.state[j].astype(np.float64)
        nb = self.layout.n_body
        state[:nb] += trajectory_bias(self.seed, epoch, self.episode[i], nb) + joint_noise(self._rng, nb)
        return state, i - j

    def images(self, i: int) -> np.ndarray:
        rows = self.video_rows(i)
        return np.stack([np.asarray(self._frames[c][rows]) for c in self.layout.cameras], axis=1)  # [9, 2, 224, 224, 3]

    def __getitem__(self, index):
        epoch, i = divmod(int(index), self.n)
        self._lazy_init()
        x = torch.from_numpy(self.images(i)).permute(0, 1, 4, 2, 3).float().div_(255.0)  # [T, V, 3, H, W]
        if self.augment:
            obs_mask = torch.zeros(x.shape[0], dtype=torch.bool)
            obs_mask[0] = True  # frame 0 = observation; frames 1.. = future-image targets
            x, p = aug.augment(x, self._rng, obs_mask)
            self._log_aug(p, i, epoch)
            raw_state, _ = self.proprio(i, epoch)
        else:
            raw_state = self.state[i].astype(np.float64)
        frames = torch.cat([x[:, v] for v in range(x.shape[1])], dim=-1).clamp_(0, 1).to(torch.float16)  # [T, 3, H, V*W]
        state = encode_state(raw_state, self.layout, self.smin, self.smax)[None].astype(np.float16)
        action = minmax(self.action[self.action_rows(i)], self.amin, self.amax).astype(np.float16)
        return dict(action=action, image=list(frames.unbind(0)), lang=self.prompt, state=state,
                    action_mask=np.ones(action.shape, dtype=bool), state_mask=np.ones(state.shape, dtype=bool))

    def _log_aug(self, p, i, epoch):
        if self._aug_log is None or self._aug_log[1] >= 20:
            return
        path, k = self._aug_log
        rec = {"row": int(i), "pass": int(epoch), "version": aug.VERSION,
               **{a: b for a, b in p.items() if a in ("boost", "exposure", "contrast", "saturation", "hue", "gamma",
                                                      "temperature", "geometry", "noise", "jpeg")}}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
        except OSError:
            pass
        self._aug_log[1] = k + 1

    # native schema: DiT4DiT/dataloader/gr00t_lerobot/datasets.py:2093 (LeRobotMixtureDataset.save_dataset_statistics)
    def save_dataset_statistics(self, path) -> None:
        st = self.meta["stats"]
        stats = {TAG: {
            "action": {**st["action"], "mask": [True] * self.layout.action_dim},
            "state": st["state"],
            "num_transitions": int(self.n),
            "num_trajectories": int(self.meta["episodes"]),
            # HB additions (native code ignores unknown keys): how the model inputs/targets were built
            "hb_action_normalization": {"mode": "min_max", "min": self.amin.tolist(), "max": self.amax.tolist(),
                                        "inverse": "a_raw = (a + 1) / 2 * (max - min) + min"},
            "hb_state_encoding": {"segments": [list(s) for s in self.layout.segments], "raw_dim": self.layout.raw_state_dim,
                                  "encoded_dim": self.layout.state_dim, "min": self.smin.tolist(), "max": self.smax.tolist(),
                                  "raw_layout": self.layout.state_desc},
            "hb_layout": {"layout": self.layout.name, "cameras": list(self.layout.cameras),
                          "image": "[cam0 | cam1] side by side, each 224x224 (480x640 stretched, bilinear)",
                          "action_layout": self.layout.action_desc, "horizon": HORIZON},
        }}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(str(path) + ".tmp")
        tmp.write_text(json.dumps(stats, indent=2))
        os.replace(tmp, path)


# --------------------------------------------------------------------------------------------------------------- loading
def collate_list(batch):  # native lerobot_datasets.collate_fn: the model consumes a list of sample dicts
    return batch


class HBLoader:
    """Per-pass distributed batches (replaces the native DataLoader + accelerate BatchSamplerShard).

    Pass p: perm = default_rng([seed, p]).permutation(N); global batch b covers perm[b*G:(b+1)*G], rank r takes its
    B-slice (G = B * world, drop_last). Indices are encoded as i + p*N (pass -> per-trajectory bias). Deterministic in
    (seed, p), so stream(start_step) resumes exactly at the batch a resumed run needs.
    """

    def __init__(self, dataset, *, per_gpu_batch: int, rank: int, world: int, num_workers: int, seed: int = 42,
                 prefetch_factor: int = 2):
        self.dataset = dataset
        self.batch_size, self.rank, self.world = int(per_gpu_batch), int(rank), int(world)
        self.global_batch = self.batch_size * self.world
        self.num_workers, self.seed, self.prefetch_factor = int(num_workers), int(seed), int(prefetch_factor)
        self.n = len(dataset)
        self.batches_per_pass = self.n // self.global_batch
        if self.batches_per_pass < 1:
            raise ValueError(f"dataset of {self.n} rows < one global batch {self.global_batch}")

    def __len__(self):
        return self.batches_per_pass

    def pass_batches(self, p: int) -> list:
        perm = np.random.default_rng([self.seed, int(p)]).permutation(self.n)
        out = []
        for b in range(self.batches_per_pass):
            s = b * self.global_batch + self.rank * self.batch_size
            out.append([int(i) + int(p) * self.n for i in perm[s:s + self.batch_size]])
        return out

    def stream(self, start_step: int = 0):
        p, b0 = divmod(int(start_step), self.batches_per_pass)
        while True:
            batches = self.pass_batches(p)[b0:]
            kw = dict(num_workers=self.num_workers, collate_fn=collate_list, pin_memory=False)
            if self.num_workers > 0:
                kw.update(prefetch_factor=self.prefetch_factor, persistent_workers=False)
            loader = torch.utils.data.DataLoader(self.dataset, batch_sampler=batches, **kw)
            yield from loader
            p, b0 = p + 1, 0
