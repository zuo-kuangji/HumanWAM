"""Robot layouts, task table, cached torch Dataset, HB augmentation v2, and the OpenPI hooks.
(The file name dates from the G1-only version; it now covers the G1 tasks b6/basket/a3/a7 and the H2 task a18.)

Sample produced by HBCachedDataset (consumed by hb_g1_config.HBInputs), per robot layout (LAYOUTS):
  observation/<cam>  uint8 [224, 224, 3] for the layout's two cameras (OpenPI resize_with_pad of 480x640, done in hb_prepare)
                       g1: ego_view -> base_0_rgb, chest -> left_wrist_0_rgb;  h2: head -> base_0_rgb, chest -> left_wrist_0_rgb
  observation/state  float32 [44] g1 = body29 | inspire L6 | inspire R6 | gravity3
                     float32 [78] h2 = observation.state75 | gravity3
  actions            float32 [16, 76] g1 = rows t..t+15 of (motion_token64 | inspire L6 | inspire R6)
                     float32 [16, 75] h2 = rows t..t+15 of (action.body_target31 | action.hands44)
                     (clamped to the episode's last row, LeRobot delta_timestamps convention)
  prompt             str (meta/tasks.jsonl)
Augmentation (training only, augment=True), CLAUDE.md v2:
  visual   hb_aug_v2.augment on the two 224px views after the resize, letterbox rows re-zeroed.
  proprio  (sec. 3) with p=0.15 the whole state observation is taken from 1-2 control periods earlier
           (round(k * fps / control_hz) rows, clamped to the episode start); then the leading n_body joints
           (g1: 29 body, h2: 31 = body + head) get a per-trajectory bias U(+-2 deg), drawn deterministically from
           (HB_AUG_SEED, loader pass, episode_index), so it is fixed within an episode for a pass and redrawn each pass,
           plus per-sample N(0, 0.25 deg) noise clipped at 3 sigma. Same constants as hb_aug_v2.perturb_joints, which
           draws bias and noise together and therefore cannot be used per trajectory.
           History-drop and joint velocity (sec. 3/4) do not apply: pi0.5 has no state history / velocity input.
  The loader pass reaches the workers through the index: hb_train's sampler yields i + pass * N.
Action labels and norm stats (hb_prepare.py) always use the raw values.
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

HORIZON, IMG, MODEL_ACTION_DIM = 16, 224, 76  # one 76-D pi05 init checkpoint for all robots
BIAS_DEG, NOISE_DEG, DELAY_P = 2.0, 0.25, 0.15


@dataclass(frozen=True)
class Layout:
    name: str
    state_dim: int
    action_dim: int  # raw; zero-padded to MODEL_ACTION_DIM by OpenPI PadStatesAndActions after Normalize
    cameras: tuple  # (-> base_0_rgb, -> left_wrist_0_rgb)
    n_body: int  # leading state dims that get bias + noise
    max_token_len: int  # prompt + discretized state must fit (verify_cpu.py checks)
    state_desc: str
    action_desc: str


LAYOUTS = {
    "g1": Layout("g1", 44, 76, ("ego_view", "chest"), 29, int(os.environ.get("HB_PI05_MAX_TOKEN_LEN_G1", "256")),
                 "body29(observation.state by name) | inspire_left6 | inspire_right6 | projected_gravity3",
                 "action.motion_token64 | inspire_left6 | inspire_right6 (same-row observed hands)"),
    "h2": Layout("h2", 78, 75, ("head", "chest"), 31, int(os.environ.get("HB_PI05_MAX_TOKEN_LEN_H2", "384")),
                 "observation.state75 (legs12|waist3|arms14|head2|hands44) | projected_gravity3",
                 "action.body_target31 | action.hands44 (+1 zero pad dim -> 76)"),
}
G1_BODY = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
           "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
           "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint",
           "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
           "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
           "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
           "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
BODY = G1_BODY  # backwards-compatible name


@dataclass(frozen=True)
class Task:
    root: Path  # LeRobot v2.1 root
    limit: int | None  # episode limit (B6: always the fixed first 100)
    episodes: int  # expected episodes in the view
    frames: int | None  # expected frames (None = not known in advance)
    layout: str
    control_hz: float  # assumption: the controller rate the recording was made at (used for the state delay)


TASKS = {
    "b6": Task(P / "b6-v5t7-ready-start-60ep-20261109/data/deliveries/g1/b6-v5t7-ready-start-20261109", 100, 100, None, "g1", 50.0),
    "basket": Task(P / "g1-basket-transfer-ed77719-40epochs/dataset", None, 100, 54337, "g1", 30.0),
    "a3": Task(P / "a3-a7-20261109/a3-fridge-open-expert-v1/780fcf1e4bd7db947a8c2d8f7b92e61f24f71b45", None, 100, 28180, "g1", 30.0),
    "a7": Task(P / "a3-a7-20261109/a7-cabinet-close-expert-v1/1935a98e8f55d4d1d2d1615ddb97c2679713595c", None, 100, 30664, "g1", 30.0),
    "a18": Task(P / "a18-h2-dataset-audit-8e956a10/dataset", None, 100, 72752, "h2", 30.0),
}


def layout_of(task: str) -> Layout:
    return LAYOUTS[TASKS[task].layout]


def config_name(task: str) -> str:
    return f"pi05_hb_{TASKS[task].layout}_{task}"


def repo_id(task: str) -> str:
    return f"hb_{TASKS[task].layout}_{task}"


def asset_id(task: str) -> str:
    return f"{TASKS[task].layout}_{task}"


def data_root() -> Path:
    return Path(os.environ.get("HB_PI05_DATA", G / "pi05-data"))


def task_dir(task: str) -> Path:
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: {sorted(TASKS)}")
    return data_root() / task


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


class HBCachedDataset(torch.utils.data.Dataset):
    def __init__(self, root, *, action_horizon: int = HORIZON, augment: bool = False, seed: int | None = None):
        root = Path(root) if ("/" in str(root) or "\\" in str(root)) else task_dir(str(root))
        if not (root / "READY").exists():
            raise FileNotFoundError(f"{root}/READY missing; run hb_prepare.py first")
        meta = json.loads((root / "meta.json").read_text())
        self.layout = LAYOUTS[meta.get("layout", "g1")]
        low = np.load(root / "lowdim.npz")
        self.state = low["state"].astype(np.float32)
        self.action = low["action"].astype(np.float32)
        self.ep_end = low["ep_end"].astype(np.int64)
        self.episode = low["episode_index"].astype(np.int64)
        _, first = np.unique(self.episode, return_index=True)
        self.ep_start = first[self.episode].astype(np.int64)  # episodes are contiguous and indexed 0..n-1
        L = self.layout
        assert self.state.shape[1:] == (L.state_dim,) and self.action.shape[1:] == (L.action_dim,), (self.state.shape, self.action.shape)
        self.n = len(self.state)
        self.root, self.h, self.augment = str(root), int(action_horizon), bool(augment)
        self.prompt = meta["prompt"]
        self.box = tuple(meta["content_box"])  # top, bottom, left, right of the non-padded area in the 224 image
        self.fps = float(meta["fps"])
        self.control_hz = float(meta.get("control_hz", self.fps))
        self.seed = int(os.environ.get("HB_AUG_SEED", "42")) if seed is None else int(seed)
        self._frames = self._rng = self._pid = None

    def __len__(self):
        return self.n

    def __getstate__(self):  # never pickle memmaps / rng into DataLoader workers
        d = dict(self.__dict__)
        d["_frames"] = d["_rng"] = d["_pid"] = None
        return d

    def _lazy_init(self):
        if self._pid == os.getpid():
            return
        self._frames = {c: np.load(f"{self.root}/frames_{c}.npy", mmap_mode="r") for c in self.layout.cameras}
        if torch.utils.data.get_worker_info() is not None:
            torch.set_num_threads(1)
        # torch.initial_seed() differs per worker; OpenPI seeds every rank's loader identically, so mix in the global rank.
        self._rng = np.random.default_rng([torch.initial_seed() % (2**63), int(os.environ.get("RANK", "0"))])
        self._pid = os.getpid()

    def proprio(self, i: int, epoch: int) -> tuple[np.ndarray, int]:
        """Augmented state for row i (training): optional delay, then trajectory bias + per-sample noise."""
        d = delay_rows(self._rng, self.fps, self.control_hz)
        j = max(i - d, int(self.ep_start[i]))
        state = self.state[j].copy()
        nb = self.layout.n_body
        state[:nb] += (trajectory_bias(self.seed, epoch, self.episode[i], nb) + joint_noise(self._rng, nb)).astype(np.float32)
        return state, i - j

    def __getitem__(self, index):
        epoch, i = divmod(int(index), self.n)
        self._lazy_init()
        idx = np.minimum(np.arange(i, i + self.h), self.ep_end[i] - 1)
        cams = self.layout.cameras
        imgs = np.stack([np.asarray(self._frames[c][i]) for c in cams])  # [2, 224, 224, 3] uint8
        actions = self.action[idx].copy()
        if self.augment:
            imgs = self._augment_images(imgs)
            state, _ = self.proprio(i, epoch)
        else:
            state = self.state[i].copy()
        return {f"observation/{cams[0]}": imgs[0], f"observation/{cams[1]}": imgs[1], "observation/state": state,
                "actions": actions, "prompt": self.prompt}

    def _augment_images(self, imgs: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().div_(255.0)[None]  # [T=1, V=2, 3, 224, 224]
        x, _ = aug.augment(x, self._rng)
        out = x[0].mul(255.0).round_().clamp_(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy().copy()
        top, bottom, left, right = self.box  # keep OpenPI's letterbox exactly black, as at inference
        out[:, :top] = 0
        out[:, bottom:] = 0
        out[:, :, :left] = 0
        out[:, :, right:] = 0
        return out


G1CachedDataset = HBCachedDataset  # backwards-compatible name


def disable_openpi_image_aug():
    """PI0Pytorch._preprocess_observation (models_pytorch/pi0_pytorch.py) calls
    preprocessing_pytorch.preprocess_observation_pytorch(observation, train=True) in forward(); with train=True it does
    random crop 95% + rotate +-5 deg (non-wrist) and brightness/contrast/saturation jitter. Force train=False (the resize
    and default-mask logic still run)."""
    import openpi.models_pytorch.preprocessing_pytorch as pp

    orig = pp.preprocess_observation_pytorch
    if getattr(orig, "_hb_noaug", False):
        return

    def no_aug(observation, *, train=False, **kwargs):  # noqa: ARG001
        return orig(observation, train=False, **kwargs)

    no_aug._hb_noaug = True
    pp.preprocess_observation_pytorch = no_aug


def install_dataset_hook(*, augment: bool):
    """openpi.training.data_loader.create_torch_data_loader calls create_torch_dataset(data_config, action_horizon,
    model_config); route repo_id 'hb_<layout>_<task>' to HBCachedDataset. Everything after it (transform_dataset,
    Normalize, model transforms, DistributedSampler, TorchDataLoader) is unchanged OpenPI code."""
    import openpi.training.data_loader as dl

    orig = getattr(dl.create_torch_dataset, "_hb_orig", dl.create_torch_dataset)

    def create_torch_dataset(data_config, action_horizon, model_config):
        rid = data_config.repo_id or ""
        parts = rid.split("_", 2)
        if len(parts) == 3 and parts[0] == "hb" and parts[2] in TASKS:
            if action_horizon != HORIZON:
                raise ValueError(f"action_horizon {action_horizon} != {HORIZON}")
            return HBCachedDataset(parts[2], action_horizon=action_horizon, augment=augment)
        return orig(data_config, action_horizon, model_config)

    create_torch_dataset._hb_orig = orig
    dl.create_torch_dataset = create_torch_dataset


def install_hooks(*, augment: bool = True):
    disable_openpi_image_aug()
    install_dataset_hook(augment=augment)
