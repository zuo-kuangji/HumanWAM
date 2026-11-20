"""Deploy a B6-v6 native FastWAM checkpoint through Humanoid_benchmark."""

import os
import sys
from pathlib import Path

import numpy as np

CAMERAS = ("ego_view", "chest")
HORIZON = 16
STATE_DIM = 44
ACTION_DIM = 76


def load_adapter(settings):
    root = Path(settings["humanwam_root"]).expanduser().resolve(strict=True)
    source = Path(settings["fastwam_source"]).expanduser().resolve(strict=True)
    cache = Path(settings["model_cache"]).expanduser().resolve(strict=True)
    os.environ["FASTWAM_SRC"] = str(source)
    os.environ["DIFFSYNTH_MODEL_BASE_PATH"] = str(cache)
    eval_root = root / "hb_training" / "eval-action-mse"
    sys.path[:0] = [str(eval_root), str(eval_root / "adapters")]
    from ad_fastwam import Adapter

    return Adapter


class Model:
    history_length = 1

    def __init__(self, cfg):
        if cfg["action_space"] != "sonic_dfx6":
            raise ValueError("B6 FastWAM requires sonic_dfx6")
        self.task = cfg["task"]
        settings = cfg["config"]
        self.adapter = load_adapter(settings)(
            ckpt=cfg["checkpoint_path"], task=self.task, split=None,
            meta={"cams": CAMERAS}, device=cfg["device"],
        )
        inputs = self.adapter.inp
        processor = inputs.proc
        shapes = processor.shape_meta
        if (
            inputs.horizon != HORIZON
            or tuple(inputs.cams) != CAMERAS
            or shapes["state"][0]["shape"] != STATE_DIM
            or shapes["action"][0]["shape"] != ACTION_DIM
        ):
            raise ValueError("Expected the B6-v6 state44/action76, ego/chest and 16-step checkpoint")
        inputs.context(self.task)
        self.info = self.adapter.info()
        self.info.pop("seed")
        self.info["seed_source"] = "benchmark request"

    def get_action(self, obs):
        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (STATE_DIM,):
            raise ValueError(f"Expected state44, got {state.shape}")
        self.adapter.seed = int(obs["seed"])
        action = self.adapter.predict({
            "images": obs["images"],
            "prompt": self.task,
            "state": state,
        })
        if action.shape != (HORIZON, ACTION_DIM):
            raise ValueError(f"Expected action76 with {HORIZON} rows, got {action.shape}")
        return {
            "action": action,
            "execution_steps": HORIZON,
            "sampling_steps": self.adapter.steps,
        }
