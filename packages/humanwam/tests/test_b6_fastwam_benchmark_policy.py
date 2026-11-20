import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def test_b6_native_fastwam_policy_uses_current_observation_and_action_rows(monkeypatch):
    path = (
        Path(__file__).resolve().parents[3]
        / "hb_training/fastwam_tools/benchmark_policy/model.py"
    )
    spec = importlib.util.spec_from_file_location("b6_fastwam_benchmark_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    task = "Put the barrel in the bin."
    action = np.arange(16 * 76, dtype=np.float32).reshape(16, 76)
    state = np.zeros(44, dtype=np.float32)
    images = {"ego_view": object(), "chest": object()}
    calls = []
    shapes = {"state": [{"shape": 44}], "action": [{"shape": 76}]}

    class Adapter:
        steps = 10
        inp = SimpleNamespace(
            horizon=16, cams=list(images), proc=SimpleNamespace(shape_meta=shapes),
            context=calls.append,
        )

        def __init__(self, **kwargs):
            assert kwargs["ckpt"] == "step_004000.pt"
            assert kwargs["meta"] == {"cams": tuple(images)}

        def info(self):
            return {"seed": 42}

        def predict(self, obs):
            assert set(obs) == {"images", "state", "prompt"}
            assert obs["state"] is state
            assert obs["images"] is images
            assert obs["prompt"] == task
            assert self.seed == 190022
            return action

    monkeypatch.setattr(module, "load_adapter", lambda settings: Adapter)
    cfg = {
        "action_space": "sonic_dfx6", "checkpoint_path": "step_004000.pt",
        "config": {}, "task": task, "device": "cuda:0",
    }
    model = module.Model(cfg)
    assert calls == [task]
    assert model.info == {"seed_source": "benchmark request"}
    result = model.get_action({"state": state, "images": images, "seed": 190022})
    assert result["action"] is action
    assert result["execution_steps"] == 16
    assert result["sampling_steps"] == 10
    with pytest.raises(ValueError, match="state44"):
        model.get_action({"state": state[:43]})
    with pytest.raises(ValueError, match="sonic_dfx6"):
        module.Model({**cfg, "action_space": "joint47"})
