"""G1 semantic boundaries; fake model outputs are not GPU/robot validation."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from humanwam.backends.fastwam import _check_hash
from humanwam.backends.fastwam_sonic import (
    ACTION_FIELDS,
    CONTRACT,
    LANGUAGE_KEY,
    STATE_FIELDS,
    FastWAMSonicPolicy,
    build_serve_command,
    checkpoint_manifest,
)
from humanwam.observations import observation_from_npz


def stats_fixture():
    result = {}
    for group, size in (("action", 76), ("state", 44)):
        result[group] = {"default": {
            "global_min": [-1.0] * size, "global_max": [1.0] * size,
            "global_mean": [0.0] * size, "global_std": [1.0] * size,
        }}
    result["action"]["default"]["stepwise_min"] = [[-1.0] * 76 for _ in range(16)]
    return result


class Runtime:
    def __init__(self, config, profile):
        # Keep the real native file-integrity boundary in the CPU stand-in.
        _check_hash(config["checkpoint"], config["checkpoint_sha256"])
        self.calls = []
        self.output = np.arange(16 * 76, dtype=np.float32).reshape(1, 16, 76) / 1000

    def infer(self, image, state, prompt, **kwargs):
        self.calls.append((image, state, prompt, kwargs))
        return self.output


@pytest.fixture
def setup_policy(tmp_path):
    checkpoint = tmp_path / "feedback76.pt"
    checkpoint.write_bytes(b"CPU fixture, not trained weights")
    stats = tmp_path / "stats.json"
    stats.write_text(json.dumps(stats_fixture()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(checkpoint_manifest(checkpoint, stats, action_frequency_hz=30)))
    config = {"checkpoint": str(checkpoint), "stats_path": str(stats),
              "checkpoint_manifest": str(manifest)}
    observation = {
        "video": {k: np.full((1, 1, 35, 48, 3), c, np.uint8)
                  for k, c in (("ego_view", 20), ("chest", 80))},
        "state": {k: np.full((1, 1, d), (i + 1) / 10, np.float32)
                  for i, (k, d) in enumerate(STATE_FIELDS)},
        "language": {LANGUAGE_KEY: [["pick up the pillow"]]},
    }
    return config, observation


def test_native_groot_observation_npz_state_order_and_batched_actions(setup_policy, tmp_path):
    config, observation = setup_policy
    # Exercise the existing offline CLI IO layout used by GR00T as well.
    arrays = {f"{g}.{k}": v for g in ("video", "state") for k, v in observation[g].items()}
    path = tmp_path / "observation.npz"
    np.savez(path, **arrays)
    observation = observation_from_npz(path, {"language": observation["language"]})
    policy = FastWAMSonicPolicy(config, runtime_factory=Runtime)
    assert policy._runtime is None
    prediction = policy.predict(observation)
    runtime = policy._runtime
    image, state, prompt, kwargs = runtime.calls[0]
    assert len(image) == 2
    np.testing.assert_array_equal(image[0], observation["video"]["ego_view"][0, 0])
    np.testing.assert_array_equal(image[1], observation["video"]["chest"][0, 0])
    np.testing.assert_array_equal(state, np.concatenate([observation["state"][k][0, 0]
                                                       for k, _ in STATE_FIELDS]))
    assert prompt.endswith("instruction: pick up the pillow")
    assert kwargs["action_horizon"] == 16 and kwargs["sigma_shift"] == 1.0
    assert prediction["action"].shape == (16, 76)
    np.testing.assert_array_equal(prediction["action"], runtime.output[0])
    offset = 0
    for key, width in ACTION_FIELDS:
        value = prediction["actions"][key]
        assert value.shape == (1, 16, width) and value.dtype == np.float32
        np.testing.assert_array_equal(value, runtime.output[:, :, offset:offset + width])
        offset += width
    assert prediction["info"]["normalized"] is False
    assert prediction["info"]["action_frequency_hz"] == 30
    assert prediction["info"]["robot_execution_verified"] is False
    assert policy.reset({"episode": 2}) == {}
    policy.get_action(observation, {})
    assert policy._runtime is runtime


@pytest.mark.parametrize("fault", ["hand7", "hand1000", "missing_gravity", "nan_state",
                                  "state46", "past_images", "float_rgb", "flat", "batch2"])
def test_reject_incompatible_observations_before_model_load(setup_policy, fault):
    config, observation = setup_policy
    if fault == "hand7":
        observation["state"]["left_hand"] = np.zeros((1, 1, 7), np.float32)
    elif fault == "hand1000":
        observation["state"]["left_hand"][:] = 500
    elif fault == "missing_gravity":
        del observation["state"]["projected_gravity"]
    elif fault == "nan_state":
        observation["state"]["waist"][:] = np.nan
    elif fault == "state46":
        observation["state"]["default"] = np.zeros((1, 1, 46), np.float32)
    elif fault == "past_images":
        observation["video"]["ego_view"] = np.zeros((1, 2, 8, 8, 3), np.uint8)
    elif fault == "float_rgb":
        observation["video"]["ego_view"] = observation["video"]["ego_view"].astype(np.float32)
    elif fault == "flat":
        observation = {"ego_rgb": observation["video"]["ego_view"]}
    elif fault == "batch2":
        observation["state"]["waist"] = np.zeros((2, 1, 3), np.float32)
    policy = FastWAMSonicPolicy(config, runtime_factory=Runtime)
    with pytest.raises(ValueError):
        policy.get_action(observation)
    assert policy._runtime is None


@pytest.mark.parametrize("field,value", [("horizon", 40), ("action_dim", 78), ("state_dim", 46),
                                        ("method", "a2a"), ("sigma_shift", 5.0)])
def test_no_release_or_stock_checkpoint_relabeling(setup_policy, field, value):
    config, _ = setup_policy
    with pytest.raises(ValueError, match=field):
        FastWAMSonicPolicy({**config, field: value})
    path = Path(config["checkpoint_manifest"])
    manifest = json.loads(path.read_text())
    manifest[field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=field):
        FastWAMSonicPolicy(config)


def test_asset_substitution_and_stats_mismatch_rejected(setup_policy):
    config, obs = setup_policy
    policy = FastWAMSonicPolicy(config, runtime_factory=Runtime)
    Path(config["checkpoint"]).write_bytes(b"substituted weights")
    with pytest.raises(ValueError, match="SHA256"):
        policy.predict(obs)
    stats = stats_fixture()
    stats["state"]["default"]["global_min"] = [0] * 46
    Path(config["stats_path"]).write_text(json.dumps(stats))
    with pytest.raises(ValueError, match="SHA256"):
        FastWAMSonicPolicy(config)


@pytest.mark.parametrize("output", [np.zeros((1, 40, 76)), np.zeros((1, 16, 78)),
                                    np.full((1, 16, 76), np.nan)])
def test_runtime_cannot_return_other_horizon_or_hand_schema(setup_policy, output):
    config, obs = setup_policy
    policy = FastWAMSonicPolicy(config, runtime_factory=Runtime)
    policy.predict(obs)
    policy._runtime.output = output
    with pytest.raises(ValueError, match="finite"):
        policy.predict(obs)


def test_modality_metadata_is_feedback76_not_stock78(setup_policy):
    config, _ = setup_policy
    policy = FastWAMSonicPolicy(config, runtime_factory=Runtime)
    modality = policy.get_modality_config()
    assert modality["action"]["delta_indices"] == list(range(16))
    assert modality["action"]["modality_keys"] == [k for k, _ in ACTION_FIELDS]
    assert modality["state"]["modality_keys"] == [k for k, _ in STATE_FIELDS]
    assert sum(d for _, d in STATE_FIELDS) == 44
    assert sum(d for _, d in ACTION_FIELDS) == 76
    copy.deepcopy(modality)["video"]["modality_keys"].append("bad")
    assert policy.get_modality_config()["video"]["modality_keys"] == ["ego_view", "chest"]


def test_serve_command_uses_real_sonic_rpc_module_and_environment(tmp_path):
    launcher = tmp_path / "launcher with spaces.json"
    launcher.write_text("{}")
    command = build_serve_command({"launcher_config": str(launcher), "python": "/env/python"}, tmp_path)
    assert command["argv"] == ["/env/python", "-m", "humanwam.sonic_policy_server", "--config",
                               str(launcher), "--host", "127.0.0.1", "--port", "5555"]


def test_lazy_policy_construction_never_imports_torch_or_groot(setup_policy):
    config, _ = setup_policy
    subprocess.run([sys.executable, "-c",
                    "import sys; from humanwam.backends.fastwam_sonic import FastWAMSonicPolicy; "
                    f"FastWAMSonicPolicy({config!r}); "
                    "assert not {'torch','fastwam','gr00t','transformers'} & sys.modules.keys()"],
                   check=True, capture_output=True)


def test_manifest_rejects_nonfinite_frequency_and_preserves_contract(setup_policy):
    config, _ = setup_policy
    for hz in [0, float("nan"), float("inf"), True]:
        with pytest.raises(ValueError, match="frequency"):
            checkpoint_manifest(config["checkpoint"], config["stats_path"], action_frequency_hz=hz)
    manifest = json.loads(Path(config["checkpoint_manifest"]).read_text())
    assert all(manifest[k] == v for k, v in CONTRACT.items())
