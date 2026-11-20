"""HumanWAM-owned G1 ODE adapter contracts; injected model boundary is not GPU validation."""

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from humanwam.backends import groot_ode as module

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "packages/ode_g1"
pytestmark = pytest.mark.skipif(not (SOURCE / "src/ode_g1/config.py").is_file(), reason="Owned ODE package absent")


def digest(value):
    return hashlib.sha256(value).hexdigest()


@pytest.fixture
def config(tmp_path):
    head = {"horizon": 16, "body_dim": 64, "hand_dim": 12, "state_dim": 44,
            "feature_dim": 1024, "cond_dim": 512, "width": 512, "layers": 4, "field_scale": 1.0}
    stats = {
        "schema": "g1-task-ode-q01q99-fixed-hands-v1", "task": "test-task", "task_revision": "v1",
        "hand_target_time_offset": 0, "train_split_mode": "all_episodes", "validation_episode_ids": [],
    }
    for name, width, start in (("state", 44, 29), ("action", 76, 64)):
        low, high = [-1.0] * width, [3.0] * width
        low[start:start + 12] = [0.0] * 12
        high[start:start + 12] = [1.0] * 12
        stats[name + "_bounds"] = {"q01": low, "q99": high}
    (tmp_path / "ode_config.json").write_text(json.dumps(head))
    (tmp_path / "normalization.json").write_text(json.dumps(stats))
    # Explicit fixture only; actual owned loader still rejects non-model files.
    data = b"not-a-model; runtime is injected in these contract tests"
    (tmp_path / "final_head.safetensors").write_bytes(data)
    binding = {
        "config_sha256": digest((tmp_path / "ode_config.json").read_bytes()),
        "normalizer_sha256": digest((tmp_path / "normalization.json").read_bytes()),
        "camera_keys": list(module.NATIVE_CAMERAS),
        "base_identity": {"revision": "2fc962b973bccdd5d8ce4f67cc63b264d6886495", "weights": {}},
        "task": {"name": "test-task", "revision": "v1"},
    }
    manifest = {"schema": "ode-g1-policy-v1", "binding": binding,
                "binding_sha256": module._binding_digest(binding),
                "head": {"path": "final_head.safetensors", "size_bytes": len(data), "sha256": digest(data)}}
    (tmp_path / "policy_manifest.json").write_text(json.dumps(manifest))
    native = {"device": "cpu", "dataset": {"name": "test-task", "revision": "v1"},
              "paths": {"output": str(tmp_path)}, "head": head,
              "base_model": {"path": str(tmp_path / "base"), "cosmos_path": str(tmp_path / "cosmos")},
              "inference": {"verify_head_sha256": True}}
    path = tmp_path / "native.yaml"
    path.write_text(yaml.safe_dump(native))
    return {"source_dir": str(SOURCE), "native_config": str(path)}


@pytest.fixture
def observation():
    return {
        "video": {key: np.full((1, 1, 12, 18, 3), i + 5, np.uint8) for i, key in enumerate(module.CAMERAS)},
        "state": {key: np.full((1, 1, width), i / 10, np.float32)
                  for i, (key, width) in enumerate(module.STATE_FIELDS)},
        "language": {module.LANGUAGE_KEY: [["Move the pillow"]]},
    }


class Normalizer:
    def __init__(self):
        self.inputs = []

    def normalize_actions_unclipped(self, value):
        self.inputs.append(value.copy())
        # Deliberately out-of-range; adapter must preserve the issued coordinate.
        return value * 2 - 1


class Runtime:
    def __init__(self, options, native):
        self.normalizer = Normalizer()
        self.calls = []

    def predict_rgb(self, images, instruction, state, *, initial_body):
        self.calls.append(("rgb", images, instruction, state, initial_body.copy()))
        return np.concatenate((np.tile(initial_body, (16, 1)), np.full((16, 12), 0.25)), axis=-1)

    def predict_cached(self, condition, state, *, initial_body):
        self.calls.append(("cached", condition, state, initial_body.copy()))
        return np.concatenate((np.tile(initial_body, (16, 1)), np.full((16, 12), 0.75)), axis=-1)


def condition():
    return {"backbone_features": np.zeros((1, 3, 2048), np.float16),
            "backbone_attention_mask": np.array([[1, 1, 0]], bool),
            "image_mask": np.array([[1, 0, 0]], bool)}


def test_rgb_request_owned_carry_and_native_modality_schema(config, observation):
    policy = module.GrootODEPolicy(config, runtime_factory=Runtime)
    assert policy._runtime is None
    schema = policy.get_modality_config()
    assert schema["action"]["delta_indices"] == list(range(16))
    assert schema["video"]["modality_keys"] == ["ego_view", "chest"]
    actions, info = policy.get_action(observation, {"at_episode_start": True})
    assert {key: value.shape for key, value in actions.items()} == {
        "motion_token": (1, 16, 64), "left_hand_joints": (1, 16, 6), "right_hand_joints": (1, 16, 6)}
    assert all(value.dtype == np.float32 for value in actions.values())
    runtime = policy._runtime
    np.testing.assert_array_equal(runtime.calls[0][-1], 0)
    np.testing.assert_array_equal(runtime.calls[0][1][module.NATIVE_CAMERAS[0]], observation["video"]["ego_view"][0, 0])
    assert runtime.calls[0][2] == "Move the pillow"
    expected_state = np.concatenate([observation["state"][key][0, 0] for key, _ in module.STATE_FIELDS])
    np.testing.assert_array_equal(runtime.calls[0][3], expected_state)
    sent = np.full(64, 20, np.float32)
    policy.get_action(observation, {"previous_body64": sent})
    np.testing.assert_array_equal(runtime.normalizer.inputs[-1], np.r_[sent, np.zeros(12)])
    np.testing.assert_array_equal(runtime.calls[-1][-1], 39)  # no [-5,5] clipping
    # A subsequent BOS request does not reuse the previous result's endpoint.
    policy.get_action(observation, {"at_episode_start": True})
    np.testing.assert_array_equal(runtime.calls[-1][-1], 0)
    assert len(runtime.normalizer.inputs) == 1
    assert info["action_frequency_hz"] == 30 and info["normalized"] is False
    assert policy.reset({"episode": "new"}) == {}
    with pytest.raises(ValueError, match="Each request"):
        policy.get_action(observation)


@pytest.mark.parametrize("options", [None, {}, {"at_episode_start": False}, {"at_episode_start": 1},
    {"at_episode_start": True, "previous_body64": [0] * 64}, {"previous_body64": [0] * 63},
    {"previous_body64": [float("nan")] * 64}, {"previous_body64": [1j] * 64},
    {"previous_body64": [True] * 64}, {"previous_body64": [0] * 64, "commit": True}])
def test_invalid_or_ambiguous_carry_rejected_before_runtime(config, observation, options):
    policy = module.GrootODEPolicy(config, runtime_factory=Runtime)
    with pytest.raises(ValueError):
        policy.get_action(observation, options)
    assert policy._runtime is None


@pytest.mark.parametrize("change", ["hands7", "float64", "nan", "camera_history", "missing_camera", "future_field", "language"])
def test_bad_rgb_observations_do_not_load_model(config, observation, change):
    observation = copy.deepcopy(observation)
    if change == "hands7":
        observation["state"]["left_hand"] = np.zeros((1, 1, 7), np.float32)
    elif change == "float64":
        observation["state"]["left_leg"] = np.zeros((1, 1, 6), np.float64)
    elif change == "nan":
        observation["state"]["left_leg"][:] = np.nan
    elif change == "camera_history":
        observation["video"]["ego_view"] = np.zeros((1, 2, 12, 18, 3), np.uint8)
    elif change == "missing_camera":
        del observation["video"]["chest"]
    elif change == "future_field":
        observation["future_action"] = np.zeros((16, 76))
    else:
        observation["language"][module.LANGUAGE_KEY] = [""]
    policy = module.GrootODEPolicy(config, runtime_factory=Runtime)
    with pytest.raises(ValueError):
        policy.get_action(observation, {"at_episode_start": True})
    assert policy._runtime is None


def test_predict_extracts_offline_metadata_without_mutating_inputs(config, observation):
    policy = module.GrootODEPolicy(config, runtime_factory=Runtime)
    inputs = {**observation, "at_episode_start": True}
    result = policy.predict(inputs)
    assert inputs["at_episode_start"] is True
    assert result["action"].shape == (16, 76)
    result = policy.predict({"condition": condition(), "current_state": np.zeros(44, np.float32),
                             "previous_body64": np.full(64, 4, np.float32)})
    assert result["action"].shape == (16, 76)
    assert policy._runtime.calls[-1][0] == "cached"
    np.testing.assert_array_equal(policy._runtime.calls[-1][-1], 7)


@pytest.mark.parametrize("change", ["feature_width", "future_slot", "no_text", "no_image", "nan", "extra_target", "wrong_state"])
def test_cached_schema_rejects_leakage_and_bad_shapes_before_loading(config, change):
    policy = module.GrootODEPolicy(config, runtime_factory=Runtime)
    cached = condition()
    state = np.zeros(44, np.float32)
    if change == "feature_width":
        cached["backbone_features"] = np.zeros((1, 3, 1024), np.float32)
    elif change == "future_slot":
        cached["frame_ids"] = np.array([[0, 1, 0]], np.int64)
    elif change == "no_text":
        cached["image_mask"][:] = True
    elif change == "no_image":
        cached["image_mask"][:] = False
    elif change == "nan":
        cached["backbone_features"][0, 0, 0] = np.nan
    elif change == "extra_target":
        cached["target"] = np.zeros((16, 76))
    else:
        state = np.zeros(43, np.float32)
    with pytest.raises(ValueError):
        policy.predict_cached(cached, state, options={"at_episode_start": True})
    assert policy._runtime is None


@pytest.mark.parametrize("file", ["ode_config.json", "normalization.json", "final_head.safetensors", "policy_manifest.json"])
def test_changed_checkpoint_binding_is_rejected_on_cpu(config, file):
    root = Path(config["native_config"]).parent
    path = root / file
    if file == "policy_manifest.json":
        document = json.loads(path.read_text())
        document["binding"]["task"]["name"] = "other"
        path.write_text(json.dumps(document))
    else:
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        module.GrootODEPolicy(config, runtime_factory=Runtime)


def test_native_config_rejects_download_opt_in_and_horizon_relabeling(config):
    for overrides in ({"base_model.allow_download": True}, {"head.horizon": 8}):
        with pytest.raises(ValueError):
            module.GrootODEPolicy({**config, "native_overrides": overrides}, runtime_factory=Runtime)


def test_import_and_construction_are_model_and_transport_lazy(config):
    script = "from humanwam.backends.groot_ode import GrootODEPolicy; import sys; "
    script += "GrootODEPolicy(" + repr(config) + "); "
    script += "assert not {'torch','transformers','gr00t','ode_g1','zmq'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)


def test_owned_package_does_not_require_upstream_git_head_or_clean_worktree(config, monkeypatch):
    def no_external_pin(*args):
        raise AssertionError("Owned ODE code must not be pinned to an external Git commit")

    monkeypatch.setattr(module, "_git_source", no_external_pin)
    policy = module.GrootODEPolicy(config, runtime_factory=Runtime)
    assert policy.source == SOURCE
    assert policy._runtime is None


def test_wrong_gr00t_installation_is_rejected_before_loading(config, monkeypatch):
    wrong = ROOT / "third_party/Isaac-GR00T/gr00t/__init__.py"
    monkeypatch.setattr(module.importlib.util, "find_spec", lambda name: SimpleNamespace(origin=str(wrong)))
    with pytest.raises(ValueError, match="Installed GR00T"):
        module._native_imports(SOURCE)


@pytest.mark.parametrize("pin", ["Isaac-GR00T-ODE", "Isaac-GR00T"])
def test_native_clients_receive_real_ode_actions_over_legacy_zmq(config, observation, monkeypatch, pin):
    pytest.importorskip("zmq")
    pytest.importorskip("msgpack_numpy")
    from test_sonic_policy_server import _native_classes, _running_server

    source = ROOT / "third_party" / pin
    if not (source / "gr00t/policy/server_client.py").is_file():
        pytest.skip("Optional native GR00T client source absent")
    monkeypatch.setenv("HUMANWAM_GR00T_SOURCE", str(source))
    native = _native_classes()
    policy = module.GrootODEPolicy(config, runtime_factory=Runtime)
    with _running_server(policy) as (server, _):
        host, port = server.endpoint.removeprefix("tcp://").split(":")
        client = native["PolicyClient"](host=host, port=int(port))
        try:
            assert client.ping()
            schema = client.get_modality_config()
            assert isinstance(schema["action"], native["ModalityConfig"])
            assert schema["action"].delta_indices == list(range(16))
            assert schema["action"].modality_keys == [key for key, _ in module.ACTION_FIELDS]
            assert policy._runtime is None
            actions, info = client.get_action(observation, {"at_episode_start": True})
            assert actions["motion_token"].shape == (1, 16, 64)
            assert actions["left_hand_joints"].shape == (1, 16, 6)
            assert all(value.dtype == np.float32 for value in actions.values())
            assert info["normalized"] is False
            assert client.reset({"episode": 1}) == {}
            with pytest.raises(RuntimeError, match="Each request"):
                client.get_action(observation)
            sent = np.full(64, 20, np.float32)
            actions, _ = client.get_action(observation, {"previous_body64": sent})
            np.testing.assert_array_equal(actions["motion_token"], 39)
            assert client.ping()
        finally:
            # The23ace client predates close()/context-manager support.
            client.socket.close(linger=0)
            client.context.term()


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="Optional CPU Torch absent")
def test_actual_native_runtime_uses_affine_sent_token_and_never_commits(config, observation):
    import torch

    package = str(SOURCE / "src")
    if package not in sys.path:
        sys.path.insert(0, package)
    from ode_g1.data import TaskNormalizer
    from ode_g1.runtime import ODEPolicy

    class Head:
        config = SimpleNamespace(horizon=16)

        def eval(self):
            return self

        def predict_cached(self, batch, state, initial):
            body = initial[:, None].expand(1, 16, 64)
            return {"action_pred": torch.cat((body, torch.zeros((1, 16, 12))), dim=-1)}

    def factory(options, cfg):
        return ODEPolicy(Head(), TaskNormalizer.load(Path(cfg["paths"]["output"]) / "normalization.json"),
                         device="cpu", rgb_encoder=lambda images, text: condition())

    policy = module.GrootODEPolicy(config, runtime_factory=factory)
    actions, _ = policy.get_action(observation, {"previous_body64": np.full(64, 20, np.float32)})
    np.testing.assert_allclose(actions["motion_token"], 20)
    np.testing.assert_allclose(actions["left_hand_joints"], 0.5)
    assert policy._runtime.carry.executed_steps == 0
    actions, _ = policy.get_action(observation, {"at_episode_start": True})
    np.testing.assert_allclose(actions["motion_token"], 1)  # Normalized-zero BOS, not raw-zero.
    assert policy._runtime.carry.executed_steps == 0
    actions, _ = policy.predict_cached(condition(), np.zeros(44, np.float32),
                                       options={"previous_body64": np.full(64, -20, np.float32)})
    np.testing.assert_allclose(actions["motion_token"], -20)
    assert policy._runtime.carry.executed_steps == 0
