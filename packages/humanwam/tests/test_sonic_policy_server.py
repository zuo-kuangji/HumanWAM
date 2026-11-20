"""Real loopback RPC, with optional pinned GR00T client interoperability."""

import ast
import functools
import io
import json
import os
import queue
import subprocess
import sys
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from humanwam.sonic_policy_server import MsgSerializer, PolicyServer, _modality_envelopes

pytest.importorskip("zmq")
msgpack = pytest.importorskip("msgpack")
mnp = pytest.importorskip("msgpack_numpy")
import zmq  # noqa: E402


class Policy:
    def get_action(self, observation, options=None):
        if observation.get("bad"):
            raise ValueError("bad observation")
        return {
            "motion_token": np.asarray(observation["state"], dtype=np.float32),
            "left_hand_joints": np.zeros((1, 16, 6), np.float32),
            "right_hand_joints": np.ones((1, 16, 6), np.float32),
        }, {"options": options}

    def reset(self, options=None):
        return {"reset_options": options}

    def get_modality_config(self):
        return {
            "video": {"delta_indices": [0], "modality_keys": ["ego_view", "chest"]},
            "action": {
                "delta_indices": list(range(16)),
                "modality_keys": ["motion_token", "left_hand_joints", "right_hand_joints"],
            },
        }


@contextmanager
def _running_server(policy):
    ready = queue.Queue()
    failures = queue.Queue()

    def worker():
        try:
            with PolicyServer(policy, port=0) as server:
                ready.put(server)
                server.run()
        except BaseException as exc:
            failures.put(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    server = ready.get(timeout=5)
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, 3000)
    socket.setsockopt(zmq.SNDTIMEO, 3000)
    socket.connect(server.endpoint)
    yield server, socket
    server.running = False
    socket.close(linger=0)
    context.term()
    thread.join(timeout=5)
    assert not thread.is_alive(), "RPC server did not stop"
    if not failures.empty():
        raise failures.get()


@pytest.fixture
def running_server():
    with _running_server(Policy()) as server:
        yield server


def call(socket, request):
    serializer = MsgSerializer()
    socket.send(serializer.to_bytes(request))
    return serializer.from_bytes(socket.recv())


def test_real_loopback_numpy_action_reset_modality_and_kill(running_server):
    server, socket = running_server
    assert call(socket, {"endpoint": "ping"})["status"] == "ok"
    state = np.arange(1024, dtype=np.float32).reshape(1, 16, 64)
    actions, info = call(
        socket,
        {"data": {"observation": {"state": state}, "options": {"seed": 7}}},
    )
    np.testing.assert_array_equal(actions["motion_token"], state)
    assert actions["motion_token"].dtype == np.float32
    assert actions["left_hand_joints"].shape == (1, 16, 6)
    assert info == {"options": {"seed": 7}}
    assert call(socket, {"endpoint": "reset", "data": {"options": {"episode": 2}}}) == {
        "reset_options": {"episode": 2}
    }
    modalities = call(socket, {"endpoint": "get_modality_config"})
    assert modalities["action"]["__ModalityConfig__"] is True
    assert json.loads(modalities["action"]["as_json"])["delta_indices"] == list(range(16))
    assert call(socket, {"endpoint": "kill"}) is None
    assert not server.running


@pytest.mark.parametrize(
    "rpc_request,fragment",
    [
        ([], "mapping"),
        ({"endpoint": "unknown"}, "Unknown endpoint"),
        ({"endpoint": "get_action", "data": []}, "mapping"),
        ({"data": {"observation": {"bad": True}}}, "bad observation"),
    ],
)
def test_bad_request_error_reply_preserves_rep_state(running_server, rpc_request, fragment):
    _, socket = running_server
    assert fragment in call(socket, rpc_request)["error"]
    assert call(socket, {"endpoint": "ping"})["status"] == "ok"


def test_malformed_message_gets_error_then_ping_succeeds(running_server):
    _, socket = running_server
    socket.send(b"\xc1")  # Reserved msgpack byte.
    assert "error" in MsgSerializer().from_bytes(socket.recv())
    assert call(socket, {"endpoint": "ping"})["status"] == "ok"


def test_pickle_bearing_ndarrays_rejected_without_invoking_pickle():
    serializer = MsgSerializer()
    with pytest.raises(TypeError, match="Object arrays"):
        serializer.to_bytes(np.array([object()], dtype=object))
    for kind in (b"O", "O"):
        payload = msgpack.packb({b"nd": True, b"kind": kind, b"data": b"not pickle"})
        with pytest.raises(ValueError, match="Object ndarray"):
            serializer.from_bytes(payload)
    payload = io.BytesIO()
    np.save(payload, np.array([object()], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        serializer.from_bytes(
            msgpack.packb({"__ndarray_class__": True, "as_npy": payload.getvalue()})
        )


def test_numeric_legacy_ndarray_roundtrip():
    array = np.arange(5, dtype=np.int64)
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    np.testing.assert_array_equal(
        MsgSerializer().from_bytes(
            msgpack.packb({"__ndarray_class__": True, "as_npy": stream.getvalue()})
        ),
        array,
    )


def test_import_does_not_load_transport_or_model_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import humanwam.sonic_policy_server; "
            "assert not {'numpy','zmq','msgpack','msgpack_numpy','torch','gr00t'} & sys.modules.keys()",
        ],
        check=True,
        capture_output=True,
    )


def _native_classes():
    """Execute actual native class bodies, excluding their heavy package imports."""
    root = Path(__file__).resolve().parents[3]
    source = Path(os.environ.get("HUMANWAM_GR00T_SOURCE", root / "third_party/Isaac-GR00T"))
    if not (source / "gr00t/policy/server_client.py").is_file():
        pytest.skip("Pinned GR00T source is not checked out")
    namespace = {
        "dataclass": dataclass,
        "asdict": asdict,
        "is_dataclass": is_dataclass,
        "ABC": ABC,
        "abstractmethod": abstractmethod,
        "Enum": Enum,
        "Any": Any,
        "np": np,
        "io": io,
        "json": json,
        "functools": functools,
        "zmq": zmq,
        "mnp": mnp,
        "msgpack": msgpack,
    }
    for relative, names in (
        ("gr00t/data/types.py", {"ActionRepresentation", "ActionType", "ActionFormat", "ActionConfig", "ModalityConfig"}),
        ("gr00t/data/utils.py", {"to_json_serializable"}),
        ("gr00t/policy/policy.py", {"BasePolicy"}),
        ("gr00t/policy/server_client.py", {"MsgSerializer", "PolicyClient"}),
    ):
        path = source / relative
        tree = ast.parse(path.read_text())
        selected = [node for node in tree.body if getattr(node, "name", None) in names]
        assert len(selected) == len(names)
        exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def test_actual_pinned_gr00t_client_interoperates_over_loopback(running_server):
    native = _native_classes()
    server, _ = running_server
    host_port = server.endpoint.removeprefix("tcp://").split(":")
    with native["PolicyClient"](host=host_port[0], port=int(host_port[1])) as client:
        assert client.ping()
        state = np.arange(1024, dtype=np.float32).reshape(1, 16, 64)
        actions, info = client.get_action({"state": state}, options={"seed": 42})
        np.testing.assert_array_equal(actions["motion_token"], state)
        assert actions["left_hand_joints"].dtype == np.float32
        assert info["options"] == {"seed": 42}
        modalities = client.get_modality_config()
        assert isinstance(modalities["action"], native["ModalityConfig"])
        assert modalities["action"].delta_indices == list(range(16))
        assert client.reset({"episode": 3}) == {"reset_options": {"episode": 3}}
        with pytest.raises(RuntimeError, match="bad observation"):
            client.get_action({"bad": True})
        assert client.ping()


def test_native_client_to_real_fastwam_sonic_policy_contract(tmp_path):
    """Native client -> real REP server -> actual adapter -> injected model boundary."""
    from humanwam.backends.fastwam_sonic import (
        LANGUAGE_KEY,
        STATE_FIELDS,
        FastWAMSonicPolicy,
        checkpoint_manifest,
    )

    native = _native_classes()
    # Load the existing GR00T feedback76 recipe itself rather than duplicating
    # its modality schema as test expectations.
    root = Path(__file__).resolve().parents[3]
    recipe = ast.parse((root / "configs/humanwam/groot/feedback76_h16.py").read_text())
    assignment = next(
        node for node in recipe.body
        if isinstance(node, ast.Assign)
        and any(getattr(target, "id", None) == "feedback76_h16" for target in node.targets)
    )
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "feedback76_h16.py", "exec"), native)
    # Native GR00T policies may return dataclasses containing enum-valued
    # ActionConfig fields; the same wire server preserves those too.
    decoded_recipe = native["MsgSerializer"].from_bytes(
        MsgSerializer().to_bytes(_modality_envelopes(native["feedback76_h16"]))
    )
    assert asdict(decoded_recipe["action"]) == asdict(native["feedback76_h16"]["action"])
    stats = {}
    for group, dim in (("action", 76), ("state", 44)):
        stats[group] = {"default": {
            "global_min": [-1.0] * dim, "global_max": [1.0] * dim,
            "global_mean": [0.0] * dim, "global_std": [1.0] * dim,
        }}
    stats["action"]["default"]["stepwise_min"] = [[-1.0] * 76 for _ in range(16)]
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    checkpoint = tmp_path / "test-not-real-model.pt"
    checkpoint.write_bytes(b"explicitly injected test model")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(checkpoint_manifest(
        checkpoint, stats_path, action_frequency_hz=30.0,
    )))

    class Runtime:
        def __init__(self, config, profile):
            self.calls = []

        def infer(self, image, state, prompt, **options):
            self.calls.append((image, state, prompt, options))
            result = np.zeros((1, 16, 76), np.float32)
            result[:, :, :64] = np.arange(64, dtype=np.float32) / 100
            result[:, :, 64:70] = 0.25
            result[:, :, 70:] = 0.75
            return result

    policy = FastWAMSonicPolicy({
        "checkpoint": str(checkpoint), "stats_path": str(stats_path),
        "checkpoint_manifest": str(manifest_path),
    }, runtime_factory=Runtime)
    observation = {
        "video": {
            "ego_view": np.full((1, 1, 20, 30, 3), 25, np.uint8),
            "chest": np.full((1, 1, 30, 20, 3), 125, np.uint8),
        },
        "state": {key: np.full((1, 1, dim), index / 10, np.float32)
                  for index, (key, dim) in enumerate(STATE_FIELDS)},
        "language": {LANGUAGE_KEY: [["pick up the pillow"]]},
    }
    with _running_server(policy) as (server, _):
        host, port = server.endpoint.removeprefix("tcp://").split(":")
        with native["PolicyClient"](host=host, port=int(port)) as client:
            assert policy._runtime is None
            modalities = client.get_modality_config()
            for name, expected in native["feedback76_h16"].items():
                assert asdict(modalities[name]) == asdict(expected)
            assert policy._runtime is None
            actions, info = client.get_action(observation)
            assert set(actions) == {"motion_token", "left_hand_joints", "right_hand_joints"}
            assert actions["motion_token"].shape == (1, 16, 64)
            assert actions["left_hand_joints"].shape == (1, 16, 6)
            assert actions["right_hand_joints"].shape == (1, 16, 6)
            assert all(value.dtype == np.float32 for value in actions.values())
            np.testing.assert_allclose(actions["left_hand_joints"], 0.25)
            np.testing.assert_allclose(actions["right_hand_joints"], 0.75)
            np.testing.assert_allclose(actions["motion_token"][0, 0], np.arange(64) / 100)
            assert info["action_frequency_hz"] == 30.0
            assert info["normalized"] is False
            assert client.reset({"episode": 4}) == {}
            bad = {**observation, "state": {**observation["state"], "left_hand": np.zeros((1, 1, 7), np.float32)}}
            with pytest.raises(RuntimeError, match="state.left_hand"):
                client.get_action(bad)
            assert client.ping()
            client.get_action(observation)
    assert len(policy._runtime.calls) == 2
    image, state, prompt, options = policy._runtime.calls[0]
    assert len(image) == 2
    np.testing.assert_array_equal(image[0], 25)
    np.testing.assert_array_equal(image[1], 125)
    np.testing.assert_array_equal(state, np.concatenate([
        observation["state"][key][0, 0] for key, _ in STATE_FIELDS
    ]))
    assert prompt.endswith("instruction: pick up the pillow")
    assert options["action_horizon"] == 16
    assert options["sigma_shift"] == 1.0
