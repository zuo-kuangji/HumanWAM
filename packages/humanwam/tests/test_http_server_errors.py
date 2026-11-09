"""Exercise the real HTTP handler with failing lazy policies on loopback only."""

import base64
import io
import json
import os
import socket
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pytest


@contextmanager
def failing_server(failure):
    # A subprocess runs the unchanged main/HTTPServer/handler. Only the two
    # policy/config construction functions are mocked; no model is imported.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    script = """
import sys
from unittest.mock import patch
from humanwam import http_server

class LazyFailurePolicy:
    def predict(self, observation):
        if sys.argv[1] == "missing_checkpoint":
            raise FileNotFoundError("Missing checkpoint: /models/unavailable.pt")
        raise ImportError("Missing ImageWAM dependency: flux2")

    def reset(self):
        return None

with patch.object(http_server, "load_config", return_value={"backend": "test_imagewam"}), \\
     patch.object(http_server, "build_policy", return_value=LazyFailurePolicy()):
    http_server.main(["--config", "unused.json", "--host", "127.0.0.1", "--port", sys.argv[2]])
"""
    source = Path(__file__).resolve().parents[1] / "src"
    env = {**os.environ, "PYTHONPATH": str(source) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    process = subprocess.Popen(
        [sys.executable, "-c", script, failure, str(port)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 8
        while True:
            if process.poll() is not None:
                _stdout, stderr = process.communicate()
                pytest.fail("HTTP fixture exited: " + stderr.decode(errors="replace"))
            try:
                with urlopen(base + "/health", timeout=0.2) as response:
                    assert response.status == 200
                break
            except URLError:
                if time.monotonic() >= deadline:
                    pytest.fail("HTTP fixture did not start")
                time.sleep(0.02)
        yield base
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)


@pytest.mark.parametrize(
    "failure,message",
    [
        ("missing_checkpoint", "Missing checkpoint: /models/unavailable.pt"),
        ("missing_dependency", "Missing ImageWAM dependency: flux2"),
    ],
)
def test_lazy_load_failure_is_json_and_server_remains_alive(failure, message):
    arrays = io.BytesIO()
    np.savez(arrays, current_state=np.zeros(44, dtype=np.float32))
    body = json.dumps(
        {
            "observation_npz": base64.b64encode(arrays.getvalue()).decode("ascii"),
            "metadata": {},
        }
    ).encode()
    with failing_server(failure) as base:
        request = Request(
            base + "/predict",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as caught:
            urlopen(request, timeout=2)
        with caught.value as response:
            assert response.code == 400
            assert response.headers["Content-Type"] == "application/json"
            assert json.load(response) == {"error": message}
        with urlopen(base + "/health", timeout=2) as response:
            assert response.status == 200
            assert json.load(response) == {
                "backend": "test_imagewam",
                "transport": "json-npz-http",
                "robot_actuation": False,
            }
