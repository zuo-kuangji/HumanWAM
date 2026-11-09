"""Real loopback HTTP tests; no policy, model, controller or external network."""

import base64
import io
import json
import threading
from contextlib import contextmanager, redirect_stderr
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import pytest

from humanwam.http_client import main


@contextmanager
def endpoint(*, status=200, response=None, raw_response=None):
    calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            size = int(self.headers["Content-Length"])
            calls.append((self.path, json.loads(self.rfile.read(size))))
            body = raw_response if raw_response is not None else json.dumps(response).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/predict", calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def inputs(tmp_path):
    path = tmp_path / "observation.npz"
    np.savez(
        path,
        ego_rgb=np.zeros((3, 4, 3), dtype=np.uint8),
        current_state=np.arange(44, dtype=np.float32),
        past_actions=np.empty((0, 47), dtype=np.float32),
    )
    metadata = {
        "observation_time_s": 12345.125,
        "timestamp_clock": "monotonic",
        "at_episode_start": True,
        "seed": 42,
    }
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata))
    return path, meta_path, metadata


def arguments(url, source, metadata, output):
    return [
        "--url",
        url,
        "--observation",
        str(source),
        "--metadata",
        str(metadata),
        "--output",
        str(output),
        "--timeout",
        "2",
    ]


def test_preserves_metadata_npz_and_direct_response(tmp_path):
    source, metadata_path, metadata = inputs(tmp_path)
    output = tmp_path / "prediction.json"
    response = {
        "actions": [[0.1]],
        "observation_time_s": metadata["observation_time_s"],
        "result": {"keep_this_key": True},
        "normalized": False,
    }
    with endpoint(response=response) as (url, calls):
        assert main(arguments(url, source, metadata_path, output)) == 0
    assert len(calls) == 1
    assert calls[0][0] == "/predict"
    assert calls[0][1]["metadata"] == metadata
    assert base64.b64decode(calls[0][1]["observation_npz"]) == source.read_bytes()
    assert json.loads(output.read_text()) == response


def test_http_error_reports_message_and_creates_no_output(tmp_path):
    source, metadata, _ = inputs(tmp_path)
    output = tmp_path / "prediction.json"
    errors = io.StringIO()
    with endpoint(status=400, response={"error": "stale observation; keep capture time"}) as (
        url,
        calls,
    ):
        with redirect_stderr(errors):
            assert main(arguments(url, source, metadata, output)) == 2
    assert len(calls) == 1
    assert "HTTP 400" in errors.getvalue()
    assert "stale observation; keep capture time" in errors.getvalue()
    assert not output.exists()


def test_existing_output_refused_before_request(tmp_path):
    source, metadata, _ = inputs(tmp_path)
    output = tmp_path / "prediction.json"
    output.write_text("original contents")
    with endpoint(response={"actions": []}) as (url, calls):
        assert main(arguments(url, source, metadata, output)) == 2
    assert not calls
    assert output.read_text() == "original contents"


@pytest.mark.parametrize(
    "text",
    ['{"observation_time_s": NaN}', '{"x": Infinity}', '{"x": 1e999}', '{"x": 1, "x": 2}', "[1,2]"],
)
def test_invalid_metadata_never_sends_request(tmp_path, text):
    source, metadata, _ = inputs(tmp_path)
    metadata.write_text(text)
    output = tmp_path / "prediction.json"
    with endpoint(response={"actions": []}) as (url, calls):
        assert main(arguments(url, source, metadata, output)) == 2
    assert not calls
    assert not output.exists()


@pytest.mark.parametrize(
    "body", [b'{"actions": [NaN]}', b'{"x": Infinity}', b'{"x": 1e999}', b"null", b"not-json"]
)
def test_invalid_success_json_creates_no_output(tmp_path, body):
    source, metadata, _ = inputs(tmp_path)
    output = tmp_path / "prediction.json"
    with endpoint(raw_response=body) as (url, _calls):
        assert main(arguments(url, source, metadata, output)) == 2
    assert not output.exists()
