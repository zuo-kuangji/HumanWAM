"""GR00T PolicyClient wire protocol without importing its model environment.

The pinned GR00T client uses ZeroMQ REQ/REP and msgpack-numpy, not HTTP or
SONIC's separate low-level pose publisher. Optional transport dependencies are
loaded only when a serializer/server is constructed.
"""

from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, is_dataclass
from enum import Enum


class MsgSerializer:
    """Compatible numeric ndarray envelopes; never deserialize pickle data."""

    def __init__(self):
        import msgpack
        import msgpack_numpy
        import numpy

        self.msgpack = msgpack
        self.mnp = msgpack_numpy
        self.np = numpy

    def _encode(self, value):
        if isinstance(value, self.np.ndarray) and value.dtype.hasobject:
            raise TypeError("Object arrays are not supported by the SONIC policy protocol")
        return self.mnp.encode(value)

    def _decode(self, value):
        marker = value.get("__ndarray_class__", value.get(b"__ndarray_class__"))
        if marker:
            payload = value.get("as_npy", value.get(b"as_npy"))
            if not isinstance(payload, bytes):
                raise ValueError("Legacy ndarray envelope requires as_npy bytes")
            return self.np.load(io.BytesIO(payload), allow_pickle=False)
        nd = value.get(b"nd", value.get("nd"))
        kind = value.get(b"kind", value.get("kind"))
        if nd and kind in (b"O", "O"):
            raise ValueError("Object ndarray envelopes are forbidden")
        # Structured dtypes containing objects are also unsuitable for this
        # numeric protocol, even when kind is 'V' rather than 'O'.
        if nd and b"type" in value:
            dtype_spec = value[b"type"]
            if kind in (b"V", "V"):
                dtype_spec = [tuple(x) for x in dtype_spec]
            if self.np.dtype(dtype_spec).hasobject:
                raise ValueError("Object ndarray envelopes are forbidden")
        return self.mnp.decode(value)

    def to_bytes(self, value):
        return self.msgpack.packb(value, default=self._encode)

    def from_bytes(self, value):
        return self.msgpack.unpackb(value, object_hook=self._decode, raw=False)


def _json_enum(value):
    if isinstance(value, Enum):
        return value.name
    raise TypeError(f"Unsupported modality metadata type: {type(value).__name__}")


def _modality_envelopes(modalities):
    """Let the native client reconstruct its own ModalityConfig dataclasses."""
    result = {}
    for key, value in modalities.items():
        if is_dataclass(value) and not isinstance(value, type):
            value = asdict(value)
        if not isinstance(value, dict):
            raise TypeError("Modality definitions must be mappings or dataclasses")
        # Use a JSON payload, as accepted by both native marker spellings.
        result[key] = {"__ModalityConfig__": True, "as_json": json.dumps(value, default=_json_enum)}
    return result


class PolicyServer:
    """Serve get_action/reset/modality/ping using the native policy RPC shape."""

    def __init__(self, policy, host="127.0.0.1", port=5555, *, api_token=None):
        import zmq

        self.zmq = zmq
        self.serializer = MsgSerializer()
        self.policy = policy
        self.api_token = api_token
        self.running = True
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.MAXMSGSIZE, 64 * 1024 * 1024)
        try:
            if port == 0:
                self.socket.bind_to_random_port(f"tcp://{host}")
            else:
                self.socket.bind(f"tcp://{host}:{int(port)}")
        except Exception:
            self.close()
            raise
        self.endpoint = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)

    def _dispatch(self, request):
        if not isinstance(request, dict):
            raise ValueError("Policy request must be a mapping")
        if self.api_token is not None and request.get("api_token") != self.api_token:
            raise ValueError("Unauthorized: Invalid API token")
        endpoint = request.get("endpoint", "get_action")
        if endpoint == "ping":
            return {"status": "ok", "message": "Server is running"}
        if endpoint == "kill":
            self.running = False
            return None
        if endpoint == "get_modality_config":
            return _modality_envelopes(self.policy.get_modality_config())
        data = request.get("data", {})
        if not isinstance(data, dict):
            raise ValueError("Policy request data must be a mapping")
        if endpoint == "get_action":
            return self.policy.get_action(**data)
        if endpoint == "reset":
            return self.policy.reset(**data)
        raise ValueError(f"Unknown endpoint: {endpoint}")

    def run(self):
        """Run until kill, interrupt or close; malformed requests get an error reply."""
        while self.running:
            if not self.socket.poll(100, self.zmq.POLLIN):
                continue
            message = self.socket.recv()
            try:
                result = self._dispatch(self.serializer.from_bytes(message))
                response = self.serializer.to_bytes(result)
            except Exception as exc:
                response = self.serializer.to_bytes({"error": str(exc)})
            self.socket.send(response)

    def close(self):
        self.running = False
        socket = getattr(self, "socket", None)
        if socket is not None:
            socket.close(linger=1000)
            self.socket = None
        context = getattr(self, "context", None)
        if context is not None:
            context.term()
            self.context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    args = parser.parse_args(argv)
    from .config import load_config
    from .registry import build_policy

    policy = build_policy(load_config(args.config))
    with PolicyServer(policy, args.host, args.port) as server:
        print(f"SONIC policy RPC ready at {server.endpoint}", flush=True)
        try:
            server.run()
        except KeyboardInterrupt:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
