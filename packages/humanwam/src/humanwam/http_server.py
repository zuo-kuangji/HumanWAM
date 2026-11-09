"""Serial offline-prediction HTTP service; does not execute robot actions."""

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from .cli import json_value
from .config import load_config
from .observations import observation_from_npz
from .registry import build_policy


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8090)
    args = p.parse_args(argv)
    config = load_config(args.config)
    policy = build_policy(config)

    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, body):
            payload = json.dumps(body, default=json_value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self.reply(
                200 if self.path == "/health" else 404,
                dict(backend=config["backend"], transport="json-npz-http", robot_actuation=False),
            )

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 64 * 1024**2:
                    raise ValueError("Request size must be1..64MiB")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/reset":
                    result = policy.reset()
                elif self.path == "/predict":
                    raw = base64.b64decode(payload["observation_npz"], validate=True)
                    result = policy.predict(
                        observation_from_npz(io.BytesIO(raw), payload.get("metadata"))
                    )
                else:
                    self.reply(404, dict(error="Unknown endpoint"))
                    return
                self.reply(200, result)
            except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                self.reply(400, dict(error=str(exc)))

    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
