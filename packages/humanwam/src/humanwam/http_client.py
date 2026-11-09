"""Call a HumanWAM prediction HTTP service without altering observations.

The NPZ bytes and JSON metadata are forwarded as supplied. This client does
not timestamp observations, build action history, normalize values, load a
model, or communicate with robot actuators.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

MAX_MESSAGE_BYTES = 64 * 1024**2
MAX_ERROR_BYTES = 8192


def _reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant is not allowed: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _finite_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("JSON number exceeds the finite floating-point range")
    return result


def _json_loads(value):
    return json.loads(
        value,
        parse_constant=_reject_constant,
        parse_float=_finite_float,
        object_pairs_hook=_unique_object,
    )


def _http_error_message(error):
    raw = error.read(MAX_ERROR_BYTES)
    message = raw.decode("utf-8", errors="replace").strip()
    try:
        body = _json_loads(message)
        if isinstance(body, dict) and "error" in body:
            message = str(body["error"])
    except (ValueError, TypeError):
        pass
    detail = f": {message[:2000]}" if message else ""
    return f"HTTP {error.code} {error.reason}{detail}"


def predict_http(*, url, observation_path, metadata_path=None, timeout=180):
    """Return the service's direct prediction dict; never unwrap or retimestamp it."""
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.fragment:
        raise ValueError("--url must be an http(s) prediction endpoint without a fragment")
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("--timeout must be a positive finite number of seconds")
    metadata = {} if metadata_path is None else _json_loads(Path(metadata_path).read_text())
    if not isinstance(metadata, dict):
        raise ValueError("Metadata JSON must contain an object")
    raw = Path(observation_path).read_bytes()
    request_body = json.dumps(
        {"observation_npz": base64.b64encode(raw).decode("ascii"), "metadata": metadata},
        allow_nan=False,
    ).encode("utf-8")
    if not 0 < len(request_body) <= MAX_MESSAGE_BYTES:
        raise ValueError("Encoded prediction request must fit the server's 64 MiB limit")
    request = Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read(MAX_MESSAGE_BYTES + 1)
    except HTTPError as error:
        with error:
            message = _http_error_message(error)
        raise ValueError(message) from error
    except URLError as error:
        raise ValueError(f"Prediction request failed: {error.reason}") from error
    except HTTPException as error:
        raise ValueError(f"Invalid HTTP prediction response: {error}") from error
    if len(body) > MAX_MESSAGE_BYTES:
        raise ValueError("Prediction response exceeds 64 MiB")
    result = _json_loads(body)
    if not isinstance(result, dict):
        raise ValueError("Prediction response must be a JSON object")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Full HTTP prediction URL, including /predict")
    parser.add_argument(
        "--observation",
        type=Path,
        required=True,
        help="Actual observation NPZ; bytes are forwarded unchanged",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="JSON metadata; observation timestamps are preserved, never refreshed",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New prediction JSON file; existing paths are refused",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180,
        help="Positive request timeout in seconds (default: 180)",
    )
    args = parser.parse_args(argv)
    try:
        # Refuse existing files, directories and broken symlinks before HTTP.
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output already exists; choose a new JSON file")
        if not args.output.parent.is_dir():
            raise ValueError("Output parent directory must already exist")
        result = predict_http(
            url=args.url,
            observation_path=args.observation,
            metadata_path=args.metadata,
            timeout=args.timeout,
        )
        output_bytes = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
        created = False
        try:
            with args.output.open("xb") as stream:
                created = True
                stream.write(output_bytes)
        except OSError:
            if created:
                args.output.unlink(missing_ok=True)
            raise
        print(json.dumps({"ok": True, "output": str(args.output), "robot_actuation": False}))
        return 0
    except (ValueError, TypeError, OSError) as error:
        print(f"humanwam.http_client: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
