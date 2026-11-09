#!/usr/bin/env python3
"""Offline only: validate raw G1 references and export NPZ / SONIC v1 body bytes."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from humanwam.sonic_joint_reference import (
    InsufficientLookaheadError,
    build_joint_reference,
    serialize_pose_v1,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Unnormalized reference NPZ, allow_pickle=False")
    parser.add_argument("--output", type=Path, required=True, help="New output .npz; existing files are refused")
    parser.add_argument("--wire-output", type=Path, help="Optional body-only pose v1 .bin; never transmitted")
    args = parser.parse_args(argv)
    manifest = args.output.with_suffix(args.output.suffix + ".json")
    targets = [args.output, manifest] + ([args.wire_output] if args.wire_output else [])
    if (args.output.suffix != ".npz" or len({p.resolve() for p in targets}) != len(targets)
            or args.input.resolve() in {p.resolve() for p in targets}
            or any(p.exists() for p in targets)):
        parser.error("Use distinct new output paths with a .npz primary output; no overwrites")
    try:
        with np.load(args.input, allow_pickle=False) as source:
            fields = {name: source[name] for name in source.files}
        required = {
            "joint_positions", "joint_names", "joint_order", "root_quaternions_wxyz",
            "relative_timestamps", "inspire_hand_states", "normalized", "joint_units",
            "action_semantics", "root_orientation_source", "frame_index_start",
            "body_target_source",
        }
        if not required.issubset(fields) or set(fields) - required - {"joint_velocities"}:
            raise ValueError("Input NPZ keys must match the documented explicit raw-reference schema")
        for key in ("joint_order", "normalized", "joint_units", "action_semantics", "root_orientation_source", "frame_index_start", "body_target_source"):
            fields[key] = fields[key].item()
        reference = build_joint_reference(**fields)
        packet = serialize_pose_v1(reference) if args.wire_output else None
    except (ValueError, TypeError, KeyError, OSError) as exc:
        report = {"ok": False, "error": str(exc), "transport_started": False}
        if isinstance(exc, InsufficientLookaheadError):
            report["coverage"] = exc.coverage
        print(json.dumps(report, ensure_ascii=False), file=sys.stderr)
        return 2
    # All checks and serialization finish before any output is created.
    created = []
    try:
        for target in targets:
            target.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as stream:
            created.append(args.output)
            np.savez_compressed(stream, **reference.npz_fields())
        with manifest.open("x") as stream:
            created.append(manifest)
            json.dump(reference.metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if args.wire_output:
            with args.wire_output.open("xb") as stream:
                created.append(args.wire_output)
                stream.write(packet)
    except OSError:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    print(json.dumps({"ok": True, "output": str(args.output), "manifest": str(manifest),
                      "wire_output": str(args.wire_output) if args.wire_output else None,
                      "frames": len(reference.frame_index), "transport_started": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
