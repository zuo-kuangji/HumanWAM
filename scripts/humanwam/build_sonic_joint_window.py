#!/usr/bin/env python3
"""Export a clock-validated backend prediction to SONIC v1 body bytes and Inspire NPZ.

No encoder, socket, robot driver or model starts. --now and --control-epoch
must come from the same actual monotonic domain as the prediction timestamps.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from humanwam.sonic_joint_runtime import (
    JointPlanError,
    JointReferenceBuffer,
    TimedJointPlan,
)


def reject_constant(value):
    raise ValueError(f"Nonfinite JSON constant is not a valid prediction: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Complete humanwam predict JSON result",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory; existing output is refused",
    )
    parser.add_argument(
        "--now",
        type=float,
        required=True,
        help="Actual current monotonic time in the prediction clock domain",
    )
    parser.add_argument(
        "--control-epoch",
        type=float,
        required=True,
        help="50Hz frame0 epoch in that same monotonic domain",
    )
    parser.add_argument("--clock-id", default="local_monotonic")
    parser.add_argument("--jitter-budget", type=float, default=0.05)
    parser.add_argument("--allow-experimental-source", action="store_true")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("Output directory already exists; choose a new output")
    try:
        prediction = json.loads(args.input.read_text(), parse_constant=reject_constant)
        plan = TimedJointPlan.from_policy_prediction(
            prediction, allow_experimental_source=args.allow_experimental_source
        )
        buffer = JointReferenceBuffer(
            control_epoch_time_s=args.control_epoch,
            execution_horizon=plan.execution_horizon,
            action_interval_s=plan.action_interval_s,
            jitter_budget_s=args.jitter_budget,
            clock_id=args.clock_id,
        )
        buffer.submit(plan, now_s=args.now)
        window = buffer.window(now_s=args.now)
    except (ValueError, TypeError, KeyError, OSError) as error:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": str(error),
                    "diagnostics": error.diagnostics
                    if isinstance(error, JointPlanError)
                    else {},
                    "transport_started": False,
                },
                allow_nan=False,
            ),
            file=sys.stderr,
        )
        return 2
    # No output is created until the semantic, clock, coverage and wire checks pass.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    created = []
    try:
        body = args.output_dir / "body_pose_v1.bin"
        with body.open("xb") as stream:
            created.append(body)
            stream.write(window.body_packet)
        hands = args.output_dir / "inspire_reference50hz.npz"
        with hands.open("xb") as stream:
            created.append(hands)
            np.savez_compressed(
                stream,
                inspire_hand_states=window.inspire_hand_states,
                absolute_timestamps_s=window.absolute_timestamps_s,
                frame_index=window.reference.frame_index,
            )
        manifest = args.output_dir / "window_manifest.json"
        with manifest.open("x") as stream:
            created.append(manifest)
            json.dump(
                {
                    "plan_id": window.plan_id,
                    "diagnostics": window.diagnostics,
                    "body_reference": window.reference.metadata,
                    "body_file": body.name,
                    "inspire_file": hands.name,
                    "transport_started": False,
                    "controller_validated": False,
                    "robot_execution_validated": False,
                },
                stream,
                indent=2,
                allow_nan=False,
            )
            stream.write("\n")
    except OSError:
        for path in created:
            path.unlink(missing_ok=True)
        args.output_dir.rmdir()
        raise
    print(
        json.dumps(
            {
                "ok": True,
                "output_dir": str(args.output_dir),
                "plan_id": window.plan_id,
                "frames": len(window.absolute_timestamps_s),
                "execution_deadline_s": window.execution_deadline_s,
                "transport_started": False,
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
