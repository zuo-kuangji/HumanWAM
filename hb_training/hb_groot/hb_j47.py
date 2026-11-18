"""joint47 root orientation for GR00T (2026-11-18 user experiment).

The dataset stores world-frame rotation6D per frame (action.root_rot6d_world). Every sample's action chunk is
converted to rotation6D relative to the heading of its CURRENT frame (row 0 = source offset 0 = the observation
frame), exactly as HumanWAM joint47 v2: R_relative[k] = Rz(-yaw(current)) @ R_world[k], using HumanWAM's own
joint_reference_rotation.py. Applied before normalization (fixed [-1, 1] bounds). Imported by hb_launch.py when
HB_J47_ROOT_KEY is set.
"""
import importlib.util
import os
from pathlib import Path

import numpy as np
from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as P

_default = Path(__file__).resolve().parents[2] / "HumanWAM-joint/src/imagewam/joint_reference_rotation.py"
_spec = importlib.util.spec_from_file_location("hb_jrr", os.environ.get("HB_J47_ROTATION_PY", str(_default)))
jrr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(jrr)
KEY = os.environ["HB_J47_ROOT_KEY"]
IDENT = np.array([1.0, 0.0, 0.0, 0.0])


def to_current_heading(world6d):
    q = jrr.heading_relative_rotation6d_to_world_quaternion(np.asarray(world6d, dtype=np.float64), IDENT)
    return jrr.world_quaternion_to_heading_relative_rotation6d(q, q[0].astype(np.float64)).astype(np.float32)


_prev = P.Gr00tN1d7Processor.__call__


def _call(self, messages):
    acts = getattr(messages[0]["content"], "actions", None)
    if acts and KEY in acts:
        acts[KEY] = to_current_heading(acts[KEY])
    return _prev(self, messages)


P.Gr00tN1d7Processor.__call__ = _call
