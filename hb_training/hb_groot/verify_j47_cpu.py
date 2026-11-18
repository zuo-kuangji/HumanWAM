"""CPU check of the GR00T joint47 data path (2026-11-18): loader dims (state44, action47), the stride-2 chunk, the
current-heading root rotation (hb_j47 vs HumanWAM computed directly from the recorded quaternions), the processor
patches and the v2 augmentation. Usage: <groot python> verify_j47_cpu.py <groot-data/b6v5j47>"""
import json
import os
import sys
from pathlib import Path

import numpy as np

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
os.environ["HB_PATCH_ONLY"] = "1"
os.environ.setdefault("HB_J47_ROOT_KEY", "root_rot6d")
import hb_launch  # noqa: E402,F401  (applies the processor patches, incl. hb_j47)
import hb_j47  # noqa: E402
import hb_aug_v2 as A  # noqa: E402
import g1_joint47_h40 as C  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

view = Path(sys.argv[1])
cfg = C.g1_joint47_h40
loader = LeRobotEpisodeLoader(view, cfg)
out = {"episodes": len(loader)}
ep = 3
df = loader[ep]
cols = list(df.columns)
row = df.iloc[20]
state = {c: int(np.asarray(row[c]).size) for c in cols if c.startswith("state.")}
action = {c: int(np.asarray(row[c]).size) for c in cols if c.startswith("action.")}
out.update(state_dims=state, action_dims=action, state_total=sum(state.values()), action_total=sum(action.values()))
assert out["state_total"] == 44 and out["action_total"] == 47, out
info = json.loads((view / "meta/info.json").read_text())
assert not any("vel" in k for k in info["features"] if k.startswith("observation.state")), "velocity feature"
t = 100
offsets = cfg["action"].delta_indices
assert offsets == list(range(0, 80, 2))
world6 = np.stack([np.asarray(df["action.root_rot6d"].iloc[t + k]) for k in offsets])
rel = hb_j47.to_current_heading(world6)
tab = pq.read_table(view / "data/chunk-000/episode_000003.parquet").to_pandas()
q = np.stack(tab["observation.root_orientation"].to_numpy()).astype(np.float64)
direct = hb_j47.jrr.world_quaternion_to_heading_relative_rotation6d(q[[t + k for k in offsets]], q[t])
out["rot_max_abs_diff_vs_humanwam"] = float(np.abs(rel - direct).max())
out["rot_row0_heading_deg"] = float(np.degrees(np.arctan2(rel[0, 1], rel[0, 0])))
out["rot_range"] = [float(rel.min()), float(rel.max())]
assert out["rot_max_abs_diff_vs_humanwam"] < 1e-5 and abs(out["rot_row0_heading_deg"]) < 1e-3
body = np.stack([np.asarray(df["action.body29"].iloc[t + k]) for k in offsets])
names = info["features"]["observation.state"]["names"]
assert np.allclose(body[0], np.stack(tab["observation.state"].to_numpy())[t][names.index("left_hip_pitch_joint"):][:29])
stats = json.loads((view / "meta/stats.json").read_text())
out["rot_stats"] = {k: stats["action.root_rot6d_world"][k][:2] for k in ("min", "max")}
rng = np.random.default_rng(0)
dq = A.perturb_joints(np.zeros((1, 29)), rng)
out["proprio_max_abs_deg"] = round(float(np.degrees(np.abs(dq).max())), 3)
out["patched"] = Gr00tN1d7Processor.__call__.__module__ == "hb_j47" and Gr00tN1d7Processor._get_vlm_inputs.__name__ == "_get_vlm_inputs"
assert out["patched"], "processor patches missing"
print("J47_CPU_VERIFY_OK", json.dumps(out), flush=True)
