"""CPU check of the GR00T data path for one task view: loader shapes, v2 augmentation, processor patch import."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
os.environ["HB_PATCH_ONLY"] = "1"
import hb_launch  # noqa: E402,F401  (applies the processor patch)
import hb_aug_v2 as A  # noqa: E402
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader  # noqa: E402
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor  # noqa: E402

view = Path(sys.argv[1])
if view.name == "a18":
    import h2_body31_hand44_h16 as C  # noqa: E402
    cfg = C.h2_body31_hand44_h16
else:
    import g1_feedback76_h16 as C  # noqa: E402
    cfg = C.feedback76_h16
loader = LeRobotEpisodeLoader(view, cfg)
df = loader[0]
cols = list(df.columns)
row = df.iloc[20]
out = {"episodes": len(loader), "columns": cols[:40], "rows_ep0": len(df)}
state_dims = {c: int(np.asarray(row[c]).size) for c in cols if c.startswith("state.")}
action_dims = {c: int(np.asarray(row[c]).size) for c in cols if c.startswith("action.")}
out.update(state_dims=state_dims, action_dims=action_dims, state_total=sum(state_dims.values()),
           action_total=sum(action_dims.values()))
vids = [c for c in cols if c.startswith("video.")]
imgs = [np.asarray(row[c]) for c in vids]
out["video"] = {c: list(i.shape) for c, i in zip(vids, imgs)}
x = torch.stack([torch.from_numpy(i).permute(2, 0, 1).float().div(255) for i in imgs])[None]
rng = np.random.default_rng(0)
boosts = {}
for _ in range(60):
    y, p = A.augment(x, rng)
    boosts[p["boost"]] = boosts.get(p["boost"], 0) + 1
    assert y.shape == x.shape and torch.isfinite(y).all() and float(y.min()) >= 0 and float(y.max()) <= 1
out["aug_boost_counts_60"] = boosts
out["mean_abs_change_last"] = round(float((y - x).abs().mean()), 4)
q = np.zeros((1, 29))
dq = A.perturb_joints(q, rng)
out["proprio_max_abs_deg"] = round(float(np.degrees(np.abs(dq).max())), 3)
out["patched"] = Gr00tN1d7Processor._get_vlm_inputs.__name__ == "_get_vlm_inputs" and \
    Gr00tN1d7Processor.__call__.__module__ == "hb_launch"
print("CPU_VERIFY", json.dumps(out), flush=True)
