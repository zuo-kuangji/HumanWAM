"""CPU check (no model): the FastWAM adapter's inputs built from the model-agnostic eval cache == the inputs our FastWAM
training dataset (RobotVideoDataset of the run's config, no HB augmentation) builds for the same frame.
  python test_fastwam_inputs.py <task> <run out dir> [n_queries]"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "adapters"))
import hb_eval_common as E  # noqa: E402
import ad_fastwam as A  # noqa: E402

task, out = sys.argv[1], Path(sys.argv[2])
nq = int(sys.argv[3]) if len(sys.argv) > 3 else 3
A.setup_env()
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import fastwam.runtime as native  # noqa: E402
from fastwam.utils import misc  # noqa: E402

cfg = OmegaConf.load(out / "config.yaml")
inp = A.Inputs(cfg, out / "dataset_stats.json")
misc.register_work_dir("/tmp/hbfw-inputs-check")
OmegaConf.update(cfg, "data.train.pretrained_norm_stats", str(out / "dataset_stats.json"), merge=False, force_add=True)
tr, _ = native.build_datasets(cfg.data)
ddir = Path(cfg.data.train.dataset_dirs[0])
lens = [json.loads(l)["length"] for l in (ddir / "meta/episodes.jsonl").read_text().splitlines() if l.strip()]
meta = E.load_meta(task, "train10")
split = "train10"
worst = {}
done = 0
for ep, z in E.iter_episodes(task, split, None):
    base = int(sum(lens[:ep]))
    for qi, t in enumerate(list(z["query_t"])[1:1 + nq], start=1):
        t = int(t)
        s = tr[base + t]
        obs = {"images": {c: z[f"img_{c}"][qi] for c in meta["cams"]}, "state": z["state"][t],
               "prompt": meta["prompts"][str(int(z["task_index"]))]}
        img = inp.image(obs["images"])[0]
        d_img = float((img - s["video"][:, 0].float()).abs().max())
        pr = inp.proprio(obs["state"])[0]
        d_pr = float((pr - s["proprio"][0].float()).abs().max())
        ctx, mask = inp.context(obs["prompt"])
        d_ctx = float((ctx.float() - s["context"].float()).abs().max())
        d_mask = int((mask != s["context_mask"].bool()).sum())
        gt = z["action"][t:t + 16]
        a = inp.denorm(s["action"][:16])
        d_act = float(np.abs(a[:len(gt)] - gt).max())
        row = dict(ep=ep, t=t, idx=base + t, img=d_img, proprio=d_pr, ctx=d_ctx, mask=d_mask, act_denorm_vs_cache=d_act,
                   prompt_same=s["prompt"] == inp.DEFAULT_PROMPT.format(task=obs["prompt"]))
        print("CHECK", json.dumps(row), flush=True)
        for k in ("img", "proprio", "ctx", "act_denorm_vs_cache"):
            worst[k] = max(worst.get(k, 0.0), row[k])
        done += 1
    if done >= 2 * nq:
        break
print("WORST", json.dumps(worst))
