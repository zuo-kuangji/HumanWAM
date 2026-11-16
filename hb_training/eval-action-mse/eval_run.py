"""Run one open-loop action-MSE evaluation: one model checkpoint x one task x one cached split.

Usage (inside the model's own environment, see run_eval.sh):
  python eval_run.py --model {groot,pi05,psi0,act,deltaflow,dit4dit} --task b6 --split heldout --ckpt <path> --out <json>
         [--device cuda:0] [--steps N] [--max-episodes N] [--max-queries N] [--opt key=value ...]
"""
import argparse
import importlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "adapters"))
import hb_eval_common as E  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=None, help="override the model's native denoising steps")
    ap.add_argument("--max-episodes", type=int, default=None)
    ap.add_argument("--max-queries", type=int, default=None)
    ap.add_argument("--no-pred", action="store_true")
    ap.add_argument("--opt", action="append", default=[], help="adapter option key=value")
    a = ap.parse_args()
    opts = dict(o.split("=", 1) for o in a.opt)
    mod = importlib.import_module(f"ad_{a.model}")
    meta = E.load_meta(a.task, a.split)
    adapter = mod.Adapter(ckpt=a.ckpt, task=a.task, split=a.split, meta=meta, device=a.device, steps=a.steps, **opts)
    res = E.run(adapter, a.task, a.split, a.out, max_episodes=a.max_episodes, max_queries=a.max_queries,
                save_pred=not a.no_pred)
    res["ckpt"] = a.ckpt
    res["model"] = a.model
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("[eval] RESULT", json.dumps({k: res[k] for k in ("model", "task", "split", "queries", "nmse", "mse_raw",
                                                           "infer_s_per_query")}),
          json.dumps({g: round(v["nmse"], 4) for g, v in res["groups"].items()}), flush=True)


if __name__ == "__main__":
    main()
