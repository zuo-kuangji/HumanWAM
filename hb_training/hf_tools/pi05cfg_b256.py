"""Write the fully resolved openpi TrainConfig (train_config.json) for every batch-256 pi0.5 run that has an hb_plan.json.
Runs in the openpi env (after `source hb_pi05/hb_env.sh`); loop it, the autoupload daemon waits for the file.
Usage: python pi05cfg_b256.py $P
"""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1])
G = P / "policies-20261109"
OUT = P / "hf-repro-20261111" / "pi05"
TASKDIR = {"b6": "b6-pedal-bin", "a3": "a3-fridge-open", "a7": "a7-cabinet-close", "a18": "a18-pour-water",
           "trolley": "a10-trolley-push", "basket": "a20-basket-transfer", "a17": "a17-unscrew-bottle",
           "b7": "b7-black-clothes-hamper", "a6": "a6-door-pull", "c6": "c6-bathroom-door-fetch"}
SUB = {"a3": "b256", "a7": "b256", "b6": "b256", "a18": "b256", "trolley": "b256", "basket": "b256", "a17": "b256",
       "b7": "b256"}  # everything except the new tasks A6 / C6 goes to <task>/b256/
sys.path.insert(0, str(G / "hb_pi05"))
sys.path.insert(0, str(P / "hf-sidecar-tools-20261110"))
import hb_g1_config as C  # noqa: E402
from repro_conv import conv  # noqa: E402

for t, td in TASKDIR.items():
    runs = list((G / "pi05-runs/b256" / t / "run/ckpt").glob("*/run"))
    if len(runs) != 1 or not (runs[0] / "hb_plan.json").is_file():
        continue
    dst = OUT / (td + ("__" + SUB[t] if t in SUB else "")) / "train_config.json"
    if dst.is_file():
        continue
    run = runs[0]
    plan = json.loads((run / "hb_plan.json").read_text())
    cfg = C.make_train_config(t, exp_name=plan["exp_name"], batch_size=plan["global_batch"],
                              num_train_steps=plan["num_train_steps"], num_workers=8,
                              checkpoint_base_dir=str(run.parent.parent), resume=False,
                              pytorch_weight_path="default", log_interval=10)
    d = {"note": "Fully resolved openpi TrainConfig of this run, rebuilt with code/hb_pi05/hb_g1_config.py "
                 "make_train_config() from training_plan.json (same arguments as hb_train.py). Absolute paths "
                 "refer to the training NAS.", "training_plan": plan, "train_config": conv(cfg)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, indent=1) + "\n")
    tmp.replace(dst)
    print("CFG", t, dst, dst.stat().st_size, flush=True)
