"""Humanoid Benchmark auto-upload daemon (user rule 2026-11-11): as soon as a training run finishes, upload its last two
saved checkpoints + everything needed to evaluate / deploy them + the simple training records (no resume state) to the
public model repo humanoid-benchmark/<model>/<task-dir>/epoch-XXX/, same layout as groot / pi05. Then verify every
uploaded file (LFS sha256 / git blob sha1) and refresh the repo README.

Jobs: $P/hf-autoupload/jobs.json  [{"model": "act|psi0|deltaflow|dit4dit", "task": "<task>"} ...] (re-read every loop)
State: $P/hf-autoupload/state.json ; log: $P/hf-autoupload/autoupload.log ; staging: $P/hf-autoupload/stage/
Also preserves the epoch-10 checkpoint of running ACT / Psi-Zero jobs by hard link (trainer keeps only 2 checkpoints).
Run on master-0: setsid nohup python hb_autoupload.py $P > /dev/null 2>&1 &
"""
import df_metrics  # noqa: E402  (2026-11-12: Delta Flow loss tables)
import hb_modelcard  # noqa: E402  (2026-11-12: model cards rebuilt after every upload)
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi

P = Path(sys.argv[1])
G = P / "policies-20261109"
A = P / "hf-autoupload"
A.mkdir(parents=True, exist_ok=True)
ORG = "humanoid-benchmark"
_TOKEN = Path("/root/.hb-secrets/hf_token").read_text().strip()
api = HfApi(token=_TOKEN)
TOKB = _TOKEN.encode()
ONCE = "--once" in sys.argv

TASKDIR = {"b6": "b6-pedal-bin", "a3": "a3-fridge-open", "a7": "a7-cabinet-close", "a18": "a18-pour-water",
           "trolley": "a10-trolley-push", "basket": "a20-basket-transfer", "a17": "a17-unscrew-bottle", "b7": "b7-black-clothes-hamper"}
DESC = {"b6": "Pick up a chip barrel, step on the pedal bin, drop it in", "a3": "Open the fridge door",
        "a7": "Close the cabinet door", "a18": "Pour water from a pitcher into a glass", "trolley": "Push a trolley",
        "basket": "Transfer a basket", "a17": "Unscrew the cap of a bottle",
        "b7": "Take the black garment off the hanger and drop it into the hamper"}
TASKNAME = {"b6": "B6 pedal-bin trash disposal (G1)", "a3": "A3 open fridge door (G1)", "a7": "A7 close cabinet door (G1)",
            "a18": "A18 pour water (H2)", "trolley": "A10 trolley push (G1)", "basket": "A20 basket transfer (G1)",
            "a17": "A17 unscrew bottle cap (H2)", "b7": "B7 black clothes into the laundry hamper (G1)"}
HFDATA = {"b6": "b6 v5t7 ready-start (first 100 episodes)", "a3": "a3-fridge-open-expert-v1",
          "a7": "a7-cabinet-close-expert-v1", "a18": "a18 H2 pour water", "trolley": "trolley-push-expert-v1 @ef9024a",
          "basket": "basket-transfer-expert-v1 @b0ff522 (current-dynamics-100-v21)",
          "a17": "a17-unscrew-bottle-expert-v1 @407e6e2",
          "b7": "b7-black-clothes-hamper-expert-v1 @1bb002a (first 100 of 107)"}
for _d, _v in ((TASKDIR, dict(a6="a6-door-pull", c6="c6-bathroom-door-fetch")),
               (DESC, dict(a6="Pull a door open", c6="Go through the bathroom door to another room and fetch an object")),
               (TASKNAME, dict(a6="A6 door pull (G1)", c6="C6 bathroom-door cross-room fetch (G1)")),
               (HFDATA, dict(a6="door-pull-expert-v1 @cf3ee9b", c6="c6-bathroom-door-fetch-expert-v1 @3afba69e")),
               (TASKDIR, dict(a14="a14-unscrew-bulb")),
               (DESC, dict(a14="Insert a bulb into the socket and screw it in until it lights up")),
               (TASKNAME, dict(a14="A14 screw in a bulb (H2, cameras head + chest + wrist)")),
               (HFDATA, dict(a14="a14-unscrew-bulb-expert-v1 @b4390fc")),
               (TASKDIR, dict(a1="a1-ground-object-pickup")),
               (DESC, dict(a1="Walk to a carton on the floor, squat, grasp and lift it with the right hand, stand and take two steps holding it")),
               (TASKNAME, dict(a1="A1 ground object pickup (G1)")),
               (HFDATA, dict(a1="a1-ground-object-pickup-expert-v1 @f47f07e")),
               (TASKDIR, dict(a10v2="a10-trolley-push-v2")),
               (DESC, dict(a10v2="Walk to the platform trolley, grasp the handle with both hands and push it alongside the box")),
               (TASKNAME, dict(a10v2="A10 trolley push v2 (G1, 150 episodes)")),
               (HFDATA, dict(a10v2="a10-trolley-push-expert-v2 @d3abb4c (all 150 episodes)")),
               (TASKDIR, dict(b7green="b7-green-clothes-hamper")),
               (DESC, dict(b7green="Walk around the side table to the clothes stand, take the green garment off the hanger with the right hand and drop it into the laundry hamper")),
               (TASKNAME, dict(b7green="B7 green clothes -> hamper (G1)")),
               (HFDATA, dict(b7green="b7-green-clothes-hamper-expert-v1 @ada288c"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
ROBOT = {t: ("H2 (Sharpa hands)" if t in ("a18", "a17", "a14") else "Unitree G1") for t in TASKDIR}
AUG = ("Humanoid Benchmark online augmentation spec v2 (2026-11-09): 15 visual effects, one effect per training sample "
       "at 1.5x, plus proprioception noise/delay/drop; model-native image augmentation off")
AUG_ALIGNED = ("Humanoid Benchmark online augmentation spec v2 (2026-11-09): 15 visual effects, one effect per training "
               "sample at 1.5x; proprioception aligned with the GR00T A20 recipe (2026-11-12): per-sample joint bias "
               "+-2 deg + Gaussian noise 0.25 deg (hb_aug_v2.perturb_joints), no proprio delay, no history drop; "
               "model-native image augmentation off")
OLD_PSI0 = ("b6", "a3", "a7", "a18", "trolley", "a17")


def aug_text(model, task):
    if "/" in repo_path(model, task) and model == "psi0":
        return AUG_ALIGNED
    if model == "act" or (model == "psi0" and task in OLD_PSI0):
        return AUG
    if model == "psi0" and task == "basket":
        return AUG_ALIGNED.replace("per-sample joint bias", "per-trajectory joint bias (run started before the "
                                   "per-sample patch)")
    return AUG_ALIGNED


RESUME = re.compile(r"(optimizer|scheduler|random_states|rng_state|_optim_states|^global_step|zero_to_fp32|^latest$)")
MODEL_TEXT = {
    "act": ("ACT — Humanoid Benchmark", "ACT (Action Chunking Transformer, ResNet18, VAE; Psi0 baselines/act)",
            "Global batch 696 = 87 per GPU x 8 GPUs, one node per run, 20 epochs, saved at epoch 10 and 20."),
    "psi0": ("Psi-Zero — Humanoid Benchmark", "Psi-Zero (Psi0 action expert, flow matching)",
             "4 nodes x 8 GPUs, 20 epochs, saved at epoch 10 and 20."),
    "deltaflow": ("Delta Flow — Humanoid Benchmark", "Delta Flow (ImageWAM / HumanWAM whole-body delta-flow policy)",
                  "4 nodes x 8 GPUs, 40 epochs; the last two saved checkpoints (epoch 30 and 40) are published."),
    "groot": ("GR00T N1.7 — Humanoid Benchmark", "GR00T N1.7 (Gr00tN1d7, action head fine-tuned, VLM frozen)",
              "Global batch 696 (2026-11-10 round) or 256 (from 2026-11-12; A3/A7 re-trains under <task>/b256/), 20 epochs, "
              "saved at epoch 10 and 20."),
    "pi05": ("pi0.5 — Humanoid Benchmark", "pi0.5 (openpi, PyTorch weights)",
             "Global batch 696 (2026-11-10 round), 512 (B7) or 256 (from 2026-11-12; A3/A7 re-trains under <task>/b256/), "
             "20 epochs, saved at epoch 10 and 20."),
    "dit4dit": ("DiT4DiT — Humanoid Benchmark", "DiT4DiT (Cosmos-Predict2.5-2B video DiT + action DiT)",
                "4 nodes x 8 GPUs, 20 epochs, saved at epoch 10 and the last step; native sin/cos state encoding "
                "inside the model (G1 73-D, H2 153-D) of the same raw 44 / 78-D benchmark state."),
}
SUBDIR = {("groot", "a3"): "b256", ("groot", "a7"): "b256",
          **{("pi05", t): "b256" for t in ("b6", "a3", "a7", "a18", "trolley", "basket", "a17", "b7")}}


# 2026-11-12: aligned Psi-Zero re-trains (job model "psi0a", global batch 512) go to humanoid-benchmark/psi0 under
# <task>/b512/ (user rule: HF sub-folders are always named after the global batch, b<N>; never "aligned")
SUBDIR.update({("psi0a", t): "b512" for t in ("b6", "a3", "a7", "a18", "trolley", "a17", "basket")})
# 2026-11-12: ACT back at global batch 256 (job model "act256"): earlier 696 tasks go to act/<task>/b256/, A6 / C6 to act/<task>/
SUBDIR.update({("act256", t): "b256" for t in ("b6", "a3", "a7", "a18", "trolley", "basket", "a17", "b7")})
# 2026-11-12 15:4x: GR00T 256 re-trains of the remaining 696 tasks -> groot/<task>/b256/
SUBDIR.update({("groot", t): "b256" for t in ("b6", "a17", "b7", "a18", "trolley", "basket")})
# 2026-11-12 19:5x: GR00T full fine-tuning, fixed 4000 steps (job model "grootft", runs in groot-runs-ft4k) -> repo groot,
# <task>/b256/step-002000|step-004000 for the eight earlier tasks, <task>/step-... for A6 / C6 / A14
SUBDIR.update({("grootft", t): "b256" for t in ("b6", "a3", "a7", "a17", "b7", "a18", "trolley", "basket")})
# 2026-11-13 19:4x user: GR00T global batch 512 full fine-tune (groot-runs-ft512) -> groot/<task>/b512/step-XXXXXX
SUBDIR.update({("grootft512", t): "b512" for t in ("b6", "a3", "a7", "a6", "a1", "a10v2", "a14", "a17", "a18", "b7",
                                                    "b7green", "basket", "c6", "trolley")})
REPO_OF = {"psi0a": "psi0", "act256": "act", "grootft": "groot", "grootft512": "groot"}


def repo_path(model, task):
    sub = SUBDIR.get((model, task))
    return TASKDIR[task] + ("/" + sub if sub else "")


MODEL_TEXT["psi0a"] = MODEL_TEXT["psi0"]
MODEL_TEXT["act256"] = MODEL_TEXT["act"]
MODEL_TEXT["grootft"] = MODEL_TEXT["groot"]
MODEL_TEXT["grootft512"] = MODEL_TEXT["groot"]
CODE_DIR = {"act": "hb_act", "psi0": "hb_psi0", "deltaflow": "hb_deltaflow", "dit4dit": "hb_dit4dit",
            "groot": "hb_groot", "pi05": "hb_pi05", "psi0a": "hb_psi0", "act256": "hb_act", "grootft": "hb_groot", "grootft512": "hb_groot"}


def log(*a):
    with open(A / "autoupload.log", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a) + "\n")


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if src.stat().st_size >= 50 * 1024 * 1024:
        os.link(src, dst)
    else:
        data = src.read_bytes()
        if TOKB in data or re.search(rb"hf_[A-Za-z0-9]{30,}", data):
            raise RuntimeError(f"token-like string in {src}")
        dst.write_bytes(data)


def put_tree(src, dst):
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        if f.is_file() and not any(RESUME.search(part) for part in rel.parts):
            put(f, dst / rel)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha1(p):
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def one(globbed):
    xs = sorted(globbed)
    return xs[0] if len(xs) == 1 else None


def rd(p):
    return json.loads(Path(p).read_text())


# ---------------------------------------------------------------- preserve epoch-10 checkpoints of running jobs
def preserve():
    for root in (sorted((G / "act-runs-b696").glob("*")) + sorted((G / "act-runs-b256").glob("*"))
                 + sorted((G / "psi0-runs").glob("*/hb1110")) + sorted((G / "psi0-runs").glob("*/b512"))
                 + sorted((G / "psi0-runs").glob("*/aligned"))):
        dec = root / "decision.json"
        if not dec.is_file():
            continue
        s = rd(dec)["save_steps"]
        for ck in root.glob(f"train/*/*/checkpoints/ckpt_{s}/model.safetensors"):
            dst = root / "preserved" / f"ckpt_{s}" / "model.safetensors"
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                os.link(ck, dst)
                log("preserved", dst)


# ---------------------------------------------------------------- collectors: return None until the run is complete
def collect_psi_like(model, task, root, done):
    if not done:
        return None
    dec = rd(root / "decision.json")
    run = one(c.parent for c in root.glob("train/*/*/checkpoints"))
    if run is None:
        return None
    eps = []
    for ep, step in ((10, dec["save_steps"]), (20, dec["steps"])):
        src = run / "checkpoints" / f"ckpt_{step}" / "model.safetensors"
        if not src.is_file():
            src = root / "preserved" / f"ckpt_{step}" / "model.safetensors"
        e = dict(epoch=ep, step=step, files={"model.safetensors": src} if src.is_file() else None)
        if dec.get("fixed_steps"):  # 2026-11-12: fixed-step runs -> step-XXXXXX folders (never overwrite epoch- ones)
            e.update(epoch=None, name=f"step-{step:06d}")
        eps.append(e)
    common = {n: run / n for n in ("run_config.json", "dataset_statistics.json", "clip_pooled_cache.pt", "argv.txt",
                                   "hb_init_report.json", "hb_metrics.jsonl", "envs.txt") if (run / n).is_file()}
    common["training_plan.json"] = root / "decision.json"
    for n, src in (("train.log", root / "logs" / "production.log"), ("train.log", root / "logs" / "node-0.log"),
                   ("train_args.txt", root / "train_args.txt")):
        if src.is_file() and n not in common:
            common[n] = src
    proto = dict(epochs=dec.get("epochs"), epochs_equivalent=dec.get("epochs_equivalent"), total_steps=dec["steps"],
                 save_steps=dec["save_steps"],
                 global_batch=dec["global_batch"], per_gpu_batch=dec["per_gpu_batch"],
                 num_gpus=dec["global_batch"] // dec["per_gpu_batch"], warmup_steps=dec.get("warmup_steps"),
                 frames=dec["frames"], state_dim=dec.get("state_dim"), action_dim=dec.get("action_dim"),
                 cameras=dec.get("image_keys"), normalization="dataset_statistics.json",
                 config="run_config.json (full training config); argv.txt (launch arguments)")
    if model == "act":
        proto.update(lr=1e-4, chunk=16, kl_weight=10.0, inference="Psi0 baselines/act policy (third_party/Psi0) with "
                     "the HB config in code/hb_act (hb_act_config.py, hb_act_patches.py); load model.safetensors with "
                     "run_config.json and dataset_statistics.json.")
    else:
        proto.update(inference="Psi0 code (third_party/Psi0) plus code/hb_psi0 (hb_psi0_config.py, hb_patches.py); "
                     "load model.safetensors with run_config.json, dataset_statistics.json, clip_pooled_cache.pt.")
    return dict(eps=eps, common=common, proto=proto)


def collect_act(task):
    root = G / "act-runs-b696" / task
    return collect_psi_like("act", task, root, (root / "production-complete.json").is_file())


def collect_act256(task):
    root = G / "act-runs-b256" / task
    return collect_psi_like("act", task, root, (root / "production-complete.json").is_file())


def collect_psi0(task):
    root = G / "psi0-runs" / task / "b512"
    if not root.is_dir():
        root = G / "psi0-runs" / task / "hb1110"
    ex = root / "logs" / "prod-node-0.exit"
    ok = ex.is_file() and ex.read_text().strip() == "0"
    return collect_psi_like("psi0", task, root, ok)


def collect_psi0a(task):
    root = G / "psi0-runs" / task / "aligned"
    ex = root / "logs" / "prod-node-0.exit"
    ok = ex.is_file() and ex.read_text().strip() == "0"
    return collect_psi_like("psi0", task, root, ok)


def collect_deltaflow(task):
    R = G / "deltaflow-runs" / task
    if not (R / "deltaflow-complete.json").is_file():
        return None
    u = int(rd(R / "data-verified.json")["updates_per_epoch"])
    W = R / "run" / "production" / "checkpoints" / "weights"
    eps = [dict(epoch=e, step=e * u, files={"model.pt": W / f"step_{e * u:06d}.pt"}) for e in (30, 40)]
    dv = rd(R / "data-verified.json")
    if dv.get("fixed_steps"):  # 2026-11-12: 8000-step runs -> step-006000 / step-008000 (never overwrite epoch- ones)
        eps = [dict(epoch=None, step=s, name=f"step-{s:06d}", files={"model.pt": W / f"step_{s:06d}.pt"})
               for s in dv["save_steps"]]
    for e in eps:
        if not e["files"]["model.pt"].is_file():
            raise RuntimeError(f"missing {e['files']['model.pt']}")
    common = {}
    cfg = R / "run" / "production" / "config.yaml"
    common["training_config.yaml"] = cfg if cfg.is_file() else R / "run" / "production.yaml"
    for f in sorted((R / "stats").glob("*")):
        if f.is_file():
            common[f"{f.name}"] = f
    for f in sorted((R / "qwen-text").glob("*")):
        if f.is_file():
            common[f"qwen-text/{f.name}"] = f
    for n in ("data-verified.json", "split-manifest.json", "deltaflow-complete.json", "prepared.ok"):
        if (R / n).is_file():
            common[n] = R / n
    # 2026-11-12: node-0 trainer logs of every attempt (supervised runs log under run/retry/s*/attempt-*/) plus the
    # loss table parsed from them, like the training records of the other models
    logs = sorted(R.glob("run/retry/**/node-0.log"), key=lambda q: q.stat().st_mtime)
    logs += [f for f in sorted((R / "run" / "node-logs").glob("*node-0*.log")) if f.is_file()]
    logs = [f for f in logs if f.stat().st_size < 50 * 1024 * 1024 and "step=" in f.read_text(errors="replace")]
    names = []
    for i, f in enumerate(logs):
        n = f"logs/{i:02d}-" + str(f.relative_to(R / "run")).replace("/", "__")
        common[n] = f
        names.append(n)
    if logs:
        mdir = R / "run" / "hb-metrics"
        df_metrics.write(mdir, df_metrics.parse(logs), names)
        common["metrics.csv"] = mdir / "metrics.csv"
        common["loss_summary.json"] = mdir / "loss_summary.json"
    proto = dict(epochs=None if dv.get("fixed_steps") else 40, updates_per_epoch=u,
                 total_steps=dv["max_steps"] if dv.get("fixed_steps") else 40 * u, num_gpus=32,
                 epochs_equivalent=dv.get("epochs_equivalent"),
                 normalization="dataset_stats.json / normalization.json in this folder",
                 config="training_config.yaml (production config; absolute paths refer to the training NAS)",
                 inference="HumanWAM ImageWAM delta-flow (code/hb_deltaflow installs the HB dataset and model "
                           "factories); load model.pt with training_config.yaml, the statistics file and the "
                           "qwen-text cache in this folder.")
    return dict(eps=eps, common=common, proto=proto)


def collect_dit4dit(task):
    D = G / "dit4dit-runs" / "chain-hb1110n" / task
    if not ((D / "DONE").is_file() and (D / "run" / "HB_DONE").exists()):
        return None
    run = D / "run"
    plan = rd(run / "hb_plan.json")
    # weights only: steps_<N>_pytorch_model.pt (never the steps_<N>_resume optimizer state)
    cks = [c for c in (run / "checkpoints").iterdir() if c.is_file() and re.fullmatch(r"steps_\d+_pytorch_model\.pt", c.name)]
    cks = sorted(cks, key=lambda c: int(re.findall(r"\d+", c.name)[-1]))[-2:]
    gb, frames = plan.get("global_batch"), plan.get("frames")
    eps = []
    for c in cks:
        step = int(re.findall(r"\d+", c.name)[-1])
        ep = round(step * gb / frames) if gb and frames else step
        e = dict(epoch=ep, step=step, files={c.name: c})
        if plan.get("fixed_steps"):  # 2026-11-12: 4000-step runs -> step-002000 / step-004000
            e.update(epoch=None, name=f"step-{step:06d}")
        eps.append(e)
    common = {n: run / n for n in ("dataset_statistics.json", "hb_full_config.yaml", "hb_plan.json", "hb_metrics.jsonl")
              if (run / n).is_file()}
    if (D / "decision.env").is_file():
        common["decision.env"] = D / "decision.env"
    proto = dict(plan={k: plan[k] for k in plan if isinstance(plan[k], (int, float, str, list))},
                 normalization="dataset_statistics.json", config="hb_full_config.yaml (full training config)",
                 inference="DiT4DiT (starVLA) with code/hb_dit4dit (hb_d4d_config.py, hb_d4d_data.py: raw 44 / 78-D "
                           "state -> native sin/cos 73 / 153-D); base model Cosmos-Predict2.5-2B from its own repo.")
    return dict(eps=eps, common=common, proto=proto)


def collect_groot(task):
    run = G / "groot-runs-b256" / task
    if not (run / "production-complete.json").is_file():
        return None
    return collect_groot_from(run, rd(run / "production-complete.json"))


def collect_groot_from(run, plan):
    task = run.name
    data = G / "groot-data" / task
    eps = []
    for ep, step in ((10, plan["save_steps"]), (20, plan["max_steps"])):
        src = run / "production" / f"checkpoint-{step}"
        files = {n: src / n for n in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
                                       "model.safetensors.index.json", "config.json", "embodiment_id.json",
                                       "processor_config.json", "statistics.json", "trainer_state.json",
                                       "training_args.bin", "wandb_config.json") if (src / n).is_file()}
        for f in sorted((src / "experiment_cfg").glob("*")):
            files["experiment_cfg/" + f.name] = f
        eps.append(dict(epoch=ep, step=step, files=files))
    common = {"dataset_meta/modality.json": data / "meta/modality.json",
              "dataset_meta/hb-dataset-receipt.json": data / "hb-dataset-receipt.json",
              "training_plan.json": run / "production-complete.json", "train.log": run / "logs/production.log"}
    for f in sorted((G / "hb_groot").glob("*.py")):
        common["code/hb_groot/" + f.name] = f
    proto = dict(epochs=plan.get("epochs"), total_steps=plan["max_steps"], save_steps=plan["save_steps"],
                 global_batch=plan["global_batch"], per_gpu_batch=plan["per_gpu_batch"], num_gpus=plan["world_size"],
                 lr=1e-4, lr_schedule="cosine, warmup_ratio 0.05", weight_decay=1e-5,
                 proprio_augmentation="per-sample joint bias +-2 deg + noise 0.25 deg (hb_aug_v2.perturb_joints); "
                                      "no proprio delay / history drop (the GR00T A20 recipe)",
                 config="experiment_cfg/conf.yaml", normalization="statistics.json",
                 inference="NVIDIA Isaac-GR00T N1.7 policy loader with embodiment_tag new_embodiment and "
                           "experiment_cfg/conf.yaml; VLM nvidia/Cosmos-Reason2-2B from its own repository.")
    return dict(eps=eps, common=common, proto=proto)


def collect_grootft(task):
    run = G / "groot-runs-ft4k" / task
    if not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, plan)
    for e in c["eps"]:
        e["name"] = f"step-{e['step']:06d}"
        e["epoch"] = None
    c["proto"].update(epochs=None, epochs_equivalent=plan.get("epochs_equivalent"), recipe=plan.get("recipe"),
                      gradient_accumulation_steps=plan.get("gradient_accumulation_steps"),
                      trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)")
    return c


def collect_grootft512(task):
    """Every saved checkpoint (2000, 2500, ..., 4000) -> step-XXXXXX; resume state is never uploaded."""
    run = G / "groot-runs-ft512" / task
    if not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, dict(plan, save_steps=plan["save_list"][0]))
    keep = set(c["eps"][-1]["files"])
    eps = []
    for step in plan["save_list"]:
        srcd = run / "production" / f"checkpoint-{step}"
        files = {k: srcd / k for k in keep if (srcd / k).is_file()}
        eps.append(dict(epoch=None, step=step, name=f"step-{step:06d}", files=files))
    c["eps"] = eps
    c["proto"].update(epochs=None, epochs_equivalent=plan.get("epochs_equivalent"), recipe=plan.get("recipe"),
                      save_steps=plan["save_list"], gradient_accumulation_steps=plan.get("gradient_accumulation_steps"),
                      trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)")
    return c


def collect_pi05(task):
    runs = list((G / "pi05-runs/b256" / task / "run/ckpt").glob("*/run"))
    if len(runs) != 1 or not (runs[0] / "HB_DONE").exists():
        return None
    run = runs[0]
    plan = rd(run / "hb_plan.json")
    eps = []
    for ep, step in zip((10, 20), plan["save_steps"]):
        src = run / str(step)
        files = {"model.safetensors": src / "model.safetensors", "metadata.pt": src / "metadata.pt"}
        for a in (src / "assets").rglob("*"):
            if a.is_file():
                files["assets/" + str(a.relative_to(src / "assets"))] = a
        e = dict(epoch=ep, step=step, files=files)
        if plan.get("fixed_steps"):  # 2026-11-12: 4000-step runs -> step-002000 / step-004000
            e.update(epoch=None, name=f"step-{step:06d}")
        eps.append(e)
    lg = run.parent.parent.parent / "logs"
    common = {"training_plan.json": run / "hb_plan.json",
              "train.log": next(x for x in (lg / "train.log", lg / "train-node-0.log") if x.is_file())}
    cfg = P / "hf-repro-20261111/pi05" / repo_path("pi05", task).replace("/", "__") / "train_config.json"
    if not cfg.is_file():
        return None  # written by pi05cfg_b256.py (openpi env) once hb_plan.json exists
    common["train_config.json"] = cfg
    for f in sorted((G / "hb_pi05").glob("*.py")):
        common["code/hb_pi05/" + f.name] = f
    proto = dict(epochs=plan.get("epochs"), epochs_equivalent=plan.get("epochs_equivalent"),
                 total_steps=plan["num_train_steps"], save_steps=plan["save_steps"],
                 global_batch=plan["global_batch"], per_gpu_batch=plan["per_gpu_batch"], num_gpus=plan["world_size"],
                 peak_lr=plan.get("peak_lr"), warmup_steps=plan.get("warmup_steps"), openpi_config=plan.get("config"),
                 proprio_augmentation="per-sample joint bias +-2 deg + noise 0.25 deg (hb_aug_v2.perturb_joints); "
                                      "no proprio delay (aligned with the GR00T A20 recipe)",
                 normalization="assets/<asset_id>/norm_stats.json",
                 inference="openpi policy_config.create_trained_policy(get_config(openpi_config), <this folder>) after "
                           "importing code/hb_pi05/hb_g1_config.py")
    return dict(eps=eps, common=common, proto=proto)


COLLECT = {"act": collect_act, "psi0": collect_psi0, "deltaflow": collect_deltaflow, "dit4dit": collect_dit4dit, "groot": collect_groot, "pi05": collect_pi05, "psi0a": collect_psi0a, "act256": collect_act256, "grootft": collect_grootft, "grootft512": collect_grootft512}


# ---------------------------------------------------------------- stage, upload, verify
def stage(model, task, c):
    td = repo_path(model, task)
    dst = A / "stage" / model / td.replace("/", "__")
    if dst.exists():
        shutil.rmtree(dst)
    missing = [e["epoch"] for e in c["eps"] if e.get("files") is None and e.get("tree") is None]
    for e in c["eps"]:
        if e.get("files") is None and e.get("tree") is None:
            continue
        ed = dst / (e.get("name") or f"epoch-{e['epoch']:03d}")
        if e.get("tree") is not None:
            put_tree(e["tree"], ed)
        for name, src in (e.get("files") or {}).items():
            put(src, ed / name)
        for name, src in c["common"].items():
            put(src, ed / name)
        proto = {"model": MODEL_TEXT[model][1], "task": TASKNAME[task], "task_dir": td, "robot": ROBOT[task],
                 "training_data": HFDATA[task], "episodes": EPISODES_OF.get(task, 100), "epoch_completed": e["epoch"], "global_step": e["step"],
                 "augmentation": aug_text(model, task), **c["proto"], "training_records_included": True, "resume_state_included": False}
        if missing:
            proto["note"] = (f"epoch {missing} checkpoint not available: the trainer kept only the last 2 checkpoints "
                             "and saved twice at the end of training, which rotated the epoch-10 save out.")
        proto["files"] = {str(p.relative_to(ed)): p.stat().st_size for p in sorted(ed.rglob("*")) if p.is_file()}
        (ed / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False, default=str) + "\n")
    return dst


def readme(model, tasks):
    title, name, how = MODEL_TEXT[model]
    rows = "\n".join(f"| `{TASKDIR[t]}/` | {DESC[t]} | {ROBOT[t]} |" for t in TASKDIR if t in tasks)
    legacy = ("\n`legacy/` holds earlier experiments that do NOT follow the benchmark protocol; see "
              "`legacy/README.md`.\n" if model == "deltaflow" else "")
    return f"""---
tags:
- robotics
- humanoid
- imitation-learning
---
# {title}

{name} checkpoints trained on Humanoid Benchmark tasks (100 demonstrations per task). {how}

| Folder | Task | Robot |
|---|---|---|
{rows}

Each task folder has one folder per published checkpoint (`epoch-XXX/`). Every epoch folder is self-contained for
evaluation and deployment (weights, configuration, normalization statistics, `protocol.json`) and also carries the
simple training records kept for reproducibility (training plan, logs, metrics). Optimizer / scheduler / RNG /
DeepSpeed resume state is not included. `code/` holds the adapter code used for training and inference.

Model inputs are identical across all benchmark models: G1 state 44-D (29 body joint angles + 6 + 6 measured hand
values + 3 projected gravity), H2 state 78-D (75 joint angles + 3 projected gravity), no joint velocities.
Training used the Humanoid Benchmark online data augmentation spec v2 with the model's own image augmentation off.
{legacy}"""


def upload(model, task, dst):
    repo = f"{ORG}/{REPO_OF.get(model, model)}"
    td = repo_path(model, task)
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    if api.model_info(repo).private:
        raise RuntimeError(f"{repo} is private")
    # 2026-11-12 user: never delete other checkpoint folders of the task (earlier rounds, b<N>/ variants, epoch- vs
    # step- naming); only the checkpoint folders uploaded now are replaced as a whole.
    own = [d.name for d in dst.iterdir() if d.is_dir()]
    c = api.upload_folder(repo_id=repo, folder_path=str(dst), path_in_repo=td, delete_patterns=[f"{n}/*" for n in own],
                          commit_message=f"{task}: last two checkpoints + eval/deploy files + training records")
    tree = {f.path: f for f in api.list_repo_tree(repo, recursive=True, expand=True) if hasattr(f, "size")}
    bad = []
    for p in sorted(dst.rglob("*")):
        if not p.is_file():
            continue
        k = f"{td}/{p.relative_to(dst)}"
        r = tree.get(k)
        ok = r is not None and r.size == p.stat().st_size and \
            ((r.lfs is not None and r.lfs.sha256 == sha256(p)) or (r.lfs is None and r.blob_id == git_sha1(p)))
        if not ok:
            bad.append(k)
    extra = [k for k in tree if any(k.startswith(f"{td}/{n}/") for n in own) and not (dst / k[len(td) + 1:]).is_file()]
    # README + code
    tasks = sorted({t for t in TASKDIR if any(k.startswith(TASKDIR[t] + "/epoch-") for k in tree)})
    if model in ("groot", "pi05", "psi0a", "act256", "grootft", "grootft512", "grootcontA", "grootcontB", "psi0cont", "groot2x32"):
        try:
            log(f"model card {repo}: {hb_modelcard.update(api, repo.split('/')[1], _TOKEN)}")
        except Exception as e:  # never block the checkpoint upload on the card
            log(f"model card {repo} FAILED {e!r}")
        info = api.model_info(repo)
        return dict(commit=c.oid, revision=info.sha, private=info.private,
                    files=sum(1 for p in dst.rglob("*") if p.is_file()), bad=bad, extra=extra, tasks_in_repo=tasks)
    side = A / "stage" / f"{model}-side"
    if side.exists():
        shutil.rmtree(side)
    side.mkdir(parents=True)
    (side / "README.md").write_text(hb_modelcard.build(api, repo.split("/")[1], _TOKEN))  # model card from the repo contents
    for f in sorted((G / CODE_DIR[model]).glob("*.py")):
        put(f, side / "code" / CODE_DIR[model] / f.name)
    api.upload_folder(repo_id=repo, folder_path=str(side), commit_message="Update README and adapter code")
    info = api.model_info(repo)
    return dict(commit=c.oid, revision=info.sha, private=info.private, files=sum(1 for p in dst.rglob("*") if p.is_file()),
                bad=bad, extra=extra, tasks_in_repo=tasks)


# 2026-11-14 user: GR00T / Psi-Zero basket continuation, 4000 more steps from the final checkpoints on
# humanoid-benchmark/a20-basket-transfer-pre-reach-trim-expert-v2 @c4df4d40 (pre-reach waits trimmed, 54,433 frames)
TRIMV2 = "humanoid-benchmark/a20-basket-transfer-pre-reach-trim-expert-v2 @ c4df4d40e51c439d7b41697d7c0d0184c844cb93"
SUBDIR.update({("grootcontA", "basket"): "b512-trimv2-from-b512", ("grootcontB", "basket"): "b512-trimv2-from-b256",
               ("psi0cont", "basket"): "b512-trimv2"})
REPO_OF.update(grootcontA="groot", grootcontB="groot", psi0cont="psi0")
MODEL_TEXT.update(grootcontA=MODEL_TEXT["groot"], grootcontB=MODEL_TEXT["groot"], psi0cont=MODEL_TEXT["psi0"])
CODE_DIR.update(grootcontA="hb_groot", grootcontB="hb_groot", psi0cont="hb_psi0")


def collect_grootcont(run_name, src_label):
    run = G / "groot-runs-cont" / run_name
    if not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, dict(plan, save_steps=plan["save_list"][0]))
    keep = set(c["eps"][-1]["files"])
    c["eps"] = [dict(epoch=None, step=st, name=f"step-{st:06d}",
                     files={k: run / "production" / f"checkpoint-{st}" / k for k in keep
                            if (run / "production" / f"checkpoint-{st}" / k).is_file()}) for st in plan["save_list"]]
    data = G / "groot-data" / "basketv2"
    c["common"].update({"dataset_meta/modality.json": data / "meta/modality.json",
                        "dataset_meta/hb-dataset-receipt.json": data / "hb-dataset-receipt.json"})
    c["proto"].update(epochs=None, epochs_equivalent=plan.get("epochs_equivalent"), recipe=plan.get("recipe"),
                      save_steps=plan["save_list"], gradient_accumulation_steps=plan.get("gradient_accumulation_steps"),
                      trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)",
                      training_data=TRIMV2, frames=plan.get("frames"),
                      continued_from=f"groot/<task>/{src_label}/step-004000 (trained 4000 steps on basket-transfer-expert-v1)",
                      continuation="weights only: fresh AdamW state, fresh warmup 5 % + cosine over 4000 steps; the "
                                   "source checkpoint's normalization statistics (statistics.json) are kept")
    return c


def collect_psi0cont(task):
    root = G / "psi0-runs" / "basketv2" / "cont-aligned"
    ex = root / "logs" / "prod-node-0.exit"
    c = collect_psi_like("psi0", "basket", root, ex.is_file() and ex.read_text().strip() == "0")
    if c is None:
        return None
    dec = rd(root / "decision.json")
    c["proto"].update(training_data=TRIMV2, frames=dec.get("frames"),
                      continued_from="psi0/<task>/b512/step-004000 (aligned run, 4000 steps on basket-transfer-expert-v1)",
                      continuation="weights only: fresh optimizer, fresh warmup + cosine over 4000 steps; normalization "
                                   "statistics of the source run (psi0-data/basket/meta/stats_psi0.json) are kept")
    return c


COLLECT.update(grootcontA=lambda t: collect_grootcont("from-b512", "b512"),
               grootcontB=lambda t: collect_grootcont("from-b256", "b256"), psi0cont=collect_psi0cont)


# 2026-11-14 16:xx user: continuation 2000 steps with the peak lr cut to 0.3x; b256 continuation dropped;
# GR00T B7 green v2 (HF b7-green-clothes-hamper @c12757e, 120 episodes) 2 nodes x 8 x 32, no accumulation.
GREENV2 = ("humanoid-benchmark/b7-green-clothes-hamper @ c12757e726a7005eafa50c093b6854200ccfe681 "
           "(v2 at the repo root: 120 episodes, 106,041 frames)")
SUBDIR.update({("groot2x32", "b7green"): "b512-datav2"})
REPO_OF.update(groot2x32="groot")
MODEL_TEXT.update(groot2x32=MODEL_TEXT["groot"])
CODE_DIR.update(groot2x32="hb_groot")
_collect_grootcont_v1 = collect_grootcont


def collect_grootcont(run_name, src_label):
    c = _collect_grootcont_v1(run_name, src_label)
    if c is None:
        return None
    plan = rd(G / "groot-runs-cont" / run_name / "production-complete.json")
    c["proto"].update(recipe=plan.get("recipe", "").replace("source normalization statistics kept", "normalization statistics recomputed from the v2 data"),
                      lr=plan.get("learning_rate"), lr_schedule="linear warmup 5 % (100 steps) to the peak, cosine to 0",
                      total_steps=plan["max_steps"], num_gpus=plan["world_size"], num_nodes=plan.get("num_nodes"),
                      continuation="weights only: fresh AdamW state; warmup 5 % to a reduced peak lr 3e-5 (0.3 x the "
                                   "original 1e-4), cosine to 0 over 2000 steps; normalization statistics recomputed from the "
                                   "trimmed v2 data (GR00T default, override_pretraining_statistics=True; std within ~1 % of v1)")
    return c


_collect_psi0cont_v1 = collect_psi0cont


def collect_psi0cont(task):
    c = _collect_psi0cont_v1(task)
    if c is None:
        return None
    dec = rd(G / "psi0-runs" / "basketv2" / "cont-aligned" / "decision.json")
    c["proto"].update(learning_rates=dec.get("learning_rates"),
                      continuation="weights only: fresh optimizer; warmup 100 steps to a reduced peak = 0.3 x every original "
                                   "learning rate (action header / mm projector 3e-5, vision tower 3e-6, language backbone "
                                   "3e-7), cosine to 0 over 2000 steps; normalization statistics of the trimmed v2 "
                                   "data (psi0-data/basketv2/meta/stats_psi0.json), as for the GR00T continuation")
    return c


def collect_groot2x32(task):
    run = G / "groot-runs-2x32" / "b7greenv2"
    if not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, dict(plan, save_steps=plan["save_list"][0]))
    keep = set(c["eps"][-1]["files"])
    c["eps"] = [dict(epoch=None, step=st, name=f"step-{st:06d}",
                     files={k: run / "production" / f"checkpoint-{st}" / k for k in keep
                            if (run / "production" / f"checkpoint-{st}" / k).is_file()}) for st in plan["save_list"]]
    c["proto"].update(epochs=None, epochs_equivalent=plan.get("epochs_equivalent"), recipe=plan.get("recipe"),
                      save_steps=plan["save_list"], gradient_accumulation_steps=plan.get("gradient_accumulation_steps"),
                      num_nodes=plan.get("num_nodes"), training_data=GREENV2, frames=plan.get("frames"),
                      trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)")
    return c


COLLECT.update(grootcontA=lambda t: collect_grootcont("from-b512", "b512"), psi0cont=collect_psi0cont,
               groot2x32=collect_groot2x32)
COLLECT.pop("grootcontB", None)


# 2026-11-14 17:xx user: A7 cabinet close v2 + B8 table wipe (GR00T 2x32 pair, Psi-Zero, Delta Flow)
for _d, _v in ((TASKDIR, dict(a7v2="a7-cabinet-close-v2", b8="b8-table-wipe")),
               (DESC, dict(a7v2="Close the open cabinet door with the right hand, then withdraw",
                           b8="Wipe the tabletop with a flat block held in the right hand, then set the block down")),
               (TASKNAME, dict(a7v2="A7 close cabinet door v2 (G1)", b8="B8 table wipe (G1)")),
               (HFDATA, dict(a7v2="a7-cabinet-close-expert-v2 @aecf070", b8="b8-table-wipe-expert-v1 @3b2e618")),
               (ROBOT, dict(a7v2="Unitree G1", b8="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "a7v2"): "b512", ("groot2x32", "b8"): "b512"})
_collect_groot2x32_green = collect_groot2x32


def collect_groot2x32_task(task):
    if task == "b7green":
        return _collect_groot2x32_green(task)
    run = G / "groot-runs-2x32" / task
    if not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, dict(plan, save_steps=plan["save_list"][0]))
    keep = set(c["eps"][-1]["files"])
    c["eps"] = [dict(epoch=None, step=st, name=f"step-{st:06d}",
                     files={k: run / "production" / f"checkpoint-{st}" / k for k in keep
                            if (run / "production" / f"checkpoint-{st}" / k).is_file()}) for st in plan["save_list"]]
    c["proto"].update(epochs=None, epochs_equivalent=plan.get("epochs_equivalent"), recipe=plan.get("recipe"),
                      save_steps=plan["save_list"], gradient_accumulation_steps=plan.get("gradient_accumulation_steps"),
                      num_nodes=plan.get("num_nodes"), training_data=HFDATA.get(task), frames=plan.get("frames"),
                      trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)")
    return c


COLLECT.update(groot2x32=collect_groot2x32_task)


# 2026-11-14 17:4x user: Delta Flow basket on the trimmed v2 data (deltaflow-runs/basketv2) -> deltaflow/<basket>/trimv2/
for _d, _v in ((TASKDIR, dict(basketv2=TASKDIR["basket"])), (DESC, dict(basketv2=DESC["basket"] + " (pre-reach waits trimmed)")),
               (TASKNAME, dict(basketv2=TASKNAME["basket"] + ", data v2 trimmed")),
               (HFDATA, dict(basketv2="a20-basket-transfer-pre-reach-trim-expert-v2 @c4df4d40")), (ROBOT, dict(basketv2="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("deltaflow", "basketv2"): "trimv2"})


# 2026-11-14 18:2x user: A18 / A17 2-camera re-train (head + chest) -> same task folders as A18 / A17, new subfolders
for _d, _v in ((TASKDIR, dict(a18c2=TASKDIR["a18"], a17c2=TASKDIR["a17"], a14c2=TASKDIR["a14"])),
               (DESC, dict(a18c2=DESC["a18"] + " (cameras head + chest)", a17c2=DESC["a17"] + " (cameras head + chest)", a14c2=DESC["a14"] + " (cameras head + chest)")),
               (TASKNAME, dict(a18c2=TASKNAME["a18"] + ", 2 cameras", a17c2=TASKNAME["a17"] + ", 2 cameras", a14c2="A14 screw in a bulb (H2, cameras head + chest)")),
               (HFDATA, dict(a18c2=HFDATA["a18"], a17c2=HFDATA["a17"], a14c2=HFDATA["a14"])),
               (ROBOT, dict(a18c2="H2 (Sharpa hands)", a17c2="H2 (Sharpa hands)", a14c2="H2 (Sharpa hands)"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "a18c2"): "b512", ("groot2x32", "a17c2"): "b512",
               ("psi0", "a18c2"): "b512-2cam", ("psi0", "a17c2"): "b512-2cam"})
# 2026-11-14 21:xx user: B1 rice ball into the microwave (G1), all 115 episodes
for _d, _v in ((TASKDIR, dict(b1="b1-riceball-microwave")),
               (DESC, dict(b1="Put the rice ball into the pre-opened microwave, release it, withdraw and close the door")),
               (TASKNAME, dict(b1="B1 rice ball into microwave (G1)")),
               (HFDATA, dict(b1="b1-riceball-microwave-expert-v1 @36da298")),
               (ROBOT, dict(b1="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "b1"): "b512"})

# 2026-11-14 22:xx: H2 training guides (hb_training/docs/h2): GR00T a17 / a18 -> b512-2cam, steps 2000 + 4000 only
EPISODES_OF = {"b1": 115, "b7greenv2": 120}
H2DATA = {"a18c2": "humanoid-benchmark/a18-pour-water-expert-v1 @ 8e956a102c84f991d2745a8cc66155c9991b742d",
          "a17c2": "humanoid-benchmark/a17-unscrew-bottle-expert-v1 @ 407e6e20db2debc9b77fc2813e3445f549d51a33"}
H2REF = {"a18c2": "groot/a18-pour-water/b256/step-004000", "a17c2": "groot/a17-unscrew-bottle/b256/step-004000",
         "a14v2": "groot/a14-unscrew-bulb/step-004000 (v1, 3 cameras, action75)"}
H2DATA["a14v2"] = "humanoid-benchmark/a14-unscrew-bulb-expert-v2 @ 8411a6dac45d0097d6068a60307de0c2e3b621a7"
H2ACTION = {"a14v2": "action82 = action.body_target (31) + action.hands (44: left 22, right 22) + action.planner (7: "
                     "planner_active, walk, move_cos, move_sin, face_cos, face_sin, speed_m_s); GR00T modalities "
                     "body_target / left_hand_target / right_hand_target / planner, ABSOLUTE, 16 rows per query"}
for _d, _v in ((TASKDIR, dict(a14v2=TASKDIR["a14"])), (DESC, dict(a14v2=DESC["a14"])),
               (TASKNAME, dict(a14v2="A14 screw in a bulb v2 (H2, cameras head + chest, action82 with the planner command)")),
               (HFDATA, dict(a14v2="a14-unscrew-bulb-expert-v2 @8411a6da")), (ROBOT, dict(a14v2="H2 (Sharpa hands)"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "a14v2"): "v2-b512", ("psi0", "a14v2"): "v2-b512"})
SUBDIR.update({("groot2x32", "a18c2"): "b512-2cam", ("groot2x32", "a17c2"): "b512-2cam"})
MODEL_TEXT["groot2x32"] = (MODEL_TEXT["groot"][0],
                           "GR00T N1.7 (Gr00tN1d7), full fine-tune: all modules (tune_llm, tune_visual, projector, "
                           "diffusion action head, VLLN)", MODEL_TEXT["groot"][2])
_collect_groot2x32_before_h2 = COLLECT["groot2x32"]


def collect_groot2x32_h2(task):
    c = _collect_groot2x32_before_h2(task)
    if c is None or task not in H2DATA:
        return c
    run = G / "groot-runs-2x32" / task
    c["eps"] = [e for e in c["eps"] if e["step"] in (2000, 4000)]
    if [e["step"] for e in c["eps"]] != [2000, 4000]:
        raise RuntimeError(f"groot2x32/{task}: checkpoints 2000 and 4000 not both present")
    hist = [h for h in rd(run / "production/checkpoint-4000/trainer_state.json")["log_history"] if "loss" in h]
    curve = run / "loss_curve.csv"
    curve.write_text("step,loss,learning_rate,grad_norm\n" + "".join(
        f"{h['step']},{h['loss']},{h.get('learning_rate', '')},{h.get('grad_norm', '')}\n" for h in hist))
    last = [h["loss"] for h in hist[-50:]]
    c["common"]["loss_curve.csv"] = curve
    c["proto"].update(
        cameras=["observation.images.head", "observation.images.chest"],
        training_data=H2DATA[task], episodes=100, split="all 100 episodes for training",
        state_layout="state78 = observation.state (75: legs 0-11, waist 12-14, left arm 15-21, right arm 22-28, "
                     "head 29-30, hands 31-74) + observation.projected_gravity (3); no joint velocity",
        action_layout=H2ACTION.get(task, "action75 = action.body_target (31) + action.hands (44: left 22, right 22); "
                      "GR00T modalities body_target / left_hand_target / right_hand_target, ABSOLUTE, 16 rows per query"),
        initialization="GR00T N1.7 base (groot-n17-base); VLM nvidia/Cosmos-Reason2-2B",
        recipe_reference=f"{H2REF[task]} recipe with cameras head + chest and global batch 512 (2 nodes x 8 GPUs x 32, "
                         "no gradient accumulation); everything else unchanged (H2 training guide 4.1)",
        published_steps=[2000, 4000], final_loss_mean_last50=round(sum(last) / len(last), 5),
        loss_curve="loss_curve.csv")
    return c


COLLECT.update(groot2x32=collect_groot2x32_h2)
# 2026-11-15 user: A10 trolley push v3 (G1), all 148 episodes, strict SONIC-only expert with pre-reach trim
for _d, _v in ((TASKDIR, dict(a10v3="a10-trolley-push-v3")),
               (DESC, dict(a10v3="Walk to the platform trolley, grasp the handle with both hands, push it alongside the box, release and withdraw")),
               (TASKNAME, dict(a10v3="A10 trolley push v3 (G1)")),
               (HFDATA, dict(a10v3="a10-trolley-push-expert-v3 @6e410c5 (all 148 episodes)")),
               (ROBOT, dict(a10v3="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "a10v3"): "b512"})
EPISODES_OF["a10v3"] = 148
# 2026-11-15 user: GR00T A18 / A17 variant: dropouts 0, action horizon 32, 16 flow sampling steps, layer-wise lr
REPO_OF.update(grooth32="groot")
CODE_DIR.update(grooth32="hb_groot")
MODEL_TEXT["grooth32"] = (MODEL_TEXT["groot"][0],
                          "GR00T N1.7 (Gr00tN1d7), full fine-tune: all modules; variant 2026-11-15: every dropout 0, "
                          "action horizon 32, 16 flow sampling steps, layer-wise lr (language 1e-6, vision 1e-5, "
                          "vision-language merger and action head 1e-4)", MODEL_TEXT["groot"][2])
SUBDIR.update({("grooth32", "a18c2"): "b512-2cam-h32", ("grooth32", "a17c2"): "b512-2cam-h32"})


def collect_grooth32(task):
    run = G / "groot-runs-h32" / task
    if not (run / "production-complete.json").is_file() or task not in H2DATA:
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, dict(plan, save_steps=plan["save_list"][0]))
    keep = set(c["eps"][-1]["files"])
    c["eps"] = [dict(epoch=None, step=st, name=f"step-{st:06d}",
                     files={k: run / "production" / f"checkpoint-{st}" / k for k in keep
                            if (run / "production" / f"checkpoint-{st}" / k).is_file()}) for st in (2000, 4000)]
    hist = [h for h in rd(run / "production/checkpoint-4000/trainer_state.json")["log_history"] if "loss" in h]
    curve = run / "loss_curve.csv"
    curve.write_text("step,loss,learning_rate,grad_norm\n" + "".join(
        f"{h['step']},{h['loss']},{h.get('learning_rate', '')},{h.get('grad_norm', '')}\n" for h in hist))
    last = [h["loss"] for h in hist[-50:]]
    c["common"]["loss_curve.csv"] = curve
    c["proto"].update(
        cameras=["observation.images.head", "observation.images.chest"], training_data=H2DATA[task], episodes=100,
        state_layout="state78 = observation.state (75) + observation.projected_gravity (3); no joint velocity",
        action_layout="action75 = action.body_target (31) + action.hands (44); 32 rows per chunk (horizon 32)",
        initialization="GR00T N1.7 base (groot-n17-base); VLM nvidia/Cosmos-Reason2-2B",
        trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)",
        recipe_reference=f"{H2REF[task]} recipe (head + chest, 2 x 8 x 32 = 512, no accumulation, 4000 steps) with: "
                         "state_dropout_prob 0, action-head attn_dropout 0, DiT dropout 0, action horizon 32, "
                         "num_inference_timesteps 16, layer-wise lr (user 2026-11-15)",
        model_overrides=plan.get("model_overrides"), lr_groups=plan.get("lr_groups"),
        launcher="code/hb_groot/hb_launch_v.py + groot_pair2x32_h2v.sh, modality h2_body31_hand44_h32.py",
        published_steps=[2000, 4000], final_loss_mean_last50=round(sum(last) / len(last), 5), loss_curve="loss_curve.csv")
    return c


COLLECT.update(grooth32=collect_grooth32)

# 2026-11-15: Delta Flow H2 cells of the H2 training guides on the legacy a18 e40 recipe
for _d, _v in ((TASKDIR, dict(a17lg=TASKDIR["a17"], a14lg=TASKDIR["a14"])),
               (DESC, dict(a17lg=DESC["a17"] + " (cameras head + chest)", a14lg=DESC["a14"] + " (cameras head + chest)")),
               (TASKNAME, dict(a17lg=TASKNAME["a17"] + ", 2 cameras, legacy a18 e40 recipe",
                               a14lg="A14 screw in a bulb v2 (H2, cameras head + chest, action82 with the planner command), legacy a18 e40 recipe")),
               (HFDATA, dict(a17lg=HFDATA["a17"], a14lg="a14-unscrew-bulb-expert-v2 @8411a6da")),
               (ROBOT, dict(a17lg="H2 (Sharpa hands)", a14lg="H2 (Sharpa hands)"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("deltaflow", "a17lg"): "b256-2cam", ("deltaflow", "a14lg"): "v2-b256"})
_collect_deltaflow_before_lg = COLLECT["deltaflow"]


def collect_deltaflow_lg(task):
    c = _collect_deltaflow_before_lg(task)
    if c is None or task not in ("a17lg", "a14lg"):
        return c
    R = G / "deltaflow-runs" / task
    dv = rd(R / "data-verified.json")
    c["common"]["normalization.json"] = R / "normalization.json"
    code = Path(dv["code"])
    for f in sorted((code / "src" / "imagewam").rglob("*.py")):
        c["common"]["code/imagewam/" + str(f.relative_to(code / "src" / "imagewam"))] = f
    c["common"]["code/scripts/wholebody/train_deltaflow.py"] = code / "scripts/wholebody/train_deltaflow.py"
    c["common"]["code/deltaflow-code.json"] = code / "deltaflow-code.json"
    c["proto"].update(
        recipe="legacy-a18-e40: configuration of deltaflow/legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/"
               "epoch-040 (H2PourDataset, b6-visual-aug-v2, no proprio augmentation, same model / losses / optimizer); "
               "changed: dataset, normalization refit on the task, the task's Qwen3 text cache, aug policy_id, and the "
               "user's Delta-flow schedule (global 256 = 32 x 8, no accumulation, 8000 steps, warmup 400, cosine to 0)",
        cameras=dv.get("cameras"), training_data=dv.get("dataset_source"), state_dim=dv.get("state_dim"),
        action_dim=dv.get("action_dim"), global_batch=dv.get("global_batch"), gradient_accumulation_steps=1,
        published_steps=dv.get("save_steps"),
        action_layout=("action82 = action.body_target (31) + action.hands (44) + action.planner (7)"
                       if dv.get("action_dim") == 82 else "action75 = action.body_target (31) + action.hands (44)"),
        inference_package="code/imagewam (model + loader of this run's code tree), normalization.json, qwen-text/, "
                          "training_config.yaml; FLUX.2-klein-base-4B, FLUX.2 AE and Qwen3-4B from their own repositories")
    return c


COLLECT.update(deltaflow=collect_deltaflow_lg)
# 2026-11-15 user: A3 fridge open v3 (125 episodes) and A7 cabinet close v3 (100 episodes), G1; hand action label =
# measured hands as every G1 task (the datasets' own v3 scripts use the teleop hand commands; the user chose measured)
for _d, _v in ((TASKDIR, dict(a3v3="a3-fridge-open-v3", a7v3="a7-cabinet-close-v3")),
               (DESC, dict(a3v3="Walk to the refrigerator, pull its upper door open with the right hand, release the handle and pull the hand back",
                           a7v3="Use the right hand to close the open cabinet door, then withdraw")),
               (TASKNAME, dict(a3v3="A3 open fridge door v3 (G1)", a7v3="A7 close cabinet door v3 (G1)")),
               (HFDATA, dict(a3v3="a3-fridge-open-expert-v3 @24882d5 (all 125 episodes)",
                             a7v3="a7-cabinet-close-expert-v3 @c2e15b0 (all 100 episodes)")),
               (ROBOT, dict(a3v3="Unitree G1", a7v3="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "a3v3"): "b512", ("groot2x32", "a7v3"): "b512"})
EPISODES_OF.update(a3v3=125, a7v3=100)
# 2026-11-15 17:xx user: B6 continuations on v6 (542 episodes) from the last B6 v5 (100 episodes) checkpoints, and B5
B6V6 = ("humanoid-benchmark/b6-pedal-bin @ ad2f5bf07f2eeef765b8b401b401a8e4dc133505 "
        "(v6 at the repo root: all 542 episodes, 331,577 frames)")
B5DATA = "humanoid-benchmark/b5-chair-push-expert-v1 @ b85b7e9f67cba0fe1fae8d65e73334bcd864b9a4 (all 100 episodes, 43,075 frames)"
for _d, _v in ((TASKDIR, dict(b5="b5-chair-push")),
               (DESC, dict(b5="Grasp the chair back with both hands, straighten it by stepping behind it, then push it under the table")),
               (TASKNAME, dict(b5="B5 chair push (G1)")),
               (HFDATA, dict(b5="b5-chair-push-expert-v1 @b85b7e9 (all 100 episodes)")),
               (ROBOT, dict(b5="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "b5"): "b512", ("grootcont6", "b6"): "b512-v6-cont-from-b256", ("psi0cont6", "b6"): "b512-v6-cont"})
EPISODES_OF.update(b5=100)
REPO_OF.update(grootcont6="groot", psi0cont6="psi0")
MODEL_TEXT.update(grootcont6=MODEL_TEXT["groot"], psi0cont6=MODEL_TEXT["psi0"])
CODE_DIR.update(grootcont6="hb_groot", psi0cont6="hb_psi0")


def collect_grootcont6(task):
    run = G / "groot-runs-cont" / "b6v6-from-ft4k"
    if task != "b6" or not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
    c = collect_groot_from(run, dict(plan, save_steps=plan["save_list"][0]))
    keep = set(c["eps"][-1]["files"])
    c["eps"] = [dict(epoch=None, step=st, name=f"step-{st:06d}",
                     files={k: run / "production" / f"checkpoint-{st}" / k for k in keep
                            if (run / "production" / f"checkpoint-{st}" / k).is_file()}) for st in plan["save_list"]]
    data = G / "groot-data" / "b6v6"
    c["common"].update({"dataset_meta/modality.json": data / "meta/modality.json",
                        "dataset_meta/hb-dataset-receipt.json": data / "hb-dataset-receipt.json"})
    hist = [h for h in rd(run / "production" / f"checkpoint-{plan['max_steps']}" / "trainer_state.json")["log_history"] if "loss" in h]
    curve = run / "loss_curve.csv"
    curve.write_text("step,loss,learning_rate,grad_norm\n" + "".join(
        f"{h['step']},{h['loss']},{h.get('learning_rate', '')},{h.get('grad_norm', '')}\n" for h in hist))
    c["common"]["loss_curve.csv"] = curve
    c["proto"].update(epochs=None, epochs_equivalent=plan.get("epochs_equivalent"), recipe=plan.get("recipe"),
                      save_steps=plan["save_list"], gradient_accumulation_steps=plan.get("gradient_accumulation_steps"),
                      num_nodes=plan.get("num_nodes"), num_gpus=plan.get("world_size"), total_steps=plan["max_steps"],
                      lr=plan.get("learning_rate"), lr_schedule="linear warmup 5 % (200 steps) to the peak, cosine to 0",
                      trainable="all modules (tune_llm, tune_visual, projector, diffusion action head, VLLN)",
                      training_data=B6V6, frames=plan.get("frames"), episodes=542,
                      continued_from="groot/b6-pedal-bin/b256/step-004000 (full fine-tune, 256 x 4000 steps on the first 100 "
                                     "B6 v5 episodes, 65,127 frames)",
                      continuation="weights only: fresh AdamW state; warmup 5 % to a reduced peak lr 3e-5 (0.3 x the original "
                                   "1e-4), cosine to 0 over 4000 steps (6.2 epochs of v6); global batch 512 = 2 x 8 x 32, no "
                                   "accumulation; normalization statistics recomputed from the v6 data (GR00T default, "
                                   "override_pretraining_statistics=True)", loss_curve="loss_curve.csv")
    return c


def collect_psi0cont6(task):
    root = G / "psi0-runs" / "b6v6" / "cont-aligned"
    ex = root / "logs" / "prod-node-0.exit"
    if task != "b6":
        return None
    c = collect_psi_like("psi0", "b6", root, ex.is_file() and ex.read_text().strip() == "0")
    if c is None:
        return None
    dec = rd(root / "decision.json")
    c["proto"].update(training_data=B6V6, frames=dec.get("frames"), episodes=542, learning_rates=dec.get("learning_rates"),
                      num_nodes=dec.get("num_nodes"),
                      continued_from="the final checkpoint (step 4000) of the Psi-Zero B6 v5 run psi0-runs/b6/aligned "
                                     "(4 x 8 x 16 = 512, 4000 steps on the first 100 B6 v5 episodes, 65,127 frames)",
                      continuation="weights only: fresh optimizer; warmup 200 steps (5 %) to a reduced peak = 0.3 x every "
                                   "original learning rate (action header / mm projector 3e-5, vision tower 3e-6, language "
                                   "backbone 3e-7), cosine to 0 over 4000 steps (6.2 epochs of v6); 2 x 8 x 32 = 512; "
                                   "normalization statistics of the v6 data (psi0-data/b6v6/meta/stats_psi0.json)")
    return c


COLLECT.update(grootcont6=collect_grootcont6, psi0cont6=collect_psi0cont6)
# 2026-11-15 17:xx user: Delta Flow B6 v6 continuation (weights only from deltaflow-runs/b6 step_010200) and B5
for _d, _v in ((TASKDIR, dict(b6v6=TASKDIR["b6"])), (DESC, dict(b6v6=DESC["b6"])),
               (TASKNAME, dict(b6v6="B6 pedal-bin trash disposal v6 (G1)")),
               (HFDATA, dict(b6v6="b6-pedal-bin @ad2f5bf0 (v6 at the repo root, all 542 episodes)")),
               (ROBOT, dict(b6v6="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("deltaflow", "b6v6"): "b256-v6-cont", ("deltaflow", "b5"): "b256"})
EPISODES_OF.update(b6v6=542)
_collect_deltaflow_before_b6v6 = COLLECT["deltaflow"]


def collect_deltaflow_b6v6(task):
    c = _collect_deltaflow_before_b6v6(task)
    if c is None or task != "b6v6":
        return c
    dv = rd(G / "deltaflow-runs" / task / "data-verified.json")
    c["proto"].update(training_data=dv.get("dataset_source"), episodes=542, frames=dv.get("frames"),
                      continued_from="deltaflow/b6-pedal-bin (B6 v5 run, first 100 episodes, 40 epochs = step 10200); "
                                     + str(dv.get("continued_from")),
                      continuation=dv.get("continuation"), lr=dv.get("learning_rate"))
    return c


COLLECT.update(deltaflow=collect_deltaflow_b6v6)
# 2026-11-15 18:4x user: A3 / A7 v4 replace v3 (action 76 = SONIC token + measured hands, the v4 layout)
for _d, _v in ((TASKDIR, dict(a3v4="a3-fridge-open-v4", a7v4="a7-cabinet-close-v4")),
               (DESC, dict(a3v4="Walk to the refrigerator, pull its upper door open with the right hand, release the handle and pull the hand back",
                           a7v4="Use the right hand to close the open cabinet door, then withdraw")),
               (TASKNAME, dict(a3v4="A3 open fridge door v4 (G1)", a7v4="A7 close cabinet door v4 (G1)")),
               (HFDATA, dict(a3v4="a3-fridge-open-expert-v4 @6393d4a (all 125 episodes)",
                             a7v4="a7-cabinet-close-expert-v4 @90cc2b5 (all 100 episodes)")),
               (ROBOT, dict(a3v4="Unitree G1", a7v4="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("groot2x32", "a3v4"): "b512", ("groot2x32", "a7v4"): "b512"})
EPISODES_OF.update(a3v4=125, a7v4=100)
# 2026-11-15 22:xx user: GR00T A18 / A17 h32 variant on the cropped v2 data (waits trimmed; same episodes / cameras)
for _d in (TASKDIR, DESC, TASKNAME, HFDATA, ROBOT, EPISODES_OF):
    for _n, _o in (("a18cr2", "a18c2"), ("a17cr2", "a17c2")):
        if _o in _d:
            _d.setdefault(_n, _d[_o])
HFDATA.update(a18cr2="a18-pour-water-expert-cropped-v2 @44cdd57 (100 episodes, waits trimmed)",
              a17cr2="a17-unscrew-bottle-expert-cropped-v2 @3520b70 (100 episodes, waits trimmed)")
H2DATA.update(a18cr2="humanoid-benchmark/a18-pour-water-expert-cropped-v2 @ 44cdd5715f4681038bd0bdf377e7705585a831c3 "
                     "(100 episodes, 66,850 frames; v1 @8e956a10 with the waits trimmed)",
              a17cr2="humanoid-benchmark/a17-unscrew-bottle-expert-cropped-v2 @ 3520b70f0cc30fec10c5d806167c791cbbfea718 "
                     "(100 episodes, 75,184 frames; v1 @407e6e20 with the waits trimmed)")
H2REF.update(a18cr2=H2REF["a18c2"], a17cr2=H2REF["a17c2"])
SUBDIR.update({("grooth32", "a18cr2"): "b512-2cam-h32-cropped-v2", ("grooth32", "a17cr2"): "b512-2cam-h32-cropped-v2"})
# 2026-11-15 23:xx user: Delta Flow on the newest data (A7 v4, A17 cropped v2 on the legacy a18 e40 recipe)
for _d, _v in ((TASKDIR, dict(a17cr2lg=TASKDIR["a17"])), (DESC, dict(a17cr2lg=DESC["a17"] + " (cameras head + chest)")),
               (TASKNAME, dict(a17cr2lg=TASKNAME["a17"] + ", 2 cameras, cropped v2 data, legacy a18 e40 recipe")),
               (HFDATA, dict(a17cr2lg="a17-unscrew-bottle-expert-cropped-v2 @3520b70 (100 episodes, waits trimmed)")),
               (ROBOT, dict(a17cr2lg="H2 (Sharpa hands)"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
SUBDIR.update({("deltaflow", "a7v4"): "b256", ("deltaflow", "a17cr2lg"): "b256-2cam-cropped-v2"})
EPISODES_OF.update(a17cr2lg=100)
_collect_deltaflow_before_al = COLLECT["deltaflow"]


def collect_deltaflow_al(task):
    c = _collect_deltaflow_before_al(task)
    if c is None or task != "a17cr2lg":
        return c
    R = G / "deltaflow-runs" / task
    dv = rd(R / "data-verified.json")
    c["common"]["normalization.json"] = R / "normalization.json"
    code = Path(dv["code"])
    for f in sorted((code / "src" / "imagewam").rglob("*.py")):
        c["common"]["code/imagewam/" + str(f.relative_to(code / "src" / "imagewam"))] = f
    c["common"]["code/scripts/wholebody/train_deltaflow.py"] = code / "scripts/wholebody/train_deltaflow.py"
    c["common"]["code/deltaflow-code.json"] = code / "deltaflow-code.json"
    c["proto"].update(
        recipe="legacy-a18-e40: configuration of deltaflow/legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/"
               "epoch-040 (H2PourDataset, b6-visual-aug-v2, no proprio augmentation, same model / losses / optimizer); "
               "changed: dataset (A17 cropped v2), normalization refit on the task, the task's Qwen3 text cache, aug "
               "policy_id, and the user's Delta-flow schedule (global 256 = 32 x 8, no accumulation, 8000 steps, warmup "
               "400, cosine to 0)",
        cameras=dv.get("cameras"), training_data=dv.get("dataset_source"), state_dim=dv.get("state_dim"),
        action_dim=dv.get("action_dim"), global_batch=dv.get("global_batch"), gradient_accumulation_steps=1,
        published_steps=dv.get("save_steps"), action_layout="action75 = action.body_target (31) + action.hands (44)",
        inference_package="code/imagewam (model + loader of this run's code tree), normalization.json, qwen-text/, "
                          "training_config.yaml; FLUX.2-klein-base-4B, FLUX.2 AE and Qwen3-4B from their own repositories")
    return c


COLLECT.update(deltaflow=collect_deltaflow_al)
# 2026-11-15 23:4x user: Delta Flow A3 v4 (all 125 episodes)
SUBDIR.update({("deltaflow", "a3v4"): "b256"})

# 2026-11-16 10:xx user: FastWAM gets its own public repo humanoid-benchmark/fastwam, same conventions as the other
# models (last two checkpoints step-002000 / step-004000 + eval/deploy files + simple training records, no resume
# state), uploaded automatically when each run of fastwam/runs/<task> writes complete.json.
FW = G / "fastwam"
FW_TASKS = ("a10v3", "a3v4", "a7v4", "b7", "b5", "b8", "a17cr2", "basketv2", "c6", "b6v6", "a14v2", "b1")
SUBDIR.update({("fastwam", t): "b512" for t in FW_TASKS})
MODEL_TEXT["fastwam"] = ("FastWAM — Humanoid Benchmark",
                         "FastWAM (Wan2.2-TI2V-5B video expert + 1.02 B action expert, mixture-of-transformers world "
                         "action model; upstream github.com/yuantianyuan01/FastWAM)",
                         "4 nodes x 8 GPUs, global batch 512 (16 per GPU, no gradient accumulation), 4000 steps, "
                         "saved at step 2000 and 4000.")
CODE_DIR["fastwam"] = "fastwam/tools"
FW_ACTION = {"g1": "action76 = action.motion_token (64) + measured Inspire hands (6 + 6), same-row labels",
             "h2": "action75 = action.body_target (31) + action.hands (44)",
             "h2p": "action82 = action.body_target (31) + action.hands (44) + action.planner (7)"}
FW_STATE = {"g1": "state44 = 29 body joint angles + measured Inspire hands 6 + 6 + projected gravity 3",
            "h2": "state78 = 75 joint angles + projected gravity 3", "h2p": "state78 = 75 joint angles + projected gravity 3"}


def _fw_metrics(logs, mdir):
    rows = {}
    for f in logs:
        txt = f.read_text(errors="replace").replace("\r", "\n")
        for m in re.finditer(r"epoch=(\d+) step=(\d+)/\d+(.*?)(?=epoch=\d+ step=|\Z)", txt, re.S):
            blk = m.group(3)[:600]
            r = dict(epoch=int(m.group(1)), step=int(m.group(2)))
            for k in ("loss", "loss_action", "loss_video", "lr"):
                mm = re.search(rf"(?<![a-z_]){k}=([0-9.]+(?:e[-+]?\d+)?)", blk)
                if mm:
                    r[k] = float(mm.group(1))
            mm = re.search(r"speed=([0-9.]+) step/s", blk)
            if mm:
                r["steps_per_s"] = float(mm.group(1))
            rows[r["step"]] = r
    mdir.mkdir(parents=True, exist_ok=True)
    keys = ["step", "epoch", "loss", "loss_action", "loss_video", "lr", "steps_per_s"]
    with open(mdir / "metrics.csv", "w") as fo:
        fo.write(",".join(keys) + "\n")
        for s in sorted(rows):
            fo.write(",".join("" if rows[s].get(k) is None else str(rows[s][k]) for k in keys) + "\n")
    L = [rows[s] for s in sorted(rows) if "loss" in rows[s]]
    summ = dict(points=len(rows), first=L[0] if L else None, last=L[-1] if L else None,
                min_loss=min((r["loss"] for r in L), default=None),
                mean_steps_per_s=(sum(r["steps_per_s"] for r in rows.values() if "steps_per_s" in r)
                                  / max(1, sum(1 for r in rows.values() if "steps_per_s" in r))))
    (mdir / "loss_summary.json").write_text(json.dumps(summ, indent=1) + "\n")


def _fw_prompt_files(task, run):
    """File names of THIS task's prompt embeddings in the shared FastWAM text cache (FastWAM naming:
    sha256(DEFAULT_PROMPT.format(task=...)).t5_len<context_len>.wan22ti2v5b.pt, robot_video_dataset.py)."""
    import ast
    src = (FW / "FastWAM/src/fastwam/datasets/lerobot/robot_video_dataset.py").read_text()
    prompt = ast.literal_eval(re.search(r"DEFAULT_PROMPT\s*=\s*(.+)", src).group(1).strip())
    lens = set(re.findall(r"^\s*context_len:\s*(\d+)\s*$", (run / "out" / "config.yaml").read_text(), re.M))
    if len(lens) != 1:
        raise RuntimeError(f"{task}: context_len in config.yaml is {sorted(lens)}")
    names = set()
    for line in (FW / "data" / task / "meta" / "tasks.jsonl").read_text().splitlines():
        if line.strip():
            h = hashlib.sha256(prompt.format(task=json.loads(line)["task"]).encode("utf-8")).hexdigest()
            names.add(f"{h}.t5_len{next(iter(lens))}.wan22ti2v5b.pt")
    if not names:
        raise RuntimeError(f"{task}: no prompt in tasks.jsonl")
    return sorted(names)


def collect_fastwam(task):
    R = FW / "runs" / task
    if not (R / "complete.json").is_file():
        return None
    done = rd(R / "complete.json")
    W = R / "out" / "checkpoints" / "weights"
    # 2026-11-18: the last two checkpoints from complete.json (C6 runs 8000 steps, saves 4000 / 8000)
    _fw_steps = (int(done.get("save_every", 2000)), int(done.get("max_steps", 4000)))
    eps = [dict(epoch=None, step=s, name=f"step-{s:06d}", files={"model.pt": W / f"step_{s:06d}.pt"}) for s in _fw_steps]
    for e in eps:
        if not e["files"]["model.pt"].is_file():
            raise RuntimeError(f"missing {e['files']['model.pt']}")
    D = FW / "data" / task
    iface = rd(D / "meta" / "hb_fastwam_interface.json")
    prof = iface["profile"]
    common = {"training_config.yaml": R / "out" / "config.yaml", "dataset_stats.json": R / "out" / "dataset_stats.json",
              "model_config.yaml": FW / f"model-{prof}.yaml", "complete.json": R / "complete.json",
              "dataset_meta/hb_fastwam_interface.json": D / "meta" / "hb_fastwam_interface.json",
              "dataset_meta/info.json": D / "meta" / "info.json", "dataset_meta/tasks.jsonl": D / "meta" / "tasks.jsonl",
              "batch_benchmark/summary.txt": FW / "bench-20261116-r2" / "summary.txt",
              "batch_benchmark/chosen.json": FW / "chosen.json",
              "code/hb_aug_v2.py": G / "hb_groot" / "hb_aug_v2.py",
              "code/fastwam_task.sh": G / "hb_fastwam" / "fastwam_task.sh",
              "code/fastwam_task_s.sh": G / "hb_fastwam" / "fastwam_task_s.sh",
              "code/fastwam_prep.sh": FW / "tools" / "fastwam_prep.sh"}
    # 2026-11-16 user: ship the inference adapter (and the two modules it / hb_inloop_fastwam import) with every
    # FastWAM checkpoint; collaborators deploy straight from HF
    EV = G / "eval-action-mse"
    common.update({"code/ad_fastwam.py": EV / "adapters" / "ad_fastwam.py", "code/hb_eval_common.py": EV / "hb_eval_common.py",
                   "code/hb_inloop_core.py": EV / "hb_inloop_core.py",
                   "code/test_fastwam_inputs.py": EV / "test_fastwam_inputs.py"})
    for f in sorted((FW / "tools").glob("*.py")):
        common[f"code/{f.name}"] = f
    # 2026-11-16 (A800 handoff session): only this task's prompt embeddings, not the whole shared cache, so prompts
    # added for later tasks never make an earlier upload mismatch (missing files are caught by the check below)
    for name in _fw_prompt_files(task, R):
        common[f"text_embeddings/{name}"] = FW / "text-cache" / "all" / name
    logs = [f for f in sorted((R / "logs").glob("node-*.log")) if f.stat().st_size < 50 * 1024 * 1024]
    for f in logs:
        common[f"logs/{f.name}"] = f
    mdir = R / "hb-metrics"
    _fw_metrics([f for f in logs if f.name == "node-0.log"], mdir)
    common["metrics.csv"] = mdir / "metrics.csv"
    common["loss_summary.json"] = mdir / "loss_summary.json"
    for k, v in common.items():
        if not Path(v).is_file():
            raise RuntimeError(f"missing {k}: {v}")
    proto = dict(epochs=None, total_steps=int(done.get("max_steps", 4000)), save_steps=list(_fw_steps), num_gpus=32,
                 per_gpu_batch=done["per_gpu_batch"], global_batch=done["global_batch"], gradient_accumulation_steps=1,
                 episodes=iface["episodes"], frames=iface["frames"], fps=iface.get("fps"),
                 cameras=iface["cameras"], state_dim=iface["state_dim"], action_dim=iface["action_dim"],
                 state_layout=FW_STATE[prof], action_layout=FW_ACTION[prof],
                 trainable="full fine-tune of the mixture-of-transformers DiT: video expert (Wan2.2-TI2V-5B, 5.00 B) + "
                           "action expert (1.02 B); the Wan2.2 VAE and the T5 text encoder are frozen (prompt "
                           "embeddings precomputed, text_embeddings/)",
                 recipe=done.get("recipe"),
                 video="17-frame clips (4 latent frames + the observation frame), 224 x 448 (the two cameras side by "
                       "side), action chunk 16",
                 base_model="Wan-AI/Wan2.2-TI2V-5B (DiT, VAE, T5) + the umt5-xxl tokenizer of Wan-AI/Wan2.1-T2V-1.3B, "
                            "loaded from their own repositories; ActionDiT initialised by FastWAM's "
                            "preprocess_action_dit_backbone.py",
                 upstream="github.com/yuantianyuan01/FastWAM @ 7faa71108368fbb3b6885649f112af607427a2d4",
                 batch_selection="largest per-GPU batch that fits on A800 80 GB: 16 trains at ~58 GB, 32 runs out of "
                                 "memory (batch_benchmark/)",
                 normalization="dataset_stats.json (FastWAM min / max statistics of this task)",
                 config="training_config.yaml (resolved FastWAM config; absolute paths refer to the training NAS), "
                        "model_config.yaml (the model part)",
                 inference="FastWAM: build the model from model_config.yaml, load model.pt (DiT state dict), the "
                           "normalizer from dataset_stats.json and the prompt embedding from text_embeddings/; "
                           "code/hb_fastwam_convert.py documents the state / action layout of the benchmark interface; "
                           "code/ad_fastwam.py is the inference adapter of our evaluation (action-only "
                           "FastWAM.infer_action, 10 denoising steps, seed 42; its default paths refer to the "
                           "training NAS)")
    if task == "b6v6":
        proto["note"] = ("FastWAM B6 v6 is trained from the FastWAM initialisation on all 542 v6 episodes (there is "
                         "no earlier FastWAM B6 run to continue from)")
    return dict(eps=eps, common=common, proto=proto)


COLLECT.update(fastwam=collect_fastwam)


# 2026-11-16 16:2x user: FastWAM A18 cropped v2 (trained right after A17 cropped v2, before B7)
FW_TASKS = FW_TASKS + ("a18cr2",)
SUBDIR.update({("fastwam", "a18cr2"): "b512"})

# 2026-11-16 20:4x user: FastWAM on B5 chair push standing start (robot starts right in front of the chair)
for _d, _v in ((TASKDIR, dict(b5ss="b5-chair-push-standing-start")),
               (DESC, dict(b5ss="Chair push from a standing start right in front of the chair: raise the hands and push the chair")),
               (TASKNAME, dict(b5ss="B5 chair push, standing start (G1)")),
               (HFDATA, dict(b5ss="b5-chair-push-standing-start-expert-v1 @913f5a1 (all 100 episodes)")),
               (ROBOT, dict(b5ss="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
EPISODES_OF.update(b5ss=100)
FW_TASKS = FW_TASKS + ("b5ss",)
SUBDIR.update({("fastwam", "b5ss"): "b512"})

# 2026-11-17 11:3x user: B7 black on the trimmed data (b7-black-clothes-hamper @949da46, first 100 of 107)
for _d, _v in ((TASKDIR, dict(b7t=TASKDIR["b7"])), (DESC, dict(b7t=DESC["b7"])),
               (TASKNAME, dict(b7t="B7 black clothes into the laundry hamper, trimmed data (G1)")),
               (HFDATA, dict(b7t="b7-black-clothes-hamper @949da46 (trimmed; first 100 of 107 episodes)")),
               (ROBOT, dict(b7t="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
EPISODES_OF.update(b7t=100)
FW_TASKS = FW_TASKS + ("b7t",)
SUBDIR.update({("fastwam", "b7t"): "b512"})

# 2026-11-17 11:3x user: Delta Flow B7 black on the trimmed data -> deltaflow/b7-black-clothes-hamper/b256/
SUBDIR.update({("deltaflow", "b7t"): "b256"})

# 2026-11-20 user: FastWAM on every task's latest data, ahead of Delta Flow. b7gt = B7 green on HF b7-green-clothes-hamper
# @22e3b83 (trimmed v2 at the repo root, all episodes; the groot key b7green keeps its own data record)
for _d, _v in ((TASKDIR, dict(b7gt="b7-green-clothes-hamper")), (DESC, dict(b7gt=DESC["b7green"])),
               (TASKNAME, dict(b7gt="B7 green clothes -> hamper, trimmed v2 data (G1)")),
               (HFDATA, dict(b7gt="b7-green-clothes-hamper @22e3b83 (trimmed v2, all episodes)")),
               (ROBOT, dict(b7gt="Unitree G1"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
FW_TASKS = FW_TASKS + ("b7gt",)
SUBDIR.update({("fastwam", "b7gt"): "b512"})

# 2026-11-20 user: Delta Flow on b5ss / b7gt (all episodes, tree HumanWAM-allep) -> deltaflow/<task dir>/b256/
SUBDIR.update({("deltaflow", "b5ss"): "b256", ("deltaflow", "b7gt"): "b256"})
EPISODES_OF.update(b7gt=421)

# 2026-11-20 user: pi0.5 on every G1 task latest data; new subdirs where an older pi0.5 run already sits at <task>/b256/
SUBDIR.update({("pi05", "a3v4"): "b256", ("pi05", "a7v4"): "b256", ("pi05", "a10v3"): "b256", ("pi05", "basketv2"): "b256-trimv2", ("pi05", "b1"): "b256", ("pi05", "b5"): "b256", ("pi05", "b5ss"): "b256", ("pi05", "b6v6"): "b256-v6", ("pi05", "b7t"): "b256-trim", ("pi05", "b7gt"): "b256", ("pi05", "b8"): "b256"})

# 2026-11-20 user: DiT4DiT on every G1 task latest data (global batch 64) -> dit4dit/<task dir>/b64/
SUBDIR.update({("dit4dit", "a3v4"): "b64", ("dit4dit", "a7v4"): "b64", ("dit4dit", "a10v3"): "b64", ("dit4dit", "basketv2"): "b64", ("dit4dit", "b1"): "b64", ("dit4dit", "b5"): "b64", ("dit4dit", "b5ss"): "b64", ("dit4dit", "b6v6"): "b64", ("dit4dit", "b7t"): "b64", ("dit4dit", "b7gt"): "b64", ("dit4dit", "b8"): "b64"})

# 2026-11-20 user: A14 screw-bulb cropped-v3 (newest A14, trimmed from v2 @8411a6d) on GR00T (4 x 16), Psi-Zero (4 x 16), FastWAM
H2DATA["a14cr3"] = "humanoid-benchmark/a14-screw-bulb-expert-cropped-v3 @ 79fe08a556b72999e17964d381d84e6e7332a68b"
H2REF["a14cr3"] = "a14-unscrew-bulb/v2-b512 (A14 v2, untrimmed)"
H2ACTION["a14cr3"] = H2ACTION["a14v2"]
for _d, _v in ((TASKDIR, dict(a14cr3=TASKDIR["a14"])), (DESC, dict(a14cr3=DESC["a14"])),
               (TASKNAME, dict(a14cr3="A14 screw in a bulb, cropped-v3 data (H2, cameras head + chest, action82 with the planner command)")),
               (HFDATA, dict(a14cr3="a14-screw-bulb-expert-cropped-v3 @79fe08a (trimmed from v2 @8411a6d, all 100 episodes)")),
               (ROBOT, dict(a14cr3="H2 (Sharpa hands)"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
EPISODES_OF.update(a14cr3=100)
FW_TASKS = FW_TASKS + ("a14cr3",)
SUBDIR.update({("groot2x32", "a14cr3"): "cropped-v3-b512", ("psi0", "a14cr3"): "cropped-v3-b512", ("fastwam", "a14cr3"): "cropped-v3-b512"})

# 2026-11-20 user: FastWAM continuation, 2000 more steps (weights only, fresh AdamW, warmup 5 % to peak lr 3e-5 =
# 0.3 x the original 1e-4, cosine) on the 7 tasks whose action nMSE still fell at step 4000. Run runs/<task>-cont2k,
# uploaded to fastwam/<task dir>/b512-cont2k/step-005000 and step-006000 (total steps); the 4000-step b512/ stays.
FWC_TASKS = ("b5", "b8", "a7v4", "a10v3", "basketv2", "b1", "b6v6", "a18cr2", "a3v4", "b7gt", "b7t", "a17cr2", "a14cr3")  # 11-20 22:3x: + a18cr2, a3v4 after the const-dim re-eval; 11-22 user: + b7gt b7t a17cr2 a14cr3 (all at 6000)
for _b in FWC_TASKS:
    _k = _b + "-cont2k"
    for _d in (TASKDIR, DESC, TASKNAME, HFDATA, ROBOT, EPISODES_OF):
        if _b in _d:
            _d.setdefault(_k, _d[_b])
    SUBDIR[("fastwam", _k)] = "b512-cont2k"
FW_TASKS = FW_TASKS + tuple(_b + "-cont2k" for _b in FWC_TASKS)
SUBDIR[("fastwam", "a14cr3-cont2k")] = "cropped-v3-b512-cont2k"  # 11-22: next to cropped-v3-b512
_collect_fastwam_v1 = collect_fastwam


def collect_fastwam(task):
    c = _collect_fastwam_v1(task)
    if c is None or not task.endswith("-cont2k"):
        return c
    base = task[:-len("-cont2k")]
    for e in c["eps"]:
        e["step"] += 4000
        e["name"] = f"step-{e['step']:06d}"
    c["proto"].update(total_steps=6000, save_steps=[e["step"] for e in c["eps"]], continuation_steps=2000,
                      continued_from=f"fastwam/{TASKDIR[base]}/{SUBDIR.get(('fastwam', base), 'b512')}/step-004000",
                      continuation="weights only (DiT + proprio encoder loaded before DeepSpeed init, "
                                   "code/hb_fastwam_cont.py): fresh AdamW, warmup 5 % to peak lr 3e-5 (0.3 x the "
                                   "original 1e-4), cosine to 1 % over 2000 steps; same data, normalization, "
                                   "augmentation and batch as the 4000-step run")
    c["common"]["code/fastwam_cont.sh"] = G / "hb_fastwam" / "fastwam_cont.sh"
    return c


COLLECT.update(fastwam=collect_fastwam)

# 2026-11-22 user: GR00T / Psi-Zero on the newest data still missing on HF (never over the existing b512/ dirs)
SUBDIR.update({("groot2x32", "b7t"): "b512-trim", ("groot2x32", "b7gt"): "b512-trimv2", ("groot2x32", "b5ss"): "b512",
               ("psi0", "b7t"): "b512-trim", ("psi0", "b7gt"): "b512-trimv2", ("psi0", "b5ss"): "b512",
               ("psi0", "a18cr2"): "b512-2cam-cropped-v2", ("psi0", "a17cr2"): "b512-2cam-cropped-v2"})
# 2026-11-22 user: FastWAM C6 continuation, 4000 more steps from step 8000 (C6 long-task 2x rule) -> fastwam/<c6>/b512-cont4k/
for _d in (TASKDIR, DESC, TASKNAME, HFDATA, ROBOT, EPISODES_OF):
    if "c6" in _d:
        _d.setdefault("c6-cont4k", _d["c6"])
SUBDIR[("fastwam", "c6-cont4k")] = "b512-cont4k"
FW_TASKS = FW_TASKS + ("c6-cont4k",)
_collect_fastwam_before_c6c = COLLECT["fastwam"]


def collect_fastwam_c6c(task):
    if task != "c6-cont4k":
        return _collect_fastwam_before_c6c(task)
    c = _collect_fastwam_v1(task)
    if c is None:
        return c
    for e in c["eps"]:
        e["step"] += 8000
        e["name"] = f"step-{e['step']:06d}"
    c["proto"].update(total_steps=12000, save_steps=[e["step"] for e in c["eps"]], continuation_steps=4000,
                      continued_from=f"fastwam/{TASKDIR['c6']}/{SUBDIR.get(('fastwam', 'c6'), 'b512')}/step-008000",
                      continuation="weights only (DiT + proprio encoder loaded before DeepSpeed init, "
                                   "code/hb_fastwam_cont.py): fresh AdamW, warmup 5 % to peak lr 3e-5 (0.3 x the "
                                   "original 1e-4), cosine to 1 % over 4000 steps (C6 long-task rule: 2 x the 2000-step "
                                   "continuation of the other tasks); same data, normalization, augmentation and batch")
    c["common"]["code/fastwam_cont4k.sh"] = G / "hb_fastwam" / "fastwam_cont4k.sh"
    return c


COLLECT.update(fastwam=collect_fastwam_c6c)

# 2026-11-22 user: H2 aligned with every other task. GR00T a18cr2 / a17cr2 on the standard recipe (h32 dropped) and Psi-Zero
# a14cr3 again without the proprio delay (task key a14cr3a, same data); new sub-dirs, nothing existing is overwritten.
for _d in (TASKDIR, DESC, TASKNAME, HFDATA, ROBOT, EPISODES_OF, H2DATA, H2REF, H2ACTION):
    if "a14cr3" in _d:
        _d.setdefault("a14cr3a", _d["a14cr3"])
SUBDIR.update({("groot2x32", "a18cr2"): "b512-2cam-cropped-v2", ("groot2x32", "a17cr2"): "b512-2cam-cropped-v2",
               ("psi0", "a14cr3a"): "cropped-v3-b512-nodelay"})


# 2026-11-23 user: GR00T / Psi-Zero pure B6 v6 (all 542, from base, no v5 stage) -> new sub-dirs b512-v6-pure/
for _d in (TASKDIR, DESC, TASKNAME, HFDATA, ROBOT, EPISODES_OF):
    if "b6v6" in _d:
        _d.setdefault("b6v6p", _d["b6v6"])
HFDATA["b6v6p"] = ("b6-pedal-bin @ad2f5bf0 (v6 at the repo root, all 542 episodes; trained from the base model on v6 only, "
                   "no v5 stage; user exception to the B6 v6 first-500 rule - FastWAM b6v6 used episodes 0-499)")
SUBDIR.update({("groot2x32", "b6v6p"): "b512-v6-pure", ("psi0", "b6v6p"): "b512-v6-pure"})


def main():
    lock = open(A / "daemon.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    log("daemon start")
    while True:
        try:
            preserve()
        except Exception:
            log("preserve error", traceback.format_exc()[-400:])
        jobs = json.loads((A / "jobs.json").read_text()) if (A / "jobs.json").is_file() else []
        state = json.loads((A / "state.json").read_text()) if (A / "state.json").is_file() else {}
        for j in jobs:
            key = f"{j['model']}/{j['task']}"
            st = state.get(key, {})
            if st.get("status") == "uploaded" or st.get("fails", 0) >= 5:
                continue
            if st.get("retry_after", 0) > time.time():
                continue
            try:
                c = COLLECT[j["model"]](j["task"])
                if c is None:
                    continue
                log("complete, staging", key)
                dst = stage(j["model"], j["task"], c)
                res = upload(j["model"], j["task"], dst)
                ok = not res["bad"] and not res["extra"] and not res["private"]
                state[key] = dict(status="uploaded" if ok else "mismatch", utc=time.strftime("%FT%TZ", time.gmtime()),
                                  epochs=[e["epoch"] for e in c["eps"] if e.get("files") or e.get("tree")], **res)
                if not ok:
                    state[key].update(fails=st.get("fails", 0) + 1, retry_after=time.time() + 1800)
                log("UPLOADED" if ok else "MISMATCH", key, json.dumps(state[key])[:600])
            except Exception:
                state[key] = dict(st, status="error", fails=st.get("fails", 0) + 1, retry_after=time.time() + 1800,
                                  error=traceback.format_exc()[-600:])
                log("ERROR", key, traceback.format_exc()[-800:])
            (A / "state.json").write_text(json.dumps(state, indent=1) + "\n")
        if ONCE:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
