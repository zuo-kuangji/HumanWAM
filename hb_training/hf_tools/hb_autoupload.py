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
               (HFDATA, dict(a1="a1-ground-object-pickup-expert-v1 @f47f07e"))):
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
REPO_OF = {"psi0a": "psi0", "act256": "act", "grootft": "groot"}


def repo_path(model, task):
    sub = SUBDIR.get((model, task))
    return TASKDIR[task] + ("/" + sub if sub else "")


MODEL_TEXT["psi0a"] = MODEL_TEXT["psi0"]
MODEL_TEXT["act256"] = MODEL_TEXT["act"]
MODEL_TEXT["grootft"] = MODEL_TEXT["groot"]
CODE_DIR = {"act": "hb_act", "psi0": "hb_psi0", "deltaflow": "hb_deltaflow", "dit4dit": "hb_dit4dit",
            "groot": "hb_groot", "pi05": "hb_pi05", "psi0a": "hb_psi0", "act256": "hb_act", "grootft": "hb_groot"}


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


COLLECT = {"act": collect_act, "psi0": collect_psi0, "deltaflow": collect_deltaflow, "dit4dit": collect_dit4dit, "groot": collect_groot, "pi05": collect_pi05, "psi0a": collect_psi0a, "act256": collect_act256, "grootft": collect_grootft}


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
                 "training_data": HFDATA[task], "episodes": 100, "epoch_completed": e["epoch"], "global_step": e["step"],
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
    if model in ("groot", "pi05", "psi0a", "act256", "grootft"):
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
