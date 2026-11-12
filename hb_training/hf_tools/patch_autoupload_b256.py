"""Patch hb_autoupload.py: GR00T / pi0.5 batch-256 collectors and the b256 sub-directory for A3/A7 re-trains."""
import os
import shutil
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text(encoding="utf-8")
shutil.copy2(p, p.parent / ".hb_autoupload.py.bak-b256")
NL = chr(10)

# 1. sub-directory: A3 / A7 re-trains of GR00T / pi0.5 go to <task-dir>/b256 (the 696 versions stay untouched)
a = 'CODE_DIR = {'
b = ('SUBDIR = {("groot", "a3"): "b256", ("groot", "a7"): "b256",' + NL +
     '          **{("pi05", t): "b256" for t in ("b6", "a3", "a7", "a18", "trolley", "basket", "a17", "b7")}}' + NL + NL + NL +
     'def repo_path(model, task):' + NL +
     '    sub = SUBDIR.get((model, task))' + NL +
     '    return TASKDIR[task] + ("/" + sub if sub else "")' + NL + NL + NL + 'CODE_DIR = {')
assert s.count(a) == 1
s = s.replace(a, b)
s = s.replace('CODE_DIR = {"act": "hb_act", "psi0": "hb_psi0", "deltaflow": "hb_deltaflow", "dit4dit": "hb_dit4dit"}',
              'CODE_DIR = {"act": "hb_act", "psi0": "hb_psi0", "deltaflow": "hb_deltaflow", "dit4dit": "hb_dit4dit",' + NL +
              '            "groot": "hb_groot", "pi05": "hb_pi05"}')
a = '    "dit4dit": ("DiT4DiT — Humanoid Benchmark",'
b = ('    "groot": ("GR00T N1.7 — Humanoid Benchmark", "GR00T N1.7 (Gr00tN1d7, action head fine-tuned, VLM frozen)",' + NL +
     '              "Global batch 696 (2026-11-10 round) or 256 (from 2026-11-12; A3/A7 re-trains under <task>/b256/), 20 epochs, "' + NL +
     '              "saved at epoch 10 and 20."),' + NL +
     '    "pi05": ("pi0.5 — Humanoid Benchmark", "pi0.5 (openpi, PyTorch weights)",' + NL +
     '             "Global batch 696 (2026-11-10 round), 512 (B7) or 256 (from 2026-11-12; A3/A7 re-trains under <task>/b256/), "' + NL +
     '             "20 epochs, saved at epoch 10 and 20."),' + NL + a)
assert s.count(a) == 1
s = s.replace(a, b)

# 1b. new tasks A6 / C6 and the aligned augmentation text (2026-11-12: no proprio delay / history drop)
a = 'ROBOT = {t: ('
assert s.count(a) == 1
s = s.replace(a, '''for _d, _v in ((TASKDIR, dict(a6="a6-door-pull", c6="c6-bathroom-door-fetch")),
               (DESC, dict(a6="Pull a door open", c6="Go through the bathroom door to another room and fetch an object")),
               (TASKNAME, dict(a6="A6 door pull (G1)", c6="C6 bathroom-door cross-room fetch (G1)")),
               (HFDATA, dict(a6="door-pull-expert-v1 @cf3ee9b", c6="c6-bathroom-door-fetch-expert-v1 @3afba69e"))):
    for _k, _x in _v.items():
        _d.setdefault(_k, _x)
''' + a)
a = 'RESUME = re.compile('
assert s.count(a) == 1
s = s.replace(a, '''AUG_ALIGNED = ("Humanoid Benchmark online augmentation spec v2 (2026-11-09): 15 visual effects, one effect per training "
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


''' + a)
a = '"augmentation": AUG, **c["proto"]'
assert s.count(a) == 1
s = s.replace(a, '"augmentation": aug_text(model, task), **c["proto"]')

# 2. collectors
a = 'COLLECT = {"act": collect_act,'
b = '''def collect_groot(task):
    run = G / "groot-runs-b256" / task
    if not (run / "production-complete.json").is_file():
        return None
    plan = rd(run / "production-complete.json")
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
    proto = dict(epochs=plan["epochs"], total_steps=plan["max_steps"], save_steps=plan["save_steps"],
                 global_batch=plan["global_batch"], per_gpu_batch=plan["per_gpu_batch"], num_gpus=plan["world_size"],
                 lr=1e-4, lr_schedule="cosine, warmup_ratio 0.05", weight_decay=1e-5,
                 proprio_augmentation="per-sample joint bias +-2 deg + noise 0.25 deg (hb_aug_v2.perturb_joints); "
                                      "no proprio delay / history drop (the GR00T A20 recipe)",
                 config="experiment_cfg/conf.yaml", normalization="statistics.json",
                 inference="NVIDIA Isaac-GR00T N1.7 policy loader with embodiment_tag new_embodiment and "
                           "experiment_cfg/conf.yaml; VLM nvidia/Cosmos-Reason2-2B from its own repository.")
    return dict(eps=eps, common=common, proto=proto)


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
        eps.append(dict(epoch=ep, step=step, files=files))
    lg = run.parent.parent.parent / "logs"
    common = {"training_plan.json": run / "hb_plan.json",
              "train.log": next(x for x in (lg / "train.log", lg / "train-node-0.log") if x.is_file())}
    cfg = P / "hf-repro-20261111/pi05" / repo_path("pi05", task).replace("/", "__") / "train_config.json"
    if not cfg.is_file():
        return None  # written by pi05cfg_b256.py (openpi env) once hb_plan.json exists
    common["train_config.json"] = cfg
    for f in sorted((G / "hb_pi05").glob("*.py")):
        common["code/hb_pi05/" + f.name] = f
    proto = dict(epochs=plan["epochs"], total_steps=plan["num_train_steps"], save_steps=plan["save_steps"],
                 global_batch=plan["global_batch"], per_gpu_batch=plan["per_gpu_batch"], num_gpus=plan["world_size"],
                 peak_lr=plan.get("peak_lr"), warmup_steps=plan.get("warmup_steps"), openpi_config=plan.get("config"),
                 proprio_augmentation="per-sample joint bias +-2 deg + noise 0.25 deg (hb_aug_v2.perturb_joints); "
                                      "no proprio delay (aligned with the GR00T A20 recipe)",
                 normalization="assets/<asset_id>/norm_stats.json",
                 inference="openpi policy_config.create_trained_policy(get_config(openpi_config), <this folder>) after "
                           "importing code/hb_pi05/hb_g1_config.py")
    return dict(eps=eps, common=common, proto=proto)


''' + a
assert s.count(a) == 1
s = s.replace(a, b)
a = 'COLLECT = {"act": collect_act, "psi0": collect_psi0, "deltaflow": collect_deltaflow, "dit4dit": collect_dit4dit}'
assert s.count(a) == 1
s = s.replace(a, a[:-1] + ', "groot": collect_groot, "pi05": collect_pi05}')

# 3. use repo_path() for the staging / upload paths
n0 = s.count('td = TASKDIR[task]')
s = s.replace('td = TASKDIR[task]', 'td = repo_path(model, task)')
assert n0 == 2, n0
s = s.replace('dst = A / "stage" / model / td', 'dst = A / "stage" / model / td.replace("/", "__")')
# README / code/ of groot and pi05 are maintained by build_b696_repos.py: skip the repo-level side upload
a = '    side = A / "stage" / f"{model}-side"'
assert s.count(a) == 1
s = s.replace(a, '    if model in ("groot", "pi05"):' + NL +
              '        info = api.model_info(repo)' + NL +
              '        return dict(commit=c.oid, revision=info.sha, private=info.private,' + NL +
              '                    files=sum(1 for p in dst.rglob("*") if p.is_file()), bad=bad, extra=extra, tasks_in_repo=tasks)' + NL + a)
t = p.with_name(p.name + ".tmp")
t.write_text(s, encoding="utf-8")
os.replace(t, p)
print("PATCHED")
