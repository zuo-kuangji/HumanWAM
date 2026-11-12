"""Model cards (README.md) of the public humanoid-benchmark/<model> repositories, rebuilt from what the repository
actually contains (task folders, variant sub-folders, epoch folders and their protocol.json).
Used by hb_autoupload.py after every upload; also runnable by hand:
    python hb_modelcard.py <repo> [<repo> ...] [--dry]      repo = groot | pi05 | psi0 | act | deltaflow | dit4dit
"""
import json
import re
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ORG = "humanoid-benchmark"
TOKEN_FILE = Path("/root/.hb-secrets/hf_token")
# task folder -> (task, robot); order = order in the card
TASKS = {
    "b6-pedal-bin": ("B6 — pick up a chip barrel, step on the pedal bin, drop it in", "G1"),
    "a3-fridge-open": ("A3 — open the fridge door", "G1"),
    "a7-cabinet-close": ("A7 — close the cabinet door", "G1"),
    "a6-door-pull": ("A6 — pull a door open", "G1"),
    "a10-trolley-push": ("A10 — push a trolley", "G1"),
    "a20-basket-transfer": ("A20 — transfer a basket", "G1"),
    "b7-black-clothes-hamper": ("B7 — take the black garment off the hanger and drop it into the hamper", "G1"),
    "c6-bathroom-door-fetch": ("C6 — open the bathroom door, take the tea bottle from the vanity, put it on the dining table",
                               "G1"),
    "a18-pour-water": ("A18 — pour water from a pitcher into a glass", "H2"),
    "a17-unscrew-bottle": ("A17 — unscrew the cap of a bottle", "H2"),
    "a14-unscrew-bulb": ("A14 — insert a bulb into the socket and screw it in until it lights up", "H2"),
}
ROBOT = {"G1": "Unitree G1 (Inspire hands)", "H2": "H2 (Sharpa hands)"}
HEAD = {
    "groot": ("GR00T N1.7 — Humanoid Benchmark",
              "GR00T N1.7 (Gr00tN1d7) checkpoints fine-tuned on Humanoid Benchmark tasks (action head fine-tuned, VLM "
              "backbone frozen). The VLM backbone `nvidia/Cosmos-Reason2-2B` is loaded from its own (gated) repository.",
              "Earlier repository states are kept in the git history (e.g. revision `2e8f375584631cae0454911e58b5b7850ea9f678`, "
              "an older batch setting)."),
    "pi05": ("pi0.5 — Humanoid Benchmark",
             "pi0.5 (openpi, PyTorch weights) checkpoints fine-tuned from `pi05_base` on Humanoid Benchmark tasks.",
             "Earlier repository states are kept in the git history (e.g. revision `b04edfb31a62fbcd7e85b7c3a85b6dfc8cd3587b`, "
             "an older batch setting)."),
    "psi0": ("Psi-Zero — Humanoid Benchmark",
             "Psi-Zero (Psi0 action expert, flow matching) checkpoints trained on Humanoid Benchmark tasks.", ""),
    "act": ("ACT — Humanoid Benchmark",
            "ACT (Action Chunking Transformer, ResNet18 backbone, CVAE; Psi0 `baselines/act`) checkpoints trained on "
            "Humanoid Benchmark tasks.", ""),
    "deltaflow": ("Delta Flow — Humanoid Benchmark",
                  "Delta Flow (ImageWAM / HumanWAM whole-body delta-flow policy) checkpoints trained on Humanoid Benchmark "
                  "tasks (40 epochs; the last two saved checkpoints are published).",
                  "`legacy/` holds earlier experiments that do NOT follow the benchmark protocol; see `legacy/README.md`."),
    "dit4dit": ("DiT4DiT — Humanoid Benchmark",
                "DiT4DiT (Cosmos-Predict2.5-2B video DiT + action DiT) checkpoints trained on Humanoid Benchmark tasks, "
                "with DiT4DiT's native sin/cos state encoding inside the model (G1 73-D, H2 153-D) of the same raw "
                "44 / 78-D benchmark state.", ""),
}
VARIANT_NOTE = {}  # every sub-folder is b<global batch> (user rule 2026-11-12)


def _cams(p, robot):
    c = p.get("cameras") or p.get("image_keys") or p.get("video_keys")
    if isinstance(c, dict):
        c = [v for v in c.values() if "zeros" not in str(v)]
    if not c:
        c = ["observation.images.ego_view", "observation.images.chest"] if robot == "G1" else \
            ["observation.images.head", "observation.images.chest"]
    return ", ".join(str(x).replace("observation.images.", "") for x in c)


def _proprio(model, p):
    if model == "groot":
        return "aligned (GR00T A20 recipe)"
    text = " ".join(str(p.get(k, "")) for k in ("augmentation", "proprio_augmentation"))
    if "per-trajectory" in text:
        return "per-trajectory bias, no delay/drop"
    if "no proprio delay" in text or "aligned" in text:
        return "aligned (GR00T A20 recipe)"
    return "earlier recipe (with delay/drop)"


def build(api, repo, token):
    rid = f"{ORG}/{repo}"
    files = [f.path for f in api.list_repo_tree(rid, recursive=True) if hasattr(f, "size")]
    variants = {}
    for f in files:
        m = re.match(r"^([a-z0-9-]+)/(?:([a-z0-9]+)/)?((?:epoch|step)-\d+)/protocol\.json$", f)
        if m and m.group(1) in TASKS:
            variants.setdefault((m.group(1), m.group(2) or ""), []).append(m.group(3))
    rows = []
    with tempfile.TemporaryDirectory() as td:
        for td_name in TASKS:
            for (t, sub), eps in sorted(variants.items(), key=lambda kv: (kv[0][1] != "", kv[0][1])):
                if t != td_name:
                    continue
                eps = sorted(eps, key=lambda n: (n.split("-")[0], int(n.split("-")[1])))
                folder = t + ("/" + sub if sub else "")
                p = json.load(open(hf_hub_download(rid, f"{folder}/{eps[-1]}/protocol.json", token=token,
                                                   local_dir=td)))
                robot = TASKS[t][1]
                gb = p.get("global_batch") or (p.get("plan") or {}).get("global_batch") or "—"
                note = "re-train, current protocol" if re.fullmatch(r"b\d+", sub or "") else ""
                rows.append(f"| `{folder}/` | {TASKS[t][0]}{' (' + note + ')' if note else ''} | {ROBOT[robot]} | "
                            f"{_cams(p, robot)} | {gb} | {_proprio(repo, p)} | "
                            + ", ".join(f"`{e}`" for e in eps) + " |")
    title, intro, extra = HEAD[repo]
    has_variant = any(sub for (_, sub) in variants)
    table = "\n".join(rows) if rows else "| — | no benchmark checkpoints published yet | | | | | |"
    card = f"""---
tags:
- robotics
- humanoid
- imitation-learning
---
# {title}

{intro}

## Checkpoints

100 demonstrations per task. Every row is one training run; the epoch folders are its published checkpoints.

| Folder | Task | Robot | Cameras (model input) | Global batch | Proprio augmentation | Checkpoints |
|---|---|---|---|---|---|---|
{table}
"""
    if has_variant:
        card += """
A `bNNN/` sub-folder (NNN = its global batch, e.g. `b256/`, `b512/`) is a re-train of that task under the current benchmark protocol (2026-11-12: global
batch GR00T / pi0.5 / Delta Flow / ACT 256, Psi-Zero 512, DiT4DiT 64; proprioception augmentation aligned with the
GR00T A20 recipe) and is the one to use for that task; the checkpoints directly under the task folder are the earlier
training round, kept for reference.
"""
    card += """
## What an epoch folder contains

Every checkpoint folder (`epoch-XXX/`, or `step-XXXXXX/` for runs trained for a fixed number of steps) is self-contained for evaluation and deployment: weights, model configuration, normalization
statistics and `protocol.json` (task, data, input/output dimensions, cameras, batch, steps, augmentation, file list).
Simple training records are included for reproducibility (training plan, training log, configs, metrics); optimizer /
scheduler / RNG / DeepSpeed resume state is not. `code/` holds the adapter code used for training and inference
(`code/VERSIONS.json`: upstream commits and package versions).

## Inputs and augmentation

Model inputs are identical across all benchmark models: G1 state 44-D (29 body joint angles + 6 + 6 measured hand values
+ 3 projected gravity), H2 state 78-D (75 joint angles + 3 projected gravity), no joint velocities. Cameras: G1 ego view +
chest; H2 head + chest (A14: head + chest + wrist). The third-person camera recorded in the datasets is for data review
only and is never a model input. Each camera is a separate image input, except for Delta Flow and DiT4DiT, which
place the views side by side in one frame.
Training uses the Humanoid Benchmark online data augmentation spec v2 (15 photometric / geometric / sensor / lighting
effects, one of them boosted to 1.5x per sample) with the model's own image augmentation turned off, plus joint-angle
bias and noise on the proprioception (column "Proprio augmentation"; the earlier recipe also delayed the
proprioception and dropped action-history rows).
"""
    if extra:
        card += "\n" + extra + "\n"
    return card


def update(api, repo, token, dry=False):
    card = build(api, repo, token)
    rid = f"{ORG}/{repo}"
    try:
        old = open(hf_hub_download(rid, "README.md", token=token, local_dir=tempfile.mkdtemp())).read()
    except Exception:
        old = ""
    if old == card:
        return "unchanged"
    if dry:
        print(card)
        return "dry"
    if token.encode() in card.encode():
        raise RuntimeError("token in card")
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=rid,
                    commit_message="Update model card (task / checkpoint table)")
    return "updated"


if __name__ == "__main__":
    tok = TOKEN_FILE.read_text().strip()
    a = HfApi(token=tok)
    for r in [x for x in sys.argv[1:] if not x.startswith("--")]:
        print(r, update(a, r, tok, dry="--dry" in sys.argv))
