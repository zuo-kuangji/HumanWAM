"""Move the two old Delta Flow repos into the public model repo humanoid-benchmark/deltaflow (2026-11-11, user approved).
  download : full copy of humanoid-benchmark/{g1-basket-transfer-imagewam-deltaflow76,b6-imagewam-deltaflow76} to the NAS
             backup $P/hf-archive-deltaflow-legacy-20261111/<repo>/, every file checked against HF (LFS sha256 / blob sha1)
  stage    : $P/hf-models-20261111-deltaflow/deltaflow/legacy/<task-dir>/<run>/epoch-XXX/ (hard links from the backup);
             weights renamed to model.pt (byte-identical), original protocol kept as original_protocol.json, new
             protocol.json says why the run is NOT a benchmark result. Training records are kept for reproducibility.
             Top-level task folders are reserved for the benchmark round (same layout as humanoid-benchmark/groot, pi05).
  previews : B6 augmentation preview renders (dataset content) -> private humanoid-benchmark/humanoid-benchmark-assets
Usage: HB_TOKEN_FILE=... python migrate_deltaflow.py $P download|stage|previews
"""
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

P = Path(sys.argv[1])
PHASE = sys.argv[2]
ORG = "humanoid-benchmark"
G1R, B6R = "g1-basket-transfer-imagewam-deltaflow76", "b6-imagewam-deltaflow76"
BK = P / "hf-archive-deltaflow-legacy-20261111"
OUT = P / "hf-models-20261111-deltaflow"
api = HfApi(token=Path(os.environ["HB_TOKEN_FILE"]).read_text().strip())
_TOK = Path(os.environ["HB_TOKEN_FILE"]).read_text().strip().encode()

COMMON = ("Delta Flow (ImageWAM, HumanWAM wholebody delta-flow policy): action = motion_token64 + same-frame measured "
          "hands12 (G1, 76-D) or H2 body31 + Sharpa hands44 (75-D); chunk 16, history 16, future image t+16.")
RUNS = [
    dict(task="b6-pedal-bin", run="robust-v3-20261106",
         epochs=[("epoch-060", B6R, "robust-v3-20261106/epoch-060", "model.pt")],
         extra=[(B6R, "robust-v3-20261106/README.md", "README.md"),
                (B6R, "robust-v3-20261106/rollout-support/rollout-support-step7920.tar.gz",
                 "epoch-060/rollout-support-step7920.tar.gz")],
         reason="Early B6 experiment: older B6 data (b6-pedal-bin-expert-v1 + v2, 60 episodes) instead of the benchmark's "
                "100-episode B6 set, older augmentation robust-v3 instead of spec v2, 60 epochs."),
    dict(task="b6-pedal-bin", run="mixed70-sim40-real30-v3-20261107",
         epochs=[(f"epoch-0{e}0", B6R, f"mixed70-sim40-real30-v3-20261107/epoch-0{e}0", "model.pt") for e in (4, 5, 6)],
         extra=[(B6R, "mixed70-sim40-real30-v3-20261107/training-plan.json", "training_plan.json")],
         reason="Sim + real experiment: 40 simulated + 30 real-robot B6 episodes, augmentation robust-v3, 60 epochs. "
                "Not the benchmark data or recipe."),
    dict(task="b6-pedal-bin", run="real30-from-sim60-20261108",
         epochs=[("epoch-020", B6R, "real30-from-sim60-20261108/epoch-020", "model.pt")], extra=[],
         reason="Real-robot model: robust-v3 epoch 60 fine-tuned 20 epochs on 30 real-robot episodes "
                "(ZUO66/wb-trash-bin-sonic-v11-first30). Not a simulation-benchmark result; its 44-D state interface "
                "(no joint velocities) is the reference the benchmark interface follows."),
    dict(task="a20-basket-transfer", run="old-data-ed77719-20261109",
         epochs=[("epoch-040", G1R, "epoch-040", "step_008520.pt")], extra=[],
         reason="Trained on the retired basket data revision ed77719 (54337 frames). The benchmark uses "
                "basket-transfer-expert-v1 @b0ff522 (branch current-dynamics-100-v21)."),
    dict(task="a3-fridge-open", run="state73-with-velocity-20261109",
         epochs=[("epoch-040", G1R, "a3-fridge-open/epoch-040", "step_004440.pt")], extra=[],
         reason="State input is 73-D including 29 joint velocities; the benchmark interface is 44-D without velocities."),
    dict(task="a7-cabinet-close", run="state73-with-velocity-20261109",
         epochs=[("epoch-040", G1R, "a7-cabinet-close/epoch-040", "step_004800.pt")], extra=[],
         reason="State input is 73-D including 29 joint velocities; the benchmark interface is 44-D without velocities."),
    dict(task="a18-pour-water", run="from-scratch-pre-v2-aug-20261108",
         epochs=[("epoch-040", G1R, "h2-a18-pour-water/epoch-040", "step_011400.pt")], extra=[],
         reason="Trained 2026-11-08, before augmentation spec v2; stopped at epoch 40 of a 60-epoch schedule "
                "(learning-rate schedule planned for 60). State 78-D / action 75-D match the benchmark interface."),
]


def git_sha1(p):
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tree(repo):
    return {f.path: f for f in api.list_repo_tree(f"{ORG}/{repo}", recursive=True, expand=True) if hasattr(f, "size")}


def put(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.stat().st_size >= 50 * 1024 * 1024:
        os.link(src, dst)
    else:
        data = src.read_bytes()
        if _TOK in data:
            raise SystemExit(f"ABORT: token in {src}")
        dst.write_bytes(data)


if PHASE == "download":
    BK.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for repo in (G1R, B6R):
        info = api.model_info(f"{ORG}/{repo}")
        t = tree(repo)
        d = BK / repo
        snapshot_download(f"{ORG}/{repo}", revision=info.sha, local_dir=str(d), max_workers=8,
                          token=Path(os.environ["HB_TOKEN_FILE"]).read_text().strip())
        bad = []
        for k, f in t.items():
            p = d / k
            if not p.is_file() or p.stat().st_size != f.size:
                bad.append(k)
                continue
            ok = (sha256(p) == f.lfs.sha256) if f.lfs is not None else (git_sha1(p) == f.blob_id)
            if not ok:
                bad.append(k)
        manifest[repo] = {"revision": info.sha, "files": len(t), "bytes": sum(f.size for f in t.values()), "bad": bad,
                          "lfs_sha256": {k: f.lfs.sha256 for k, f in t.items() if f.lfs is not None}}
        print("DOWNLOADED", repo, info.sha, len(t), "files", round(manifest[repo]["bytes"] / 1e9, 1), "GB bad", bad)
    (BK / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print("DOWNLOAD_OK" if not any(m["bad"] for m in manifest.values()) else "DOWNLOAD_BAD")

elif PHASE == "stage":
    man = json.loads((BK / "manifest.json").read_text())
    assert not any(m["bad"] for m in man.values()), "download not verified"
    R = OUT / "deltaflow"
    R.mkdir(parents=True, exist_ok=True)
    for p in R.iterdir():
        if p.name != ".cache":
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    rows = []
    for r in RUNS:
        base = R / "legacy" / r["task"] / r["run"]
        for ep, repo, src_dir, wname in r["epochs"]:
            src = BK / repo / src_dir
            dst = base / ep
            for f in sorted(src.rglob("*")):
                if not f.is_file() or ".cache" in f.relative_to(src).parts:
                    continue
                rel = f.relative_to(src)
                if str(rel) == wname:
                    put(f, dst / "model.pt")
                elif str(rel) == "protocol.json":
                    put(f, dst / "original_protocol.json")
                else:
                    put(f, dst / rel)
            assert (dst / "model.pt").is_file(), dst
            orig = json.loads((src / "protocol.json").read_text())
            proto = {
                "benchmark_result": False,
                "not_benchmark_because": r["reason"],
                "model": COMMON, "task_dir": r["task"], "run": r["run"], "epoch_folder": ep,
                "weights": "model.pt",
                "weights_sha256": man[repo]["lfs_sha256"][f"{src_dir}/{wname}"],
                "moved_from": {"repo": f"{ORG}/{repo}", "path": f"{src_dir}/{wname}", "revision": man[repo]["revision"],
                               "note": "original repository deleted after this move; file is byte-identical"},
                "original_protocol": "original_protocol.json (file names inside refer to the original repository; "
                                     "weights there were named " + wname + ")",
                "files": {str(p.relative_to(dst)): p.stat().st_size for p in sorted(dst.rglob("*")) if p.is_file()},
            }
            (dst / "protocol.json").write_text(json.dumps(proto, indent=2, ensure_ascii=False) + "\n")
            rows.append((r["task"], r["run"], ep, r["reason"]))
        for repo, sp, dn in r["extra"]:
            put(BK / repo / sp, base / dn)
    table = "\n".join(f"| `legacy/{t}/{run}/{ep}/` | {why} |" for t, run, ep, why in rows)
    (R / "legacy" / "README.md").write_text(f"""# Legacy Delta Flow checkpoints (NOT benchmark results)

These are earlier Delta Flow experiments. None of them follows the Humanoid Benchmark training protocol
(100 demonstrations per task, augmentation spec v2, 44-D G1 / 78-D H2 state without joint velocities, same epochs for
every task), so they must not be used or reported as benchmark baselines. They are kept for reference and reproducibility.
Each folder has `model.pt` (byte-identical to the original upload), its training config, normalization statistics,
text cache where used, the original `original_protocol.json` and a new `protocol.json` with the reason.

| Folder | Why it is not a benchmark checkpoint |
|---|---|
{table}

Moved here on 2026-11-11 from `humanoid-benchmark/{G1R}` (revision `{man[G1R]['revision']}`) and
`humanoid-benchmark/{B6R}` (revision `{man[B6R]['revision']}`).
""")
    (R / "README.md").write_text("""---
tags:
- robotics
- humanoid
- imitation-learning
---
# Delta Flow — Humanoid Benchmark

Delta Flow (ImageWAM / HumanWAM whole-body delta-flow policy) checkpoints for Humanoid Benchmark tasks.

Benchmark checkpoints use the same layout as `humanoid-benchmark/groot` and `humanoid-benchmark/pi05`:
`<task-dir>/epoch-XXX/` with task folders `b6-pedal-bin`, `a3-fridge-open`, `a7-cabinet-close`, `a18-pour-water`,
`a10-trolley-push`, `a20-basket-transfer`, `a17-unscrew-bottle`. The benchmark training round of Delta Flow is
scheduled; its task folders appear here as each run finishes.

`legacy/` holds earlier experiments that do NOT follow the benchmark protocol (different data, augmentation or input
dimensions). Do not use them as benchmark baselines; see `legacy/README.md`.
""")
    fs = [p for p in R.rglob("*") if p.is_file() and ".cache" not in p.relative_to(R).parts]
    print("STAGED", len(fs), "files", round(sum(p.stat().st_size for p in fs) / 1e9, 2), "GB")
    for p in sorted(fs):
        print("  ", p.relative_to(R), p.stat().st_size)

elif PHASE == "previews":
    src = BK / B6R / "augmentation-previews"
    target = f"{ORG}/humanoid-benchmark-assets"
    rtype = None
    for t in ("dataset", "model"):
        try:
            info = api.repo_info(target, repo_type=t)
            rtype = t
            break
        except Exception:
            continue
    assert rtype is not None, "assets repo not found"
    assert info.private, "assets repo is not private: abort"
    c = api.upload_folder(repo_id=target, repo_type=rtype, folder_path=str(src),
                          path_in_repo="augmentation-previews/b6-deltaflow-legacy",
                          commit_message="Move B6 Delta Flow augmentation previews from the public model repo")
    t = {f.path: f for f in api.list_repo_tree(target, repo_type=rtype, recursive=True, expand=True,
                                              path_in_repo="augmentation-previews/b6-deltaflow-legacy")
         if hasattr(f, "size")}
    bad = []
    for p in sorted(src.rglob("*")):
        if not p.is_file() or ".cache" in p.relative_to(src).parts:
            continue
        k = "augmentation-previews/b6-deltaflow-legacy/" + str(p.relative_to(src))
        f = t.get(k)
        ok = f is not None and ((sha256(p) == f.lfs.sha256) if f.lfs is not None else (git_sha1(p) == f.blob_id))
        if not ok:
            bad.append(k)
    print("PREVIEWS", target, rtype, "private", info.private, "files", len(t), "bad", bad, "commit", c.oid)
    print("PREVIEWS_OK" if not bad else "PREVIEWS_BAD")
