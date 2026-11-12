"""Add the training records of the legacy Delta-flow runs to humanoid-benchmark/deltaflow
(legacy/<task-dir>/<run>/training_records/): node-0 trainer logs (all production attempts), the full training config,
data / completion receipts, and the loss table parsed from the logs (metrics.csv, loss_summary.json).
    python legacy_df_records.py $P [--apply]
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import df_metrics  # noqa: E402

P = Path(sys.argv[1])
APPLY = "--apply" in sys.argv
STAGE = P / "hf-legacy-records-20261112"
RUNS = {  # legacy folder -> original experiment directory on the NAS
    "a18-pour-water/from-scratch-pre-v2-aug-20261108": "a18-h2-pour-from-scratch-20261108",
    "a20-basket-transfer/old-data-ed77719-20261109": "g1-basket-transfer-ed77719-40epochs",
    "a3-fridge-open/state73-with-velocity-20261109": "a3-a7-20261109/a3",
    "a7-cabinet-close/state73-with-velocity-20261109": "a3-a7-20261109/a7",
    "b6-pedal-bin/mixed70-sim40-real30-v3-20261107": "mixed70-from-scratch-20261107-v3",
    "b6-pedal-bin/real30-from-sim60-20261108": "real30-from-sim60-20261108",
    "b6-pedal-bin/robust-v3-20261106": "augmented-from-scratch-20261106-v3",
}
RECEIPTS = ["data-verified.json", "completion-receipt.json", "chain-complete.json", "h2-run-manifest.json",
            "run/build_receipt.json", "build_receipt.json", "run/production-gate.json", "code_manifest.json",
            "augmentation_policy.json", "preparation_receipt.json", "split-manifest.json"]
TOK = Path("/root/.hb-secrets/hf_token").read_text().strip()


def node0_logs(src):
    run = src / "run"
    cands = sorted(run.glob("retry/**/node-0.log")) + sorted(run.glob("retry/**/production-node-0.log"))
    cands += [p for p in (run / "node-logs/production-node-0.log", run / "node-logs/supervisor-node-0.log") if p.is_file()]
    seen, out = set(), []
    for p in cands:
        if p.is_file() and p.resolve() not in seen and "step=" in p.read_text(errors="replace"):
            seen.add(p.resolve())
            out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def stage_run(folder, src_rel):
    src = P / src_rel
    dst = STAGE / folder / "training_records"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    logs = node0_logs(src)
    names = []
    for i, p in enumerate(logs):
        rel = str(p.relative_to(src / "run")).replace("/", "__")
        name = f"logs/{i:02d}-{rel}"
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst / name)
        names.append(name)
    for cfg, name in ((src / "run/production.yaml", "production.yaml"), (src / "run/production/config.yaml", "resolved_config.yaml")):
        if cfg.is_file():
            shutil.copy2(cfg, dst / name)
    for r in RECEIPTS:
        if (src / r).is_file():
            shutil.copy2(src / r, dst / Path(r).name)
    rows = df_metrics.parse(logs)
    s = df_metrics.write(dst, rows, names)
    (dst / "README.md").write_text(
        f"Training records of the legacy run `{folder}` (original experiment directory `{src_rel}` on the training NAS).\n\n"
        "- `logs/`: node-0 trainer logs of every production attempt, oldest first (resumed attempts continue the step count).\n"
        "- `metrics.csv`: every logged step (loss, loss components, lr, speed) parsed from those logs; `loss_summary.json`: "
        "mean / min loss per epoch.\n- `production.yaml` / `resolved_config.yaml`: training configuration; other `*.json`: "
        "data verification, build and completion receipts.\n\nThis run does NOT follow the benchmark protocol (see "
        "`legacy/README.md`).\n")
    for p in dst.rglob("*"):
        if p.is_file() and TOK.encode() in p.read_bytes():
            raise SystemExit(f"ABORT token in {p}")
    return dst, s, names


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha1(p):
    b = Path(p).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


report = {}
for folder, src_rel in RUNS.items():
    dst, s, names = stage_run(folder, src_rel)
    report[folder] = dict(logs=names, logged_steps=s["logged_steps"], last_step=s["last_step"], total=s["total_steps"],
                          first_loss=s["first_loss"], last_loss=s["last_loss"],
                          files=sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file()))
    print(folder, json.dumps({k: report[folder][k] for k in ("logged_steps", "last_step", "total", "first_loss", "last_loss")}),
          len(report[folder]["files"]), "files", flush=True)
if not APPLY:
    sys.exit(0)
from huggingface_hub import HfApi  # noqa: E402
api = HfApi(token=TOK)
rid = "humanoid-benchmark/deltaflow"
c = api.upload_folder(repo_id=rid, folder_path=str(STAGE), path_in_repo="legacy",
                      commit_message="legacy: add training records (logs, configs, receipts, loss tables) of every legacy run")
tree = {f.path: f for f in api.list_repo_tree(rid, path_in_repo="legacy", recursive=True, expand=True) if hasattr(f, "size")}
bad = []
for p in STAGE.rglob("*"):
    if p.is_file():
        k = "legacy/" + str(p.relative_to(STAGE))
        r = tree.get(k)
        if r is None or r.size != p.stat().st_size or (r.lfs is not None and r.lfs.sha256 != sha256(p)) or \
                (r.lfs is None and r.blob_id != git_sha1(p)):
            bad.append(k)
print("COMMIT", c.oid, "bad", bad, "private", api.model_info(rid).private)
