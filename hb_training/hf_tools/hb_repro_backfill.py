"""2026-11-23 user: add the seed / reproducibility fields to every protocol.json already published on HF.

  hb_repro_backfill.py stage    download protocol.json + the run file that records the seed, write the enriched
                                files to $P/hf-autoupload/repro-backfill/<repo>/..., print a summary (no upload)
  hb_repro_backfill.py commit   upload the staged files, one commit per repo (only protocol.json files change)
Only adds fields (hb_repro.fields); existing fields are kept. legacy/ folders are skipped (not benchmark protocol).
Seed carriers: GR00T training_args.bin, Psi-Zero / ACT argv.txt or train_args.txt, FastWAM training_config.yaml,
others: the first `seed` key in a small yaml / json file of the same folder.
"""
import io
import json
import re
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hb_repro as R  # noqa: E402

ORG = "humanoid-benchmark"
REPOS = ["groot", "psi0", "fastwam", "pi05", "deltaflow", "dit4dit", "act"]
OUT = HERE.parent / "hf-autoupload" / "repro-backfill"
TODAY = "2026-11-23"
api = HfApi(token=open("/root/.hb-secrets/hf_token").read().strip())


def retry(fn, *a, **k):
    for i in range(8):
        try:
            return fn(*a, **k)
        except Exception as e:  # noqa: BLE001
            if i == 7:
                raise
            wait = 30 * (i + 1) if "429" in repr(e) else 5 * (i + 1)
            print(f"  retry {i + 1} in {wait}s: {repr(e)[:120]}", flush=True)
            time.sleep(wait)


def get(repo, path):
    return Path(retry(hf_hub_download, f"{ORG}/{repo}", path, token=api.token))


def seed_of(repo, folder, names):
    try:
        if repo == "groot" and "training_args.bin" in names:
            import torch
            return int(torch.load(get(repo, f"{folder}/training_args.bin"), weights_only=False).seed)
        for n in ("argv.txt", "train_args.txt"):
            if n in names:
                m = re.search(r"--seed[= ](\d+)", get(repo, f"{folder}/{n}").read_text())
                if m:
                    return int(m.group(1))
        cands = ["training_config.yaml"] + sorted(n for n in names if re.search(r"\.(ya?ml|json)$", n) and "/" not in n
                                                   and n not in ("protocol.json", "training_config.yaml") and "stat" not in n
                                                   and "processor" not in n and "tokenizer" not in n)
        for n in cands[:6]:
            if n in names:
                m = re.search(r'^\s*"?seed"?\s*[:=]\s*(\d+)', get(repo, f"{folder}/{n}").read_text(errors="ignore"), re.M)
                if m:
                    return int(m.group(1))
    except Exception as e:  # noqa: BLE001
        print(f"  seed read failed {repo}/{folder}: {repr(e)[:120]}", flush=True)
    return None


def stage():
    summary = {}
    for repo in REPOS:
        try:
            files = retry(api.list_repo_files, f"{ORG}/{repo}")
        except Exception as e:  # noqa: BLE001
            print("SKIP repo", repo, repr(e)[:120]); continue
        protos = [f for f in files if f.endswith("/protocol.json") and not f.startswith("legacy/")]
        by_dir = {}
        for f in files:
            d, _, n = f.rpartition("/")
            by_dir.setdefault(d, set()).add(n)
        s = summary.setdefault(repo, {"protocols": len(protos), "staged": 0, "already": 0, "seeds": {}, "verified": 0})
        for f in protos:
            dst = OUT / repo / f
            if dst.exists():
                s["staged"] += 1; continue
            proto = json.loads(get(repo, f).read_text())
            if "reproducibility" in proto:
                s["already"] += 1; continue
            folder = f.rpartition("/")[0]
            sd = seed_of(repo, folder, by_dir.get(folder, set()))
            new = dict(proto)
            rf = [get(repo, f"{folder}/config.json")] if R.is_hb_baselines(proto) and "config.json" in by_dir.get(folder, set()) else ()
            new.update(R.fields(repo, proto, seed_found=sd, run_files=rf, backfill=f"{TODAY} (fields added after publication; nothing else changed)"))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(new, indent=2, ensure_ascii=False, default=str) + "\n")
            s["staged"] += 1
            s["seeds"][str(new["seed"])] = s["seeds"].get(str(new["seed"]), 0) + 1
            s["verified"] += sd is not None
            time.sleep(0.2)
        print("REPO", repo, json.dumps(s), flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print("STAGE_DONE", flush=True)


def commit():
    for repo in REPOS:
        root = OUT / repo
        if not root.is_dir():
            continue
        paths = sorted(p for p in root.rglob("protocol.json"))
        for i in range(0, len(paths), 150):
            ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(root)), path_or_fileobj=io.BytesIO(p.read_bytes()))
                   for p in paths[i:i + 150]]
            c = retry(api.create_commit, f"{ORG}/{repo}", operations=ops,
                      commit_message="protocol.json: add seed and reproducibility fields (backfill, see hb_training/REPRODUCIBILITY.md)",
                      commit_description="Adds `seed` and a `reproducibility` block (seed source, augmentation random stream, "
                                         "layout, environment, hardware, upstream code, protocol link). No other field and no "
                                         "other file changed.")
            print("COMMIT", repo, len(ops), getattr(c, "oid", c), flush=True)
    print("COMMIT_DONE", flush=True)


if __name__ == "__main__":
    {"stage": stage, "commit": commit}[sys.argv[1]]()
