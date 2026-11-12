"""Upload the staged sidecars to humanoid-benchmark/g1-basket-transfer-imagewam-deltaflow76 and read them back.
Token: typed by the user at a hidden prompt; kept in memory only (never written, never logged).
Never overwrites an existing repo file. Usage: python upload_sidecars.py $P/hf-sidecars-20261110
"""
import getpass
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

REPO = "humanoid-benchmark/g1-basket-transfer-imagewam-deltaflow76"
CKPTS = {"a3-fridge-open/epoch-040": "step_004440.pt", "a7-cabinet-close/epoch-040": "step_004800.pt",
         "h2-a18-pour-water/epoch-040": "step_011400.pt", "epoch-040": "step_008520.pt"}
OUT = Path(sys.argv[1])
receipt_path = OUT / "upload-receipt.json"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def save(obj):
    receipt_path.write_text(json.dumps(obj, indent=1) + "\n")
    print(json.dumps(obj, indent=1), flush=True)


tf = os.environ.get("HB_TOKEN_FILE")
tok = (Path(tf).read_text() if tf else getpass.getpass("HF write token for %s (hidden): " % REPO)).strip()
if not tok:
    save({"phase": "ABORT_EMPTY_TOKEN"})
    sys.exit(4)
api = HfApi(token=tok)
who = api.whoami()
at = (who.get("auth") or {}).get("accessToken") or {}
print(json.dumps({"hf_user": who.get("name"), "token_name": at.get("displayName"), "token_role": at.get("role"),
                  "orgs": [o.get("name") for o in who.get("orgs", [])]}), flush=True)
existing = set(api.list_repo_files(REPO))
missing_ckpt = [f"{d}/{c}" for d, c in CKPTS.items() if f"{d}/{c}" not in existing]
if missing_ckpt:
    save({"phase": "ABORT_CHECKPOINT_PATHS_NOT_FOUND", "missing": missing_ckpt})
    sys.exit(2)

local = {str(p.relative_to(OUT)).replace("\\", "/"): p for p in sorted(OUT.rglob("*"))
         if p.is_file() and p.name != receipt_path.name}
clash = sorted(k for k in local if k in existing)
if clash:
    save({"phase": "ABORT_WOULD_OVERWRITE", "existing": clash})
    sys.exit(3)

ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)) for k, p in local.items()]
info = api.create_commit(REPO, operations=ops, commit_message=(
    "Add inference sidecars (training config, normalization stats, Qwen text cache, protocol) for epoch-040 "
    "of a3-fridge-open, a7-cabinet-close, h2-a18-pour-water and basket transfer"))
rev = info.oid

tmp = Path(tempfile.mkdtemp(prefix="hbside-"))
checks = {}
try:
    for k, p in local.items():
        got = hf_hub_download(REPO, k, revision=rev, token=tok, cache_dir=str(tmp))
        checks[k] = {"bytes": p.stat().st_size, "sha256": sha256(p), "hf_sha256": sha256(got)}
finally:
    shutil.rmtree(tmp, ignore_errors=True)
bad = [k for k, v in checks.items() if v["sha256"] != v["hf_sha256"]]
save({"phase": "HF_VERIFIED" if not bad else "HF_MISMATCH", "repo": REPO, "revision": rev,
      "commit_url": info.commit_url, "files": len(checks), "mismatch": bad, "checks": checks})
sys.exit(0 if not bad else 1)
