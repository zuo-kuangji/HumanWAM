"""Create public model repos humanoid-benchmark/{pi05,psi0}, upload the staged folders (resumable), then verify
every file against HF (size + LFS sha256, or git blob sha1 for small files).
Token: read from $HB_TOKEN_FILE into memory only. Usage: HB_TOKEN_FILE=... python upload_model_repos.py $P/hf-models-20261110
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

OUT = Path(sys.argv[1])
REPOS = sys.argv[2:] or ["pi05", "psi0"]
ORG = "humanoid-benchmark"
api = HfApi(token=Path(os.environ["HB_TOKEN_FILE"]).read_text().strip())
status_path = OUT.parent / f"{OUT.name}-status.json"
status = json.loads(status_path.read_text()) if status_path.is_file() else {}


def save():
    status["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status_path.write_text(json.dumps(status, indent=1) + "\n")


def local_files(d):
    return {str(p.relative_to(d)): p for p in sorted(d.rglob("*"))
            if p.is_file() and ".cache" not in p.relative_to(d).parts}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_sha1(p):
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


for name in REPOS:
    repo = f"{ORG}/{name}"
    folder = OUT / name
    status[name] = {"phase": "CREATE"}
    save()
    api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
    if api.model_info(repo).private:
        status[name] = {"phase": "ABORT_REPO_IS_PRIVATE"}
        save()
        sys.exit(2)
    status[name] = {"phase": "UPLOADING", "files": len(local_files(folder))}
    save()
    api.upload_large_folder(repo_id=repo, folder_path=str(folder), repo_type="model", num_workers=4,
                            print_report_every=300)
    status[name] = {"phase": "VERIFYING"}
    save()
    remote = {f.path: f for f in api.list_repo_tree(repo, recursive=True, expand=True) if hasattr(f, "size")}
    checks, bad = {}, []
    for k, p in local_files(folder).items():
        r = remote.get(k)
        if r is None:
            bad.append(k)
            checks[k] = {"missing": True}
            continue
        if r.lfs is not None:
            local, hf = sha256(p), r.lfs.sha256
        else:
            local, hf = git_sha1(p), r.blob_id
        ok = local == hf and r.size == p.stat().st_size
        checks[k] = {"bytes": p.stat().st_size, "local": local, "hf": hf, "ok": ok}
        if not ok:
            bad.append(k)
    info = api.model_info(repo)
    status[name] = {"phase": "HF_VERIFIED" if not bad else "HF_MISMATCH", "repo": repo, "private": info.private,
                    "revision": info.sha, "files": len(checks), "bad": bad}
    (OUT / f"{name}-verify.json").write_text(json.dumps({"repo": repo, "revision": info.sha, "checks": checks},
                                                       indent=1) + "\n")
    save()
print(json.dumps(status, indent=1))
