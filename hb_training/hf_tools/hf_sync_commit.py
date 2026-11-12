"""Make humanoid-benchmark/<repo> equal to a staged folder in ONE commit, renaming folders server side.
- LFS file whose new path is a rename (old=new pairs) of an existing remote LFS file of the same size: CommitOperationCopy
  (no re-upload). - Any other new or changed file: CommitOperationAdd from the staged folder. - Remote files not in the
  staged folder: CommitOperationDelete. Afterwards remote must equal the staged folder: small files by git blob sha1,
  LFS files <= 300 MB by sha256 of the local file, bigger LFS files by size plus the sha256 they had before (copy source
  or unchanged path; those were SHA-256 verified at upload).
Usage: HB_TOKEN_FILE=... python hf_sync_commit.py <staged_repo_dir> <repo> [old=new ...] [--apply] [--msg "..."]
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationCopy, CommitOperationDelete, HfApi

args = sys.argv[1:]
msg = None
if "--msg" in args:
    i = args.index("--msg")
    msg = args[i + 1]
    del args[i:i + 2]
folder, name = Path(args[0]), args[1]
pairs = dict(a.split("=", 1) for a in args[2:] if "=" in a)
apply = "--apply" in args
repo = "humanoid-benchmark/" + name
api = HfApi(token=Path(os.environ["HB_TOKEN_FILE"]).read_text().strip())
SMALL_SHA = 300 * 1024 * 1024


def remote_tree():
    return {f.path: f for f in api.list_repo_tree(repo, recursive=True, expand=True) if hasattr(f, "size")}


local = {str(p.relative_to(folder)): p for p in sorted(folder.rglob("*"))
         if p.is_file() and ".cache" not in p.relative_to(folder).parts}


def git_sha1(p):
    data = p.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def old_path(new):
    for old, nw in pairs.items():
        if new.startswith(nw + "/"):
            return old + new[len(nw):]
    return None


def same(r, p):
    if r.size != p.stat().st_size:
        return False
    if r.lfs is None:
        return r.blob_id == git_sha1(p)
    if p.stat().st_size <= SMALL_SHA:
        return r.lfs.sha256 == sha256(p)
    return True  # large unchanged weight file: same path and size, SHA-256 verified at its upload


before = remote_tree()
ops, expect, n = [], {}, {"copy": 0, "add": 0, "delete": 0, "unchanged": 0}
for k, p in local.items():
    r = before.get(k)
    if r is not None and same(r, p):
        n["unchanged"] += 1
        if r.lfs is not None:
            expect[k] = r.lfs.sha256
        continue
    o = old_path(k)
    src = before.get(o) if o else None
    if r is None and src is not None and src.lfs is not None and src.size == p.stat().st_size:
        ops.append(CommitOperationCopy(src_path_in_repo=o, path_in_repo=k))
        expect[k] = src.lfs.sha256
        n["copy"] += 1
    else:
        if p.stat().st_size > SMALL_SHA:
            sys.exit(f"ABORT: {k} is large and has no copy source; use upload_large_folder")
        ops.append(CommitOperationAdd(path_in_repo=k, path_or_fileobj=str(p)))
        n["add"] += 1
for k in sorted(set(before) - set(local) - {".gitattributes"}):
    ops.append(CommitOperationDelete(path_in_repo=k))
    n["delete"] += 1
print("PLAN", repo, json.dumps(n), "renames", json.dumps(pairs))
for op in ops:
    kind = type(op).__name__.replace("CommitOperation", "")
    if kind != "Delete" or n["delete"] <= 40:
        print("  ", kind, getattr(op, "src_path_in_repo", ""), op.path_in_repo)
if not apply or not ops:
    sys.exit(0)
c = api.create_commit(repo, operations=ops, commit_message=msg or "Sync with staged folder")
print("COMMIT", c.oid)

after = remote_tree()
extra = sorted(set(after) - set(local) - {".gitattributes"})
missing = sorted(set(local) - set(after))
bad = []
for k, p in local.items():
    r = after.get(k)
    if r is None:
        continue
    if r.lfs is None or p.stat().st_size <= SMALL_SHA:
        ok = same(r, p)
    else:
        ok = r.size == p.stat().st_size and r.lfs.sha256 == expect.get(k)
    if not ok:
        bad.append(k)
info = api.model_info(repo)
print("EXTRA", json.dumps(extra))
print("MISSING", json.dumps(missing))
print("BAD", json.dumps(bad))
print("SYNC_OK" if not (extra or missing or bad) else "SYNC_MISMATCH", repo, "files", len(local), "private", info.private,
      "revision", info.sha)
