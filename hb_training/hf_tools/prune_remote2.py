"""Delete files from humanoid-benchmark/<repo> that are not in the local staged folder (one commit), then check that
remote == local. Usage: HB_TOKEN_FILE=... python prune_remote2.py <staged_repo_dir> <repo> "<commit message>" [--apply]"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import CommitOperationDelete, HfApi

folder, name, msg = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
repo = "humanoid-benchmark/" + name
api = HfApi(token=Path(os.environ["HB_TOKEN_FILE"]).read_text().strip())


def state():
    local = {str(p.relative_to(folder)) for p in folder.rglob("*")
             if p.is_file() and ".cache" not in p.relative_to(folder).parts}
    remote = {f.path for f in api.list_repo_tree(repo, recursive=True) if hasattr(f, "size")}
    return local, remote


local, remote = state()
extra = sorted(remote - local - {".gitattributes"})
missing = sorted(local - remote)
print("remote", len(remote), "local", len(local), "extra", len(extra), "missing", len(missing))
print("EXTRA", json.dumps(extra))
print("MISSING", json.dumps(missing))
if missing:
    sys.exit("ABORT: local files missing on HF")
if extra and "--apply" in sys.argv:
    c = api.create_commit(repo, operations=[CommitOperationDelete(path_in_repo=p) for p in extra], commit_message=msg)
    print("DELETED_COMMIT", c.oid)
    local, remote = state()
    assert remote - {".gitattributes"} == local, "remote != local after prune"
info = api.model_info(repo)
print("PRUNE_OK", repo, "private", info.private, "revision", info.sha)
