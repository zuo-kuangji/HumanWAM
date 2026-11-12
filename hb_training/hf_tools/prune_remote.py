"""Delete files from humanoid-benchmark/<repo> that are not in the local staged folder (one commit). Prints what it does."""
import json, os, sys
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationDelete
folder, name = Path(sys.argv[1]), sys.argv[2]
repo = "humanoid-benchmark/" + name
api = HfApi(token=Path(os.environ["HB_TOKEN_FILE"]).read_text().strip())
local = {str(p.relative_to(folder)) for p in folder.rglob("*") if p.is_file() and ".cache" not in p.relative_to(folder).parts}
remote = {f.path for f in api.list_repo_tree(repo, recursive=True) if hasattr(f, "size")}
extra = sorted(remote - local - {".gitattributes"})
missing = sorted(local - remote)
print("remote", len(remote), "local", len(local), "extra", len(extra), "missing", len(missing))
print("EXTRA", json.dumps(extra)); print("MISSING", json.dumps(missing))
if extra and "--apply" in sys.argv:
    c = api.create_commit(repo, operations=[CommitOperationDelete(path_in_repo=p) for p in extra],
                          commit_message="Remove training-only files (trainer state / training args / training plan)")
    print("DELETED_COMMIT", c.oid)
print("private", api.model_info(repo).private, "revision", api.model_info(repo).sha)
