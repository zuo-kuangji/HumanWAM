"""Final gate before deleting the two old Delta Flow repos (user approved 2026-11-11: delete once the new repo is verified).
Deletes ONLY if every check passes:
  1. NAS backup manifest: every file of both old repos verified (no bad files).
  2. upload status of humanoid-benchmark/deltaflow is HF_VERIFIED with no bad files, repo is public.
  3. every weight file (> 1 GB) in each old repo has a model.pt in deltaflow with the same LFS sha256 and size,
     at the path named by that folder's protocol.json "moved_from".
Usage: HB_TOKEN_FILE=... python final_check_delete.py $P [--delete]
"""
import json
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

P = Path(sys.argv[1])
ORG = "humanoid-benchmark"
OLD = ["g1-basket-transfer-imagewam-deltaflow76", "b6-imagewam-deltaflow76"]
api = HfApi(token=Path(os.environ["HB_TOKEN_FILE"]).read_text().strip())
problems = []

man = json.loads((P / "hf-archive-deltaflow-legacy-20261111/manifest.json").read_text())
for r in OLD:
    if man[r]["bad"]:
        problems.append(f"backup bad files in {r}: {man[r]['bad']}")
st = json.loads((P / "hf-models-20261111-deltaflow-status.json").read_text()).get("deltaflow", {})
if st.get("phase") != "HF_VERIFIED" or st.get("bad"):
    problems.append(f"upload status {st}")
info = api.model_info(f"{ORG}/deltaflow")
if info.private:
    problems.append("deltaflow repo is private")
new = {f.path: f for f in api.list_repo_tree(f"{ORG}/deltaflow", recursive=True, expand=True) if hasattr(f, "size")}
stage = P / "hf-models-20261111-deltaflow/deltaflow"
moved = {}
for proto in stage.glob("legacy/**/protocol.json"):
    d = json.loads(proto.read_text())
    mf = d["moved_from"]
    newpath = str(proto.parent.relative_to(stage) / "model.pt")
    moved[(mf["repo"].split("/")[1], mf["path"])] = newpath
for r in OLD:
    old = {f.path: f for f in api.list_repo_tree(f"{ORG}/{r}", recursive=True, expand=True) if hasattr(f, "size")}
    for k, f in old.items():
        if f.lfs is None or f.size < 10 ** 9:
            continue
        np_ = moved.get((r, k))
        nf = new.get(np_) if np_ else None
        ok = nf is not None and nf.lfs is not None and nf.lfs.sha256 == f.lfs.sha256 == man[r]["lfs_sha256"].get(k) \
            and nf.size == f.size
        print("WEIGHT", r, k, "->", np_, "OK" if ok else "MISMATCH")
        if not ok:
            problems.append(f"weight {r}/{k}")
print("PROBLEMS", json.dumps(problems))
if problems:
    print("NOT_DELETING")
    sys.exit(1)
print("ALL_CHECKS_PASSED", "deltaflow revision", info.sha)
if "--delete" in sys.argv:
    for r in OLD:
        api.delete_repo(f"{ORG}/{r}", repo_type="model")
        print("DELETED", f"{ORG}/{r}")
    left = [m.id for m in api.list_models(author=ORG) if m.id.split("/")[1] in OLD]
    print("REMAINING_OLD", left)
