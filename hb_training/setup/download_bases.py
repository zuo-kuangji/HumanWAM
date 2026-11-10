"""Download pinned base weights for GR00T N1.7 (and Cosmos-Reason2 processor assets) to NAS.

Usage: [printf '%s\n' "$TOKEN" |] python download_bases.py <models dir> [--anonymous]
Token (optional) is read from stdin only when --anonymous is not given; it is used in-process only.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

out = Path(sys.argv[1]).resolve()
token = None
if "--anonymous" not in sys.argv:
    token = sys.stdin.readline().rstrip("\r\n") or None
    if token is not None and not token.startswith("hf_"):
        raise SystemExit("token format rejected")
jobs = [
    ("nvidia/GR00T-N1.7-3B", "2fc962b973bccdd5d8ce4f67cc63b264d6886495", None, "groot-n17-base"),
    ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561",
     ["*.json", "*.jinja", "*.txt", "*.model"], "cosmos-reason2-assets"),
]
api = HfApi(endpoint="https://huggingface.co", token=token)
receipt = {"anonymous": token is None, "items": []}
for repo, rev, patterns, name in jobs:
    t0 = time.monotonic()
    info = api.model_info(repo, revision=rev, files_metadata=True)
    target = out / name
    path = snapshot_download(repo, revision=rev, allow_patterns=patterns, local_dir=target, token=token, max_workers=8)
    files = {}
    for s in info.siblings:
        if patterns is not None and not any(Path(s.rfilename).match(p) for p in patterns):
            continue
        f = target / s.rfilename
        assert f.is_file(), s.rfilename
        if s.size is not None:
            assert f.stat().st_size == s.size, (s.rfilename, f.stat().st_size, s.size)
        if s.lfs is not None:
            h = hashlib.sha256()
            with f.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 22), b""):
                    h.update(chunk)
            assert h.hexdigest() == s.lfs.sha256, s.rfilename
        files[s.rfilename] = s.size
    receipt["items"].append(dict(repo=repo, revision=rev, path=str(path), files=len(files),
                                 bytes=sum(v or 0 for v in files.values()), gated=getattr(info, "gated", None),
                                 seconds=round(time.monotonic() - t0, 1)))
    print("DOWNLOADED", repo, len(files), flush=True)
token = None
(out / "bases-verified.json").write_text(json.dumps(receipt, indent=2) + "\n")
print("BASES_OK", json.dumps(receipt)[:600], flush=True)
