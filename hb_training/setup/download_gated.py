"""One authorized HF session for gated/base assets of GR00T and Psi-Zero (token from stdin, in-process only).

Usage: printf '%s\n' "$TOKEN" | python download_gated.py <models dir>
Each item is retried on stalls (HF_HUB_DOWNLOAD_TIMEOUT) inside this process; results go to gated-verified.json.
"""
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

out = Path(sys.argv[1]).resolve()
token = sys.stdin.readline().rstrip("\r\n")
if not token.startswith("hf_"):
    raise SystemExit("token format rejected")
api = HfApi(endpoint="https://huggingface.co", token=token)
jobs = [
    ("nvidia/Cosmos-Reason2-2B", "9ce19a195e423419c349abfc86fd07178b230561", ["*.json", "*.jinja", "*.txt", "*.model"],
     "cosmos-reason2-assets"),
    ("USC-PSI-Lab/psi-model", "74d7abaa61b51dfe4e95f539a83738c276ce27b2",
     ["psi0/postpre.sonic1.1.unifolm.2609181726.40k/*"], "psi0"),
    ("Qwen/Qwen3-VL-2B-Instruct", None, None, "qwen3-vl-2b-instruct"),
]
results = []
for repo, rev, patterns, name in jobs:
    t0 = time.monotonic()
    item = {"repo": repo, "revision": rev, "dir": name}
    try:
        info = api.model_info(repo, revision=rev, files_metadata=True)
        item["revision"] = info.sha
        for attempt in range(1, 31):
            try:
                snapshot_download(repo, revision=info.sha, allow_patterns=patterns, local_dir=out / name, token=token,
                                  max_workers=8)
                break
            except Exception as error:  # stalls / transient network
                item["last_retry_error"] = f"{type(error).__name__}: {str(error)[:200]}"
                if "gated" in str(error).lower() or "401" in str(error) or "403" in str(error):
                    raise
                time.sleep(5)
        else:
            raise RuntimeError("download did not finish after 30 attempts")
        count, size = 0, 0
        for s in info.siblings:
            if patterns is not None and not any(Path(s.rfilename).match(p) or s.rfilename.startswith(p.rstrip("*"))
                                                for p in patterns):
                continue
            f = out / name / s.rfilename
            assert f.is_file(), s.rfilename
            if s.size is not None:
                assert f.stat().st_size == s.size, s.rfilename
            if s.lfs is not None:
                h = hashlib.sha256()
                with f.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1 << 22), b""):
                        h.update(chunk)
                assert h.hexdigest() == s.lfs.sha256, s.rfilename
            count += 1
            size += s.size or 0
        item.update(ok=True, files=count, bytes=size)
    except Exception as error:
        item.update(ok=False, error=f"{type(error).__name__}: {str(error)[:400]}")
        traceback.print_exc()
    item["seconds"] = round(time.monotonic() - t0, 1)
    results.append(item)
    print("ITEM", json.dumps(item), flush=True)
token = None
(out / "gated-verified.json").write_text(json.dumps(results, indent=2) + "\n")
print("GATED_DONE", json.dumps([(r["repo"], r["ok"]) for r in results]), flush=True)
