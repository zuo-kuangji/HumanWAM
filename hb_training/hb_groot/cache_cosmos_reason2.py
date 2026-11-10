"""Put the full gated nvidia/Cosmos-Reason2-2B (pinned 9ce19a1) into the shared HF hub cache $G/hf-home/hub.

GR00T N1.7 builds its backbone with Qwen3VLForConditionalGeneration.from_pretrained("nvidia/Cosmos-Reason2-2B") and
its processor with Qwen3VLProcessor.from_pretrained(same name) before loading the GR00T checkpoint, so every node needs
the whole repo resolvable by name. groot_node.sh runs with HF_HUB_OFFLINE=1, so nodes read this cache and never need
the token. refs/main is pinned to the verified revision.

Usage (token from stdin, kept in-process only):
  $G/envs/groot-py312/bin/python cache_cosmos_reason2.py $G/hf-home/hub < /root/.hb-secrets/hf_token
"""
import hashlib
import json
import os
import sys
import time

from huggingface_hub import HfApi, snapshot_download

REPO = "nvidia/Cosmos-Reason2-2B"
REV = "9ce19a195e423419c349abfc86fd07178b230561"
cache = os.path.abspath(sys.argv[1])
token = sys.stdin.readline().strip()
if not token.startswith("hf_"):
    raise SystemExit("token format rejected")
api = HfApi(endpoint="https://huggingface.co", token=token)
info = api.model_info(REPO, revision=REV, files_metadata=True)
assert info.sha == REV, info.sha
t0 = time.monotonic()
for attempt in range(1, 21):
    try:
        path = snapshot_download(REPO, revision=REV, cache_dir=cache, token=token, max_workers=8)
        break
    except Exception as error:  # transient stalls
        if any(k in str(error).lower() for k in ("gated", "401", "403")):
            raise
        print(f"retry {attempt}: {type(error).__name__}: {str(error)[:200]}", flush=True)
        time.sleep(10)
else:
    raise SystemExit("download did not finish")
token = None
count, size = 0, 0
for s in info.siblings:
    f = os.path.join(path, s.rfilename)
    assert os.path.isfile(f), s.rfilename
    if s.size is not None:
        assert os.path.getsize(f) == s.size, s.rfilename
    if s.lfs is not None:
        h = hashlib.sha256()
        with open(f, "rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 22), b""):
                h.update(chunk)
        assert h.hexdigest() == s.lfs.sha256, s.rfilename
    count += 1
    size += s.size or 0
refs = os.path.join(cache, "models--" + REPO.replace("/", "--"), "refs")
os.makedirs(refs, exist_ok=True)
with open(os.path.join(refs, "main"), "w") as f:
    f.write(REV)
print("CACHE_OK", json.dumps({"repo": REPO, "revision": REV, "snapshot": path, "files": count, "bytes": size,
                              "seconds": round(time.monotonic() - t0, 1)}), flush=True)
