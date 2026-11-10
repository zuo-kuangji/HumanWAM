"""Anonymous downloads of public base assets (no token): Psi-Zero init, Qwen3-VL-2B, OpenPI pi05_base (GCS).

Usage: python fetch_public.py <models dir>
Retries stalled transfers; verifies HF LFS sha256 and GCS md5/size; writes public-verified.json.
"""
import base64
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

out = Path(sys.argv[1]).resolve()
api = HfApi(endpoint="https://huggingface.co")
results = []


def hf(repo, rev, patterns, name):
    t0 = time.monotonic()
    info = api.model_info(repo, revision=rev, files_metadata=True)
    for attempt in range(40):
        try:
            snapshot_download(repo, revision=info.sha, allow_patterns=patterns, local_dir=out / name, max_workers=8)
            break
        except Exception as error:
            print("RETRY", repo, attempt, type(error).__name__, str(error)[:160], flush=True)
            time.sleep(5)
    n = 0
    for s in info.siblings:
        if patterns and not any(s.rfilename.startswith(p.rstrip("*")) for p in patterns):
            continue
        f = out / name / s.rfilename
        assert f.is_file() and (s.size is None or f.stat().st_size == s.size), s.rfilename
        if s.lfs is not None:
            h = hashlib.sha256()
            with f.open("rb") as stream:
                for c in iter(lambda: stream.read(1 << 22), b""):
                    h.update(c)
            assert h.hexdigest() == s.lfs.sha256, s.rfilename
        n += 1
    results.append(dict(source="hf", repo=repo, revision=info.sha, dir=name, files=n,
                        seconds=round(time.monotonic() - t0, 1)))
    print("OK", repo, n, flush=True)


def gcs(bucket, prefix, name):
    t0 = time.monotonic()
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size,md5Hash),nextPageToken"}
        if token:
            q["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?" + urllib.parse.urlencode(q)
        data = json.loads(urllib.request.urlopen(url, timeout=60).read())
        items += data.get("items", [])
        token = data.get("nextPageToken")
        if not token:
            break
    total = 0
    for it in items:
        rel = it["name"][len(prefix):].lstrip("/")
        if not rel or it["name"].endswith("/"):
            continue
        f = out / name / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        size = int(it["size"])
        want = base64.b64decode(it["md5Hash"]).hex() if it.get("md5Hash") else None
        for attempt in range(40):
            if f.is_file() and f.stat().st_size == size:
                if want is None or hashlib.md5(f.read_bytes()).hexdigest() == want:
                    break
            try:
                src = f"https://storage.googleapis.com/{bucket}/" + urllib.parse.quote(it["name"])
                tmp = f.with_suffix(f.suffix + ".part")
                with urllib.request.urlopen(src, timeout=120) as r, tmp.open("wb") as w:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        w.write(b)
                tmp.replace(f)
            except Exception as error:
                print("RETRY", rel, attempt, type(error).__name__, flush=True)
                time.sleep(5)
        else:
            raise RuntimeError(f"failed {rel}")
        total += size
    results.append(dict(source="gcs", bucket=bucket, prefix=prefix, dir=name, files=len(items), bytes=total,
                        seconds=round(time.monotonic() - t0, 1)))
    print("OK gcs", prefix, len(items), total, flush=True)


hf("USC-PSI-Lab/psi-model", "74d7abaa61b51dfe4e95f539a83738c276ce27b2",
   ["psi0/postpre.sonic1.1.unifolm.2609181726.40k/"], "psi0")
hf("Qwen/Qwen3-VL-2B-Instruct", None, None, "qwen3-vl-2b-instruct")
gcs("openpi-assets", "checkpoints/pi05_base/", "openpi/pi05_base")
gcs("big_vision", "paligemma_tokenizer.model", "openpi/big_vision")
(out / "public-verified.json").write_text(json.dumps(results, indent=2) + "\n")
print("PUBLIC_DONE", json.dumps(results)[:800], flush=True)
