"""Parallel HTTP range downloader (GitHub release assets are ~20-70 KB/s per connection from the DLC nodes).

Usage: python pdl.py <url> <out> [parts=32]
Each part re-requests the original URL (GitHub re-signs the redirect), resumes on retry, then parts are joined.
"""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

url, out = sys.argv[1], sys.argv[2]
parts = int(sys.argv[3]) if len(sys.argv) > 3 else 32


def total_size():
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "hb-pdl"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Range"].split("/")[-1])


size = total_size()
step = -(-size // parts)
print("size", size, "parts", parts, flush=True)


def fetch(i):
    lo, hi = i * step, min(size, (i + 1) * step) - 1
    path = f"{out}.part{i:03d}"
    for attempt in range(200):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have >= hi - lo + 1:
            return i
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo + have}-{hi}", "User-Agent": "hb-pdl"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while True:
                    b = r.read(1 << 16)
                    if not b:
                        break
                    f.write(b)
        except Exception as e:  # noqa: BLE001
            print("retry", i, attempt, type(e).__name__, flush=True)
            time.sleep(3)
    raise RuntimeError(f"part {i} failed")


t0 = time.time()
with ThreadPoolExecutor(parts) as ex:
    for i in ex.map(fetch, range(parts)):
        pass
with open(out + ".tmp", "wb") as f:
    for i in range(parts):
        with open(f"{out}.part{i:03d}", "rb") as p:
            f.write(p.read())
assert os.path.getsize(out + ".tmp") == size
os.replace(out + ".tmp", out)
for i in range(parts):
    os.remove(f"{out}.part{i:03d}")
print("PDL_OK", out, size, f"{time.time() - t0:.0f}s", flush=True)
