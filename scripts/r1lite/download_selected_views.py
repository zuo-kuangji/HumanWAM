"""Download and verify the pinned two-camera R1Lite dataset subset."""
import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "at237299966/r1lite_place_multi_cups"
REVISION = "935cd0d7f397573616435d245f558ab160da27c5"
GROUPS = ["meta", "data/chunk-000", "videos/chunk-000/observation.images.head_rgb",
          "videos/chunk-000/observation.images.right_wrist_rgb"]


def get_json(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except (OSError, ValueError):
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    global REPO, REVISION
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--expected-files", type=int, default=334)
    args = parser.parse_args()
    REPO, REVISION = args.repo, args.revision
    dest = args.destination
    dest.mkdir(parents=True, exist_ok=True)
    info = get_json(f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}")
    assert info["sha"] == REVISION
    print(json.dumps({"status": "DOWNLOAD_START", "repo": REPO, "revision": REVISION,
                      "destination": str(dest), "groups": GROUPS}), flush=True)
    snapshot_download(repo_id=REPO, repo_type="dataset", revision=REVISION,
                      local_dir=str(dest), allow_patterns=[f"{g}/*" for g in GROUPS],
                      max_workers=6)
    checked = []
    for group in GROUPS:
        entries = get_json(f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/{group}?recursive=false&expand=false")
        for entry in entries:
            if entry["type"] != "file":
                continue
            path = dest / entry["path"]
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise RuntimeError(f"Missing or wrong size: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            expected = entry.get("lfs", {}).get("oid")
            if expected and digest.hexdigest() != expected:
                raise RuntimeError(f"SHA256 mismatch: {path}")
            checked.append({"path": entry["path"], "bytes": entry["size"],
                            "sha256": digest.hexdigest(), "lfs_sha256": expected})
    assert len(checked) == args.expected_files, f"Expected {args.expected_files} selected files, got {len(checked)}"
    manifest = {"status": "VERIFIED", "repo": REPO, "revision": REVISION,
                "files": checked, "count": len(checked),
                "total_bytes": sum(x["bytes"] for x in checked)}
    audit = dest / "imagewam_training_meta"
    audit.mkdir(exist_ok=True)
    temporary = audit / f"download_manifest.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(audit / "selected_views_download_manifest.json")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    main()
