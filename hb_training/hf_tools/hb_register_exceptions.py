"""2026-11-24 user: register the node-layout exceptions (GR00T C6 b512, Psi-Zero A10 v3 / B1 / B8 / C6) in the published
protocol.json files. Adds a top-level "exceptions" list (REPRODUCIBILITY.md section 7); no other field changes.

  hb_register_exceptions.py stage    write the updated files to $P/hf-autoupload/register-exceptions-20261124/<repo>/...
  hb_register_exceptions.py commit   one commit per repo
"""
import io
import json
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "hf-autoupload" / "register-exceptions-20261124"
ORG = "humanoid-benchmark"
REG = "https://github.com/zuo-kuangji/HumanWAM/blob/bai/hb_training/REPRODUCIBILITY.md (section 7)"
api = HfApi(token=open("/root/.hb-secrets/hf_token").read().strip())

NO_INLOOP = {"item": "in-training MSE", "this_run": "none (trained before the in-training MSE hook of 2026-11-15)",
             "protocol": "open-loop MSE every 250 steps during training", "decision": "post-hoc open-loop MSE only",
             "registry": REG}
GROOT_C6 = [
    {"item": "node layout and gradient accumulation",
     "this_run": "4 nodes x 8 GPUs x 8 per GPU, gradient_accumulation_steps 2 (global batch 512), 8000 steps (2026-11-13 recipe)",
     "protocol": "2 nodes x 8 GPUs x 32 per GPU, gradient_accumulation_steps 1 (global batch 512)",
     "decision": "user 2026-11-14: not retrained; user 2026-11-24: layout registered as an exception, not retrained",
     "registry": REG},
    NO_INLOOP,
]
PSI_LAYOUT = {"item": "node layout",
              "this_run": "4 nodes x 8 GPUs x 16 per GPU (global batch 512, no gradient accumulation)",
              "protocol": "2 nodes x 8 GPUs x 32 per GPU (global batch 512)",
              "decision": "user 2026-11-24: registered as an exception, not retrained", "registry": REG}
PSI_C6_STEPS = {"item": "steps", "this_run": "4000 steps (not doubled for the long task C6)",
                "protocol": "long task C6: 2 x the normal steps", "decision": "user 2026-11-13: not retrained", "registry": REG}

# (repo, folder prefix, exceptions, check)
TARGETS = [
    ("groot", "c6-bathroom-door-fetch/b512/", GROOT_C6,
     lambda p: p.get("num_gpus") == 32 and p.get("per_gpu_batch") == 8 and p.get("gradient_accumulation_steps") == 2),
    ("psi0", "a10-trolley-push-v3/", [PSI_LAYOUT], lambda p: p.get("num_gpus") == 32 and p.get("per_gpu_batch") == 16),
    ("psi0", "b1-riceball-microwave/", [PSI_LAYOUT], lambda p: p.get("num_gpus") == 32 and p.get("per_gpu_batch") == 16),
    ("psi0", "b8-table-wipe/", [PSI_LAYOUT], lambda p: p.get("num_gpus") == 32 and p.get("per_gpu_batch") == 16),
    ("psi0", "c6-bathroom-door-fetch/", [PSI_LAYOUT, PSI_C6_STEPS, NO_INLOOP],
     lambda p: p.get("num_gpus") == 32 and p.get("per_gpu_batch") == 16),
]


def get(repo, path):
    for i in range(6):
        try:
            return Path(hf_hub_download(f"{ORG}/{repo}", path, token=api.token, force_download=True))
        except Exception:  # noqa: BLE001
            if i == 5:
                raise
            time.sleep(10 * (i + 1))


def stage():
    files = {r: api.list_repo_files(f"{ORG}/{r}") for r in sorted({t[0] for t in TARGETS})}
    n = 0
    for repo, prefix, exc, ok in TARGETS:
        protos = sorted(f for f in files[repo] if f.startswith(prefix) and f.endswith("/protocol.json")
                        and f[len(prefix):].count("/") == 1 and f[len(prefix):].startswith("step-"))
        for f in protos:
            p = json.loads(get(repo, f).read_text())
            if not ok(p):
                print("SKIP layout mismatch", repo, f, p.get("num_gpus"), p.get("per_gpu_batch"), p.get("gradient_accumulation_steps"))
                continue
            new = dict(p)
            cur = list(p.get("exceptions") or [])
            for e in exc:
                if e not in cur:
                    cur.append(e)
            new["exceptions"] = cur
            assert all(new[k] == p[k] for k in p if k != "exceptions")
            dst = OUT / repo / f
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(new, indent=2, ensure_ascii=False, default=str) + "\n")
            print("STAGE", repo, f, "exceptions:", [e["item"] for e in cur], flush=True)
            n += 1
    print("STAGED", n, flush=True)


def commit():
    for repo in sorted({t[0] for t in TARGETS}):
        root = OUT / repo
        paths = sorted(root.rglob("protocol.json"))
        ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(root)), path_or_fileobj=io.BytesIO(p.read_bytes())) for p in paths]
        c = api.create_commit(f"{ORG}/{repo}", operations=ops,
                              commit_message="protocol.json: register node-layout exceptions (user 2026-11-24)",
                              commit_description="Adds a top-level `exceptions` list, see hb_training/REPRODUCIBILITY.md "
                                                 "section 7 of HumanWAM bai. No other field and no other file changed.")
        print("COMMIT", repo, len(ops), getattr(c, "oid", c), flush=True)


if __name__ == "__main__":
    {"stage": stage, "commit": commit}[sys.argv[1]]()
