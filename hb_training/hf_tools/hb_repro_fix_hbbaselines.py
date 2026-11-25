"""2026-11-24: correct the reproducibility block of the hb_baselines folders in humanoid-benchmark/act.

The 2026-11-23 backfill described the 12 hb_baselines ACT folders (<task>/b512*/step-*) as the Psi0 LeRobot ACT
(seed source, environment, hardware, upstream code, proprio-bias seed). Their seed (0) was read correctly from
config.json; this rewrites only the "seed" / "reproducibility" fields that the backfill added, from config.json.

  hb_repro_fix_hbbaselines.py stage    write corrected files to $P/hf-autoupload/repro-fix-hbbaselines/act/...
  hb_repro_fix_hbbaselines.py commit   upload them in one commit
"""
import io
import json
import sys
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hb_repro as R  # noqa: E402

ORG, REPO = "humanoid-benchmark", "act"
OUT = HERE.parent / "hf-autoupload" / "repro-fix-hbbaselines" / REPO
NOTE = ("2026-11-23 (fields added after publication; nothing else changed); corrected 2026-11-24: this is an "
        "hb_baselines run, the first backfill described it as the Psi0 LeRobot ACT")
api = HfApi(token=open("/root/.hb-secrets/hf_token").read().strip())


def get(path):
    for i in range(6):
        try:
            return Path(hf_hub_download(f"{ORG}/{REPO}", path, token=api.token, force_download=True))
        except Exception as e:  # noqa: BLE001
            if i == 5:
                raise
            time.sleep(10 * (i + 1))


def stage():
    files = api.list_repo_files(f"{ORG}/{REPO}")
    n = 0
    for f in sorted(x for x in files if x.endswith("/protocol.json") and not x.startswith("legacy/")):
        proto = json.loads(get(f).read_text())
        if not R.is_hb_baselines(proto):
            continue
        folder = f.rpartition("/")[0]
        cfg = json.loads(get(f"{folder}/config.json").read_text())
        seed = (cfg.get("training") or {}).get("seed")
        new = {k: v for k, v in proto.items() if k not in ("seed", "reproducibility")}
        new.update(R.fields(REPO, new, seed_found=seed if isinstance(seed, int) else None, run_config=cfg, backfill=NOTE))
        dst = OUT / f
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(new, indent=2, ensure_ascii=False, default=str) + "\n")
        kept = [k for k in proto if k not in ("seed", "reproducibility")]
        assert all(new[k] == proto[k] for k in kept), f
        rp = new["reproducibility"]
        print("FIX", f, "seed", new["seed"], "| verified", rp["training_seed_verified_from_run_files"], "|",
              rp["upstream_code"], "| layout", rp["layout"], flush=True)
        n += 1
    print("STAGED", n, flush=True)


def commit():
    paths = sorted(OUT.rglob("protocol.json"))
    ops = [CommitOperationAdd(path_in_repo=str(p.relative_to(OUT)), path_or_fileobj=io.BytesIO(p.read_bytes())) for p in paths]
    c = api.create_commit(f"{ORG}/{REPO}", operations=ops,
                          commit_message="protocol.json: correct the reproducibility block of hb_baselines folders",
                          commit_description="The 2026-11-23 backfill described these hb_baselines ACT runs as the Psi0 "
                                             "LeRobot ACT. Seed (0), source, upstream code and layout are now taken from "
                                             "config.json of each folder. No other field and no other file changed.")
    print("COMMIT", REPO, len(ops), getattr(c, "oid", c), flush=True)


if __name__ == "__main__":
    {"stage": stage, "commit": commit}[sys.argv[1]]()
