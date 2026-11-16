#!/bin/bash
# hb_relay cmd item (2026-11-16 user): "当前这个FastWAM的训练一结束，就先暂停所有训练，立刻开始把之前所有缺失的MSE都跑出来".
# Idle-slot item: this script mentions eval_queue.py, so the per-node eval queue (eval-action-mse/eval_queue.py) treats
# the node as free while this item runs and fills all 8 GPUs with pending MSE jobs (jobs.txt).
# The item ends (training resumes with the next relay item) when eval-action-mse/RELEASE-RELAY exists, which is created
# by hand or automatically here once every job in jobs.txt has results/<id>.json or results/<id>.failed and no eval is
# running (GPUs never idle), or after at most 4 h. Each node checks independently; node 0 writes the RELEASE-RELAY marker.
D=$(dirname "$(realpath "$0")")
echo "MSE_HOLD start $(date -u) $(hostname)"
drained() {
  python3 - "$D" <<'EOF'
import sys
from pathlib import Path
d = Path(sys.argv[1]); r = d / "results"
ids = [l.split()[0] for l in (d / "jobs.txt").read_text().splitlines() if l.split("#", 1)[0].strip()]
left = [i for i in ids if not (r / f"{i}.json").exists() and not (r / f"{i}.failed").exists()]
running = list(r.glob("RUNNING-*.json"))
if not (d / "smoke-fw" / "DONE").exists():  # FastWAM in-loop GPU smoke (smoke_fw_inloop.sh) must finish first
    print("waiting for smoke-fw/DONE"); sys.exit(1)
print(f"{len(left)} jobs left, {len(running)} running")
sys.exit(0 if not left and not running else 1)
EOF
}
n=0
while [ ! -e "$D/RELEASE-RELAY" ]; do
  sleep 60; n=$((n + 1))
  if [ $((n % 5)) = 0 ] && drained; then
    echo "MSE_HOLD backlog drained $(date -u)"; touch "$D/RELEASE-RELAY"
  fi
  if [ "$n" -ge 240 ]; then  # safety cap: 4 h, the rest then runs in relay gaps
    echo "MSE_HOLD 4 h cap reached $(date -u)"; touch "$D/RELEASE-RELAY"
  fi
done
echo "MSE_HOLD released $(date -u) $(hostname)"
exit 0
