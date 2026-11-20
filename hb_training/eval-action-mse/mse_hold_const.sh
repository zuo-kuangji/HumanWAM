#!/bin/bash
# hb_relay cmd item (2026-11-20 user: constant action dims removed from every nMSE; re-evaluate the 42 affected checkpoints).
# Idle-slot item like mse_hold.sh: this script mentions eval_queue.py, so the per-node eval queue (eval-action-mse/eval_queue.py)
# treats the node as free while this item runs and fills all 8 GPUs with pending MSE jobs (jobs-const-20261120.txt is at the
# top of jobs.txt). Releases when every job of jobs-const-20261120.txt has a result (.json or .failed) and none is running,
# or after at most 60 min (own marker RELEASE-CONST-20261120; the old RELEASE-RELAY marker of 11-16 is not used).
D=$(dirname "$(realpath "$0")"); M=$D/RELEASE-CONST-20261120
echo "MSE_HOLD_CONST start $(date -u) $(hostname)"
drained() {
  python3 - "$D" <<'EOF'
import sys
from pathlib import Path
d = Path(sys.argv[1]); r = d / "results"
ids = [l.split()[0] for l in (d / "jobs-const-20261120.txt").read_text().splitlines() if l.split("#", 1)[0].strip()]
left = [i for i in ids if not (r / f"{i}.json").exists() and not (r / f"{i}.failed").exists()]
running = list(r.glob("RUNNING-*.json"))
print(f"{len(left)} const jobs left, {len(running)} running")
sys.exit(0 if not left and not running else 1)
EOF
}
n=0
while [ ! -e "$M" ]; do
  sleep 60; n=$((n + 1))
  if [ $((n % 2)) = 0 ] && drained; then echo "MSE_HOLD_CONST drained $(date -u)"; touch "$M"; fi
  if [ "$n" -ge 60 ]; then echo "MSE_HOLD_CONST 60 min cap $(date -u)"; touch "$M"; fi
done
echo "MSE_HOLD_CONST released $(date -u) $(hostname)"
exit 0
