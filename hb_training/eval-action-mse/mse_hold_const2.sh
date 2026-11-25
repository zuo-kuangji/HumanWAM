#!/bin/bash
# hb_relay cmd item, 2026-11-20 20:2x: second try of the const-dim nMSE re-evaluation (42 checkpoints).
# The first try (mse_hold_const.sh, item 98) idled 60 min: only master-0 still had an eval_queue.py running (the worker
# queues exited 11-16) and it counted the FM-DiT upload daemon (hb_fmdit/fmdit_upload.py, psi0 env python) as training.
# Fix: fmdit_upload.py is in IGNORE_BASE now, and this item starts eval_queue.py on its own node if none is running.
# Idle-slot item (this text mentions eval_queue.py). Releases when every job of jobs-const-20261120.txt has a result and
# none is running, or after 60 min (marker RELEASE-CONST2-20261120).
D=$(dirname "$(realpath "$0")"); M=$D/RELEASE-CONST2-20261120
echo "MSE_HOLD_CONST2 start $(date -u) $(hostname)"
if ! pgrep -f "python3 eval_queue.py" >/dev/null; then
  (cd "$D" && setsid nohup python3 eval_queue.py >> "$D/logs/queue-$(hostname).stdout" 2>&1 < /dev/null &)
  echo "MSE_HOLD_CONST2 started eval_queue.py on $(hostname)"
fi
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
  if [ $((n % 2)) = 0 ] && drained; then echo "MSE_HOLD_CONST2 drained $(date -u)"; touch "$M"; fi
  if [ "$n" -ge 60 ]; then echo "MSE_HOLD_CONST2 60 min cap $(date -u)"; touch "$M"; fi
done
echo "MSE_HOLD_CONST2 released $(date -u) $(hostname)"
exit 0
