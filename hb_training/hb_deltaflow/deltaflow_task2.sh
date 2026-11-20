#!/bin/bash
# ImageWAM Delta-flow, one task, 40 epochs, 100 episodes, augmentation v2, 4 nodes x 8 A800.
# Run the SAME command on all 4 nodes at once (hb_relay does that); DLC env RANK/WORLD_SIZE/MASTER_ADDR required.
#   bash deltaflow_task.sh <b6|a3|a7|a18> [master_port_base]
# - already verified complete -> exit 0 immediately;
# - not prepared yet -> node 0 runs prepare_deltaflow.py <task> (CPU), the other nodes wait for it;
# - training through deltaflow_supervise.py (3 attempts, resume from the newest saved state);
# - exit 0 only after deltaflow_check_final.py verified the epoch 10/20/30/40 weights and the final state.
set -uo pipefail
TASK=${1:?usage: deltaflow_task.sh <b6|a3|a7|a18|trolley|basket|basketv2|a17|b7|a6|c6|a1|a14> [master_port_base]}
case $TASK in
  b6) DEFAULT_PORT=29800 ;;
  a3) DEFAULT_PORT=29830 ;;
  a7) DEFAULT_PORT=29860 ;;
  a18) DEFAULT_PORT=29890 ;;
  trolley) DEFAULT_PORT=29920 ;;
  basket) DEFAULT_PORT=29950 ;;
  basketv2) DEFAULT_PORT=30160 ;;
  a7v2) DEFAULT_PORT=30190 ;;
  b8) DEFAULT_PORT=30220 ;;
  a18c2) DEFAULT_PORT=30250 ;;
  a17c2) DEFAULT_PORT=30280 ;;
  a14c2) DEFAULT_PORT=30310 ;;
  b1) DEFAULT_PORT=30340 ;;
  a10v3) DEFAULT_PORT=30370 ;;
  a17lg) DEFAULT_PORT=30400 ;;
  a14lg) DEFAULT_PORT=30430 ;;
  b6v6) DEFAULT_PORT=30460 ;;
  b5) DEFAULT_PORT=30490 ;;
  a7v4) DEFAULT_PORT=30940 ;;
  a17cr2lg) DEFAULT_PORT=30970 ;;
  a3v4) DEFAULT_PORT=31000 ;;
  a17) DEFAULT_PORT=29980 ;;
  b7) DEFAULT_PORT=30010 ;;
  b7t) DEFAULT_PORT=31030 ;;
  b5ss) DEFAULT_PORT=31060 ;;  # 2026-11-20 copy (deltaflow_task.sh was running): B5 standing start
  b7gt) DEFAULT_PORT=31090 ;;  # 2026-11-20: B7 green trimmed v2, all 421  # 2026-11-17 user: B7 black trimmed data
  a6) DEFAULT_PORT=30040 ;;
  c6) DEFAULT_PORT=30070 ;;
  a1) DEFAULT_PORT=30100 ;;
  a14) DEFAULT_PORT=30130 ;;
  *) echo "deltaflow_task: unsupported task '$TASK'"; exit 64 ;;
esac
PORT_BASE=${2:-$DEFAULT_PORT}
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
P=$(dirname "$G")
PY=$P/venv/bin/python
ROOT=$G/deltaflow-runs
R=$ROOT/$TASK
NODE=${RANK:?DLC RANK missing}
[ "${WORLD_SIZE:-}" = 4 ] || { echo "deltaflow_task: WORLD_SIZE=${WORLD_SIZE:-unset}, need 4"; exit 64; }
[ -n "${MASTER_ADDR:-}" ] || { echo "deltaflow_task: MASTER_ADDR missing"; exit 64; }
mkdir -p "$ROOT/.sessions" "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/$TASK-node-$NODE.log") 2>&1
echo "$(date -u) deltaflow $TASK node $NODE host $(hostname) start"

if [ -f "$R/deltaflow-complete.json" ] && "$PY" "$H/deltaflow_check_final.py" "$R"; then
  echo "$(date -u) deltaflow $TASK already complete"; exit 0
fi

# Per-node invocation counter = session id. The relay runs every item once on all 4 nodes, so the counters agree;
# if they ever disagree the NAS barriers time out and the item fails (no silent mixing of runs).
SESSION=$(( $(cat "$ROOT/.sessions/$TASK-node-$NODE" 2>/dev/null || echo 0) + 1 ))
echo "$SESSION" > "$ROOT/.sessions/$TASK-node-$NODE"
echo "session $SESSION port_base $PORT_BASE"

if [ ! -f "$R/prepared.ok" ]; then
  MARK=$ROOT/.sessions/$TASK-prepare-s$SESSION.rc
  if [ "$NODE" = 0 ]; then
    echo "$(date -u) preparing $TASK on node 0 (CPU)"
    "$PY" "$H/prepare_deltaflow.py" "$TASK"; prc=$?
    echo "$prc" > "$MARK"
  else
    echo "$(date -u) waiting for node 0 to prepare $TASK"
    for _ in $(seq 1080); do [ -f "$MARK" ] && break; sleep 10; done   # up to 3 h
  fi
  if [ ! -f "$R/prepared.ok" ]; then echo "deltaflow_task: preparation of $TASK failed (rc $(cat "$MARK" 2>/dev/null))"; exit 66; fi
fi

"$PY" "$H/deltaflow_supervise.py" "$R" "$SESSION" "$PORT_BASE"
rc=$?
if [ "$rc" -ne 0 ]; then echo "$(date -u) deltaflow $TASK training failed rc=$rc"; exit "$rc"; fi
# give NAS metadata a moment to converge across nodes, then verify on every node
for i in 1 2 3 4 5 6; do
  "$PY" "$H/deltaflow_check_final.py" "$R" $([ "$NODE" = 0 ] && echo --write) && { echo "$(date -u) deltaflow $TASK done"; exit 0; }
  sleep 30
done
echo "$(date -u) deltaflow $TASK: training exited 0 but the final checkpoint check failed"
exit 70
