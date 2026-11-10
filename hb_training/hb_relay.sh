#!/bin/bash
# Per-node training relay (run once on EVERY node; RANK=0..3 from the DLC env):
#   setsid nohup bash $G/hb_relay.sh > /dev/null 2>&1 &
# 1. stops the old B6 Delta Flow 60-epoch run on this node (user decision 2026-11-10: discard it),
# 2. then works through $G/relay/queue.txt in order, one item at a time, with a 4-node NAS barrier between items so
#    the nodes never drift apart. Items: "pi05 <task>", "psi0 <task>", "cmd <script>" (script run on every node).
#    The queue can be appended to while the relay runs; at the end of the queue it polls every 60 s.
# Position per node: $G/relay/pos-node-<n> (restart-safe). Logs: $G/relay/node-<n>.log.
G=$(dirname "$(realpath "$0")")
R=$G/relay
Q=$R/queue.txt
NODE=${RANK:?DLC RANK missing}
mkdir -p "$R"
exec >> "$R/node-$NODE.log" 2>&1
exec 9> "$R/lock-node-$NODE"
flock -n 9 || { echo "$(date -u) relay already running on node $NODE"; exit 1; }
echo "$(date -u) relay start node $NODE host $(hostname)"

busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }

stop_deltaflow() {
  local pat='b6-v5t7-ready-start-60ep-20261109/(train/launch|chain_b6v5t7)|train_b6v5t7\.py'
  local i
  for i in $(seq 60); do
    pkill -f "$pat"
    [ "$i" -gt 10 ] && pkill -9 -f "$pat"
    sleep 5
    if ! pgrep -f "$pat" > /dev/null && ! busy; then echo "$(date -u) old Delta Flow stopped, GPUs free"; return 0; fi
  done
  echo "$(date -u) WARNING: GPUs still busy after stopping Delta Flow:"
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
}

wait_gpu_free() {
  local t=0
  while busy; do
    [ $((t % 600)) = 0 ] && { echo "$(date -u) waiting for free GPUs"; nvidia-smi --query-compute-apps=pid,process_name --format=csv; }
    sleep 20; t=$((t + 20))
  done
}

barrier() {  # all 4 nodes finished item $1
  local n
  for n in 0 1 2 3; do
    until [ -f "$R/item-$1-done-node-$n" ]; do sleep 20; done
  done
}

stop_deltaflow
pos=$(cat "$R/pos-node-$NODE" 2>/dev/null || echo 0)
while true; do
  mapfile -t L < <(grep -v -E '^[[:space:]]*(#|$)' "$Q" 2>/dev/null)
  if [ "$pos" -ge "${#L[@]}" ]; then sleep 60; continue; fi
  read -r kind arg _ <<< "${L[$pos]}"
  k=$pos
  [ "$k" = 0 ] || barrier $((k - 1))
  wait_gpu_free
  echo "$(date -u) item $k start: $kind $arg"
  case $kind in
    pi05) HB_BASE_PORT=$((30000 + 10 * k)) bash "$G/hb_pi05/pi05_chain.sh" hb1110 "$arg"; rc=$? ;;
    psi0) bash "$G/hb_psi0/psi0_chain.sh" hb1110 $((30005 + 10 * k)) "$arg"; rc=$? ;;
    cmd)  bash "$arg"; rc=$? ;;
    *)    echo "unknown item kind $kind"; rc=99 ;;
  esac
  echo "$(date -u) item $k end: $kind $arg rc=$rc"
  echo "$rc" > "$R/item-$k-done-node-$NODE"
  pos=$((k + 1))
  echo "$pos" > "$R/pos-node-$NODE"
done
