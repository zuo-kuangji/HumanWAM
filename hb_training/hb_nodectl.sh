#!/bin/bash
# Per-node control daemon (one per node, next to hb_relay.sh). Start on every node:
#   setsid nohup bash $G/hb_nodectl.sh > /dev/null 2>&1 &
# 1. Cluster-wide stop: a request file $G/control/kill-<id>.req (first line = extended regex of the command lines to
#    stop) is executed by EVERY node within ~10 s: SIGTERM, 20 s grace, SIGKILL, then the node writes
#    $G/control/kill-<id>.done-node-<n> (processes left, GPU processes). One request stops a multi-node run on all
#    nodes at once, so no node is left hanging in a collective whose peers are gone.
# 2. Orphan cleanup: when this node's relay is BETWEEN items (last relay event is "item K end") and the GPUs stay
#    occupied for ORPHAN_GRACE seconds, the leftover training processes (TRAIN_PAT) are killed. Between items no
#    training of ours may run, so anything still holding the GPUs is a leftover of the finished item (2026-11-10:
#    killed DiT4DiT ranks timed out but kept 57 GB per GPU; the relays waited for free GPUs indefinitely).
#    $G/control/hold-node-<n> (or hold-all) pauses the orphan cleanup, e.g. for manual probes.
G=$(dirname "$(realpath "$0")")
C=$G/control
mkdir -p "$C"
NODE=${RANK:-}
if [ -z "$NODE" ]; then
  case $(hostname) in *master-0) NODE=0 ;; *worker-0) NODE=1 ;; *worker-1) NODE=2 ;; *worker-2) NODE=3 ;; *) echo "unknown node"; exit 64 ;; esac
fi
LOG=$C/nodectl-node-$NODE.log
exec >> "$LOG" 2>&1
exec 9> "$C/nodectl-node-$NODE.lock"
flock -n 9 || { echo "$(date -u) nodectl already running on node $NODE"; exit 0; }
TRAIN_PAT='hb_train\.py|launch_finetune\.py|hb_launch\.py|train_deltaflow\.py|scripts/train\.py|accelerate\.commands\.launch|bin/torchrun|openpi.*train|hb_pi05/.*train'
ORPHAN_GRACE=${ORPHAN_GRACE:-120}
echo "$(date -u) nodectl start node $NODE host $(hostname) pid $$"

gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
gpu_n() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }

stop_pat() {  # $1 = regex
  local pat=$1 i
  pkill -TERM -f "$pat" 2>/dev/null
  for i in $(seq 20); do pgrep -f "$pat" > /dev/null || break; sleep 1; done
  pkill -KILL -f "$pat" 2>/dev/null
  for i in $(seq 30); do pgrep -f "$pat" > /dev/null || break; sleep 1; done
}

between_items() {  # last relay event of this node is an "item K end"
  local last
  last=$(grep -aE ' item [0-9]+ (start|end):' "$G/relay/node-$NODE.log" 2>/dev/null | tail -n 1)
  [[ "$last" == *" end: "* ]]
}

busy_since=
while true; do
  for req in "$C"/kill-*.req; do
    [ -f "$req" ] || continue
    id=$(basename "$req" .req)
    done_f=$C/$id.done-node-$NODE
    [ -f "$done_f" ] && continue
    pat=$(head -n 1 "$req")
    [ -n "$pat" ] || continue
    echo "$(date -u) $id: stopping /$pat/ ($(pgrep -fc "$pat") processes)"
    stop_pat "$pat"
    sleep 5
    printf 'left=%s gpu_procs=%s at=%s\n' "$(pgrep -fc "$pat")" "$(gpu_n)" "$(date -u +%FT%TZ)" > "$done_f"
    echo "$(date -u) $id: done $(cat "$done_f")"
  done
  if [ ! -f "$C/hold-node-$NODE" ] && [ ! -f "$C/hold-all" ] && between_items && gpu_busy; then
    now=$(date +%s)
    busy_since=${busy_since:-$now}
    if [ $((now - busy_since)) -ge "$ORPHAN_GRACE" ]; then
      echo "$(date -u) relay between items, GPUs busy for $((now - busy_since)) s: killing leftovers ($(pgrep -fc "$TRAIN_PAT") processes, $(gpu_n) GPU processes)"
      pgrep -fa "$TRAIN_PAT" | cut -c1-200
      stop_pat "$TRAIN_PAT"
      sleep 5
      echo "$(date -u) after cleanup: $(pgrep -fc "$TRAIN_PAT") processes, $(gpu_n) GPU processes"
      busy_since=
    fi
  else
    busy_since=
  fi
  sleep 10
done
