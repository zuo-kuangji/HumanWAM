#!/bin/bash
# Task pool for single-node jobs (run as ONE hb_relay "cmd" item on all 4 nodes): every node repeatedly claims the
# next unclaimed job of $G/pools/<pool>/jobs.txt (atomic mkdir on NAS) and runs it on its own 8 GPUs; a node that
# finishes takes the next job at once. Returns when no job is left (the relay then waits for the other nodes).
# jobs.txt lines (order = priority): <job_id> <model: groot|pi05> <task> <per_gpu_batch>
# Status: pools/<pool>/status/<job_id>.{node,start,end,rc}; log pools/<pool>/pool-node-<n>.log
# Usage: hb_pool.sh <pool>
H=$(dirname "$(realpath "$0")")
G=$H
POOL=$G/pools/${1:?usage: hb_pool.sh <pool>}
NODE=${RANK:?DLC RANK missing}
mkdir -p "$POOL/claims" "$POOL/status"
exec >> "$POOL/pool-node-$NODE.log" 2>&1
echo "$(date -u) pool $(basename "$POOL") node $NODE host $(hostname) start"
sleep $((NODE * 8))  # deterministic first assignment: node 0 claims job 1, node 1 job 2, ...
PAT='hb_train\.py|hb_launch\.py|bin/torchrun'
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ]; }
free_gpus() {  # this node only runs pool jobs now: clear leftovers of a finished job if the GPUs stay busy
  local i
  for i in $(seq 18); do gpu_busy || return 0; sleep 10; done
  echo "$(date -u) GPUs still busy after 3 min: killing leftovers"; pgrep -fa "$PAT" | cut -c1-160
  pkill -TERM -f "$PAT"; sleep 20; pkill -KILL -f "$PAT"; sleep 10
  gpu_busy && { echo "$(date -u) GPUs STILL busy"; return 1; }
  return 0
}
while true; do
  claimed=
  while read -r id model task bs _; do
    [ -n "$id" ] && [ "${id:0:1}" != "#" ] || continue
    if mkdir "$POOL/claims/$id" 2>/dev/null; then claimed=$id; break; fi
  done < "$POOL/jobs.txt"
  [ -n "$claimed" ] || break
  free_gpus || { echo "$(date -u) $claimed: GPUs not free, releasing claim"; rmdir "$POOL/claims/$claimed"; sleep 60; continue; }
  echo "$NODE" > "$POOL/status/$claimed.node"; date -u +%FT%TZ > "$POOL/status/$claimed.start"
  echo "$(date -u) $claimed: $model $task per_gpu=$bs START"
  case $model in
    groot) bash "$G/hb_groot/groot_single.sh" "$task" "$bs" "$G/groot-runs-b696" > "$POOL/status/$claimed.out" 2>&1; rc=$? ;;
    pi05)  bash "$G/hb_pi05/pi05_single.sh" "$task" "$bs" "$G/pi05-runs/b696/$task/run" > "$POOL/status/$claimed.out" 2>&1; rc=$? ;;
    groot256) bash "$G/hb_groot/groot_single.sh" "$task" "$bs" "$G/groot-runs-b256" > "$POOL/status/$claimed.out" 2>&1; rc=$? ;;
    act)   bash "$G/hb_act/act_single_b256.sh" "$task" "$bs" "$G/act-runs-b256" > "$POOL/status/$claimed.out" 2>&1; rc=$? ;;
    *)     echo "unknown model $model"; rc=99 ;;
  esac
  echo "$rc" > "$POOL/status/$claimed.rc"; date -u +%FT%TZ > "$POOL/status/$claimed.end"
  echo "$(date -u) $claimed: $model $task END rc=$rc"
done
echo "$(date -u) pool $(basename "$POOL") node $NODE: no job left"
