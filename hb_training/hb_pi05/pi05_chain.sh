#!/bin/bash
# Probe -> production chain for the HB tasks, sequential, NAS-file barriers.
# Run the SAME command on all 4 nodes (cluster env RANK=0..3, WORLD_SIZE=4, MASTER_ADDR):
#     bash pi05_chain.sh <chain_id> [task ...]          (default tasks: b6 basket a3 a7 a18)
# Node 0 (master): checks inputs; before the first task of each robot layout (g1, h2) it probes the per-GPU batch on its
# own 8 GPUs with that task (descending candidates, first fit wins) -> <chain>/probe-<layout>.env. The layouts differ in
# max_token_len (g1 256, h2 384), hence in activation memory, so each layout gets its own probe (h2 is probed with a18).
# Master writes <chain>/<task>/decision.env; all nodes run pi05_node.sh; master waits for 4 exit files before the next task.
# Workers only wait for decision files (or ABORT). Re-running with the same chain_id skips finished tasks.
# Env: HB_PROBE_CANDIDATES ("96 80 64 48 40 32 24 16 8"), HB_PROBE_STEPS (5), HB_PROBE_MAX_FRAC (0.95),
#      HB_BATCH_G1 / HB_BATCH_H2 / HB_BATCH (skip probe, force per-GPU batch), HB_BASE_PORT (29511),
#      HB_EPOCHS (20), HB_SAVE_EPOCHS (10).
set -euo pipefail
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
CHAIN=${1:?usage: pi05_chain.sh <chain_id> [task ...]}
shift
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(b6 basket a3 a7 a18)
NODE=${RANK:?cluster RANK missing}
[ "${WORLD_SIZE:?}" = 4 ] || { echo "WORLD_SIZE=$WORLD_SIZE, expected 4"; exit 64; }
C=$HB_PI05_RUNS/chain-$CHAIN
mkdir -p "$C"
exec > >(tee -a "$C/chain-node-$NODE.log") 2>&1
date -u; hostname; echo "chain=$CHAIN node=$NODE tasks=${TASKS[*]}"

put() { printf '%s\n' "$2" > "$1.tmp.$$" && mv "$1.tmp.$$" "$1"; }            # atomic small file on NAS
abort() { echo "ABORT: $*"; put "$C/ABORT" "node $NODE: $*"; exit 1; }
START=$C/.start-node-$NODE
touch "$START"                                                                    # ABORTs older than this launch are stale
aborted() { [ -f "$C/ABORT" ] && [ "$C/ABORT" -nt "$START" ]; }
wait_for() { until [ -f "$1" ]; do aborted && { echo "ABORT seen: $(cat "$C/ABORT")"; exit 1; }; sleep 15; done; }
layout_of() { "$HB_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("layout","g1"))' "$HB_PI05_DATA/$1/meta.json"; }

probe() {  # master only; local 8-GPU DDP, same per-GPU memory as the 32-GPU run (static_graph, grad ckpt, bf16)
  local t=$1 lay=$2 b pd
  for b in ${HB_PROBE_CANDIDATES:-96 80 64 48 40 32 24 16 8}; do
    pd=$C/probe/$lay/bs$b
    rm -rf "$pd"; mkdir -p "$pd"
    echo "probe $lay ($t) per_gpu_batch=$b"
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || abort "GPU busy before probe"
    ( unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      "$HB_ENV/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 8 "$H/hb_train.py" --task "$t" \
        --exp-name "probe-$lay-bs$b" --per-gpu-batch "$b" --num-train-steps "${HB_PROBE_STEPS:-5}" \
        --ckpt-base "$pd/ckpt" --probe-dir "$pd" ) > "$pd/log" 2>&1 || true
    sleep 10  # let the driver release memory
    if [ "$(ls "$pd"/ok_rank*.json 2>/dev/null | wc -l)" = 8 ]; then
      put "$C/probe-$lay.env" "BS=$b"; echo "probe $lay: per_gpu_batch=$b fits ($(cat "$pd/ok_rank0.json"))"; return 0
    fi
    if ls "$pd"/oom_rank*.json >/dev/null 2>&1; then echo "probe $lay: bs=$b OOM/over headroom"; continue; fi
    tail -n 40 "$pd/log"; abort "probe $lay bs=$b failed without an OOM marker (see $pd/log)"
  done
  abort "no candidate per-GPU batch fits for $lay"
}

if [ "$NODE" = 0 ]; then
  rm -f "$C/ABORT"
  for t in "${TASKS[@]}"; do [ -f "$C/$t/DONE" ] || rm -f "$C/$t/decision.env"; done   # stale decisions of a previous launch
  for t in "${TASKS[@]}"; do [ -f "$HB_PI05_DATA/$t/READY" ] || abort "$HB_PI05_DATA/$t not prepared (hb_prepare.py $t)"; done
  [ -f "$HB_PI05_INIT/model.safetensors" ] || abort "$HB_PI05_INIT/model.safetensors missing (convert_base.sh)"
else
  sleep 60  # give the master time to clear stale decision files of an aborted earlier launch
fi

i=0
for t in "${TASKS[@]}"; do
  D=$C/$t
  PORT=$(( ${HB_BASE_PORT:-29511} + i )); i=$((i + 1))
  if [ "$NODE" = 0 ]; then
    if [ -f "$D/DONE" ]; then echo "$t already done"; continue; fi
    mkdir -p "$D"; rm -f "$D"/exit-node-*
    LAY=$(layout_of "$t")
    FORCE_VAR="HB_BATCH_${LAY^^}"
    FORCE=${!FORCE_VAR:-${HB_BATCH:-}}
    if [ -n "$FORCE" ]; then put "$C/probe-$LAY.env" "BS=$FORCE"; fi
    [ -f "$C/probe-$LAY.env" ] || probe "$t" "$LAY"
    source "$C/probe-$LAY.env"
    put "$D/decision.env" "BS=$BS PORT=$PORT"
  else
    if [ -f "$D/DONE" ]; then continue; fi
    wait_for "$D/decision.env"
  fi
  source "$D/decision.env"
  echo "== $t: per_gpu_batch=$BS global_batch=$((BS * 32)) port=$PORT"
  rc=0
  bash "$H/pi05_node.sh" "$t" "$BS" "$PORT" "$D/run" || rc=$?
  put "$D/exit-node-$NODE" "$rc"
  if [ "$NODE" = 0 ]; then
    for n in 0 1 2 3; do wait_for "$D/exit-node-$n"; done
    bad=$(for n in 0 1 2 3; do echo "$n:$(cat "$D/exit-node-$n")"; done | grep -v ':0$' || true)
    [ -z "$bad" ] || abort "$t failed on nodes: $bad"
    put "$D/DONE" "ok"
    rm -f "$D/decision.env"
  else
    [ "$rc" = 0 ] || { echo "$t failed here (rc=$rc); master will abort"; exit "$rc"; }
    wait_for "$D/DONE"
  fi
done
echo CHAIN_DONE
