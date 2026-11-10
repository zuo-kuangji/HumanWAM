#!/bin/bash
# Probe -> production chain for the HB DiT4DiT tasks, sequential, NAS-file barriers.
# Run the SAME command on all 4 nodes (DLC env RANK=0..3, WORLD_SIZE=4, MASTER_ADDR):
#     bash dit4dit_chain.sh <chain_id> <port_base> [task ...]          (default tasks: b6 a3 a7 a18)
# Node 0 (master): checks inputs; before the first task of each robot layout (g1, h2) it probes the per-GPU batch on its
# own 8 GPUs with that task (descending candidates, first one that finishes HB_PROBE_STEPS steps with peak reserved
# <= HB_PROBE_MAX_FRAC of the GPU wins) -> <chain>/probe-<layout>.env. Probe and production use the same per-GPU memory
# except ZeRO-2 optimizer shards (1/8 vs 1/32 of ~26 GB fp32 Adam state), so the probe is ~2.5 GB conservative.
# Master writes <chain>/<task>/decision.env; all nodes run dit4dit_node.sh; master waits for the 4 exit files.
# Workers only wait for decision files (or ABORT). Re-running with the same chain_id skips DONE tasks and resumes the rest
# from their latest checkpoint (weights + optimizer state).
# Ports: production <port_base>+i (i = task position), probe <port_base>+50+i.
# Env: HB_PROBE_CANDIDATES ("24 16 12 8 6 4 2"), HB_PROBE_STEPS (6), HB_PROBE_MAX_FRAC (0.92),
#      HB_BATCH_G1 / HB_BATCH_H2 / HB_BATCH (skip the probe, force the per-GPU batch), HB_EPOCHS (20), HB_SAVE_EPOCHS (10).
set -euo pipefail
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
CHAIN=${1:?usage: dit4dit_chain.sh <chain_id> <port_base> [task ...]}
PB=${2:?usage: dit4dit_chain.sh <chain_id> <port_base> [task ...]}
shift 2
TASKS=("$@")
[ ${#TASKS[@]} -gt 0 ] || TASKS=(b6 a3 a7 a18)
[[ "$CHAIN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "chain id must match [A-Za-z0-9_-]+"; exit 64; }
NODE=${RANK:?cluster RANK missing}
[ "${WORLD_SIZE:?}" = 4 ] || { echo "WORLD_SIZE=$WORLD_SIZE, expected 4"; exit 64; }
C=$HB_D4D_RUNS/chain-$CHAIN
mkdir -p "$C"
exec > >(tee -a "$C/chain-node-$NODE.log") 2>&1
date -u; hostname; echo "chain=$CHAIN node=$NODE port_base=$PB tasks=${TASKS[*]}"

put() { printf '%s\n' "$2" > "$1.tmp.$$" && mv "$1.tmp.$$" "$1"; }            # atomic small file on NAS
abort() { echo "ABORT: $*"; put "$C/ABORT" "node $NODE: $*"; exit 1; }
START=$C/.start-node-$NODE
touch "$START"                                                                    # ABORTs older than this launch are stale
aborted() { [ -f "$C/ABORT" ] && [ "$C/ABORT" -nt "$START" ]; }
wait_for() { until [ -f "$1" ]; do aborted && { echo "ABORT seen: $(cat "$C/ABORT")"; exit 1; }; sleep 15; done; }
layout_of() { "$HB_ENV/bin/python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["layout"])' "$HB_D4D_DATA/$1/meta.json"; }

probe() {  # master only; local 8-GPU DeepSpeed run, real model + data, no checkpoints
  local t=$1 lay=$2 port=$3 b pd lf
  for b in ${HB_PROBE_CANDIDATES:-24 16 12 8 6 4 2}; do
    pd=$C/probe/$lay/bs$b
    rm -rf "$pd"; mkdir -p "$pd"
    lf=$pd/launch
    echo "$(date -u) probe $lay ($t) per_gpu_batch=$b"
    test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || abort "GPU busy before probe"
    ( unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
      # DLC also sets PET_NNODES=4 / PET_NODE_RANK / PET_MASTER_* ...; torch.distributed.run takes PET_* as argument
      # defaults, so a single-node probe would wait for 4 machines (2026-11-10: "1/4 clients joined" after 900 s).
      unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
      export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_DEBUG=WARN
      "$HB_ENV/bin/python" "$H/hb_d4d_config.py" launch-files --task "$t" --batch "$b" --world 8 --machines 1 --out "$lf"
      cd "$HB_D4D_SRC"
      timeout "${HB_PROBE_TIMEOUT:-3600}" "$HB_ENV/bin/python" -m accelerate.commands.launch --config_file "$lf/accelerate.yaml" \
        --num_machines 1 --num_processes 8 --main_process_port "$port" --max_restarts 0 \
        "$H/hb_train.py" --task "$t" --per-gpu-batch "$b" --run-root "$pd" --ds-config "$lf/ds_config.json" \
        --num-train-steps "${HB_PROBE_STEPS:-6}" --probe-dir "$pd" ) > "$pd/log" 2>&1 || true
    sleep 15  # let the driver release memory
    pkill -f "hb_train.py --task $t --per-gpu-batch $b" 2>/dev/null || true
    if [ "$(ls "$pd"/ok_rank*.json 2>/dev/null | wc -l)" = 8 ]; then
      put "$C/probe-$lay.env" "BS=$b"
      echo "probe $lay: per_gpu_batch=$b fits ($(tr -d '\n ' < "$pd/ok_rank0.json"))"
      return 0
    fi
    if ls "$pd"/oom_rank*.json >/dev/null 2>&1 || grep -qiE 'out of memory|OutOfMemoryError' "$pd/log"; then
      echo "probe $lay: bs=$b OOM / over headroom"; continue
    fi
    tail -n 60 "$pd/log"; abort "probe $lay bs=$b failed without an OOM marker (see $pd/log)"
  done
  abort "no candidate per-GPU batch fits for $lay"
}

if [ "$NODE" = 0 ]; then
  rm -f "$C/ABORT"
  for t in "${TASKS[@]}"; do [ -f "$C/$t/DONE" ] || rm -f "$C/$t/decision.env"; done   # stale decisions of a previous launch
  for t in "${TASKS[@]}"; do [ -f "$HB_D4D_DATA/$t/READY" ] || abort "$HB_D4D_DATA/$t not prepared (dit4dit_prepare.py $t)"; done
  [ -f "$HB_COSMOS_DIR/model_index.json" ] || abort "$HB_COSMOS_DIR missing (fetch_models.sh)"
  grep -q . "$HB_ENV/hb-diffusers-pinned" 2>/dev/null || abort "$HB_ENV not built (build_env.sh)"
else
  sleep 60  # give the master time to clear stale decision files of an aborted earlier launch
fi

i=0
for t in "${TASKS[@]}"; do
  D=$C/$t
  PORT=$((PB + i)); PPORT=$((PB + 50 + i)); i=$((i + 1))
  if [ "$NODE" = 0 ]; then
    if [ -f "$D/DONE" ]; then echo "$t already done"; continue; fi
    mkdir -p "$D"; rm -f "$D"/exit-node-*
    LAY=$(layout_of "$t")
    FORCE_VAR="HB_BATCH_${LAY^^}"
    FORCE=${!FORCE_VAR:-${HB_BATCH:-}}
    if [ -n "$FORCE" ]; then put "$C/probe-$LAY.env" "BS=$FORCE"; fi
    [ -f "$C/probe-$LAY.env" ] || probe "$t" "$LAY" "$PPORT"
    source "$C/probe-$LAY.env"
    put "$D/decision.env" "BS=$BS PORT=$PORT"
  else
    if [ -f "$D/DONE" ]; then continue; fi
    wait_for "$D/decision.env"
  fi
  source "$D/decision.env"
  echo "== $(date -u) $t: per_gpu_batch=$BS global_batch=$((BS * 32)) port=$PORT"
  rc=0
  bash "$H/dit4dit_node.sh" "$t" "$BS" "$PORT" "$D" || rc=$?
  put "$D/exit-node-$NODE" "$rc"
  if [ "$NODE" = 0 ]; then
    for n in 0 1 2 3; do wait_for "$D/exit-node-$n"; done
    bad=$(for n in 0 1 2 3; do echo "$n:$(cat "$D/exit-node-$n")"; done | grep -v ':0$' || true)
    [ -z "$bad" ] || abort "$t failed on nodes: $bad"
    [ -f "$D/run/HB_DONE" ] || abort "$t: all nodes exited 0 but $D/run/HB_DONE is missing"
    put "$D/DONE" "ok"
    rm -f "$D/decision.env"
  else
    [ "$rc" = 0 ] || { echo "$t failed here (rc=$rc); master will abort"; exit "$rc"; }
    wait_for "$D/DONE"
  fi
done
echo "CHAIN_DONE $CHAIN ${TASKS[*]}"
