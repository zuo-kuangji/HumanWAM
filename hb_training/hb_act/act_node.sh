#!/bin/bash
# Per-node ACT launcher for ONE task (same structure as hb_psi0/psi0_node.sh): batch probe on node 0 (2-GPU DDP jobs,
# 4 candidates at a time; largest per-GPU batch whose peak reserved memory <= HB_PROBE_MEM_FRAC wins) -> decision.json
# on NAS -> 4 x 8 production (torchrun static rendezvous). Usage on every node: act_node.sh <task> <chain_id> <port>
# Env: HB_PROBE_CANDIDATES, HB_PROBE_STEPS (8), HB_PROBE_MEM_FRAC (0.92), HB_NUM_WORKERS (8), HB_DECISION_TIMEOUT.
set -euo pipefail
TASK=$1 CHAIN=$2 PORT=$3
[[ "$CHAIN" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "bad chain id"; exit 2; }
H=$(dirname "$(realpath "$0")")
G=${G:-$(dirname "$H")}
export G
PSI=$G/HumanWAM/third_party/Psi0
ENV=$G/envs/psi0
PY=$ENV/bin/python
NODE=${RANK:?DLC RANK missing}
MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}
[ "${WORLD_SIZE:?}" = 4 ] || { echo "WORLD_SIZE=$WORLD_SIZE, expected 4"; exit 2; }
ROOT=$G/act-runs/$TASK/$CHAIN
LOGD=$ROOT/logs
DEC=$ROOT/decision.json
FAIL=$ROOT/probe_failed
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/node-$NODE.log") 2>&1
date -u; hostname; echo "ACT task=$TASK chain=$CHAIN port=$PORT node=$NODE master=$MASTER"
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export PATH="$ENV/bin:$PATH" PYTHONPATH="$PSI/src:$H:$G/hb_psi0" PSI_SRC="$PSI" HB_AUG_DIR="${HB_AUG_DIR:-$G/hb_groot}"
export TORCH_HOME="$G/torch-home" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE="/tmp/hb_act_hfds_$(id -u)"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TQDM_MININTERVAL=30
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -n 65535 || true
cd "$PSI"
test -s "$G/torch-home/hub/checkpoints/resnet18-f37072fd.pth" || { echo "ResNet18 ImageNet weights missing in $G/torch-home"; exit 2; }
for _ in $(seq 36); do test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && break; sleep 5; done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo GPU_NODE_BUSY; echo 65 > "$LOGD/prod-node-$NODE.exit"; exit 65; }

probe() {
  local PD=$ROOT/probe
  mkdir -p "$PD"
  local C=(${HB_PROBE_CANDIDATES:-512 384 256 192 128 96 64 32})
  local STEPS=${HB_PROBE_STEPS:-8}
  local i=0
  while [ "$i" -lt "${#C[@]}" ]; do
    local R=("${C[@]:$i:4}") pids=() j=0 B
    for B in "${R[@]}"; do
      rm -rf "$PD/bs$B" "$PD/bs$B.rc"; mkdir -p "$PD/bs$B"
      mapfile -t PA < <("$PY" "$H/hb_act_args.py" args "$TASK" --batch "$B" --steps "$STEPS" --save "$STEPS" \
          --warmup 0 --out "$PD/bs$B/run" --timestamp "${CHAIN}-probe$B" --workers "${HB_PROBE_WORKERS:-4}" --no-resume)
      [ "${#PA[@]}" -gt 50 ] || { echo "hb_act_args failed for probe bs$B"; return 1; }
      ( set +e
        CUDA_VISIBLE_DEVICES=$((2*j)),$((2*j+1)) HB_PROBE_OUT="$PD/bs$B" HB_AUG_LOG="$PD/bs$B/aug" \
          torchrun --standalone --nnodes 1 --nproc_per_node 2 "$H/hb_act_train.py" "${PA[@]}" > "$PD/bs$B.log" 2>&1
        echo $? > "$PD/bs$B.rc" ) &
      pids+=($!); j=$((j+1))
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    for B in "${R[@]}"; do echo "probe bs$B rc=$(cat "$PD/bs$B.rc" 2>/dev/null || echo missing)"; done
    if "$PY" "$H/hb_act_args.py" decide "$TASK" "$PD" "$DEC" "${C[@]:0:$((i+4))}"; then return 0; fi
    i=$((i+4))
  done
  return 1
}

if [ "$NODE" = 0 ]; then
  if [ -s "$DEC" ]; then echo "reusing $DEC"; else rm -f "$FAIL"; probe || { echo "PROBE_FAILED"; touch "$FAIL"; exit 4; }; fi
else
  echo "waiting for $DEC"; t=0
  until [ -s "$DEC" ]; do
    [ ! -e "$FAIL" ] || { echo "node 0 reported probe failure"; exit 4; }
    sleep 15; t=$((t+15)); [ "$t" -lt "${HB_DECISION_TIMEOUT:-10800}" ] || { echo "timeout waiting for $DEC"; exit 4; }
  done
fi
cat "$DEC"
[ "${HB_PROBE_ONLY:-0}" != 1 ] || exit 0
mapfile -t ARGS < <("$PY" "$H/hb_act_args.py" args "$TASK" --decision "$DEC" --out "$ROOT/train" --timestamp "$CHAIN")
[ "${#ARGS[@]}" -gt 50 ] || { echo "hb_act_args failed"; exit 5; }
[ "$NODE" != 0 ] || printf '%s\n' "${ARGS[@]}" > "$ROOT/train_args.txt"
export HB_AUG_LOG="$LOGD/aug-node$NODE"
set +e
torchrun --nnodes 4 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_act_train.py" "${ARGS[@]}"
rc=$?
echo "$rc" > "$LOGD/prod-node-$NODE.exit"
echo "EXIT $rc"
exit $rc
