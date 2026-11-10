#!/bin/bash
# Per-node launcher: 4 nodes x 8 GPUs, torchrun static rendezvous on the cluster's MASTER_ADDR.
# Usage (on every node, same args): pi05_node.sh <task> <per_gpu_batch> <master_port> <out_dir>
# Env: HB_EPOCHS (20), HB_SAVE_EPOCHS (10), HB_RESUME=1 to resume from the latest checkpoint, HB_NUM_WORKERS (8).
set -euo pipefail
TASK=$1 BS=$2 PORT=$3 OUT=$4
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
NODE=${RANK:?cluster RANK (node rank) missing}
MASTER=${MASTER_ADDR:?cluster MASTER_ADDR missing}
[ "${WORLD_SIZE:?}" = 4 ] || { echo "WORLD_SIZE=$WORLD_SIZE, expected 4 nodes"; exit 64; }
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/train-node-$NODE.log") 2>&1
date -u; hostname; echo "task=$TASK bs=$BS port=$PORT out=$OUT"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo GPU_NODE_BUSY; exit 65; }
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXTRA=()
[ "${HB_RESUME:-0}" = 1 ] && EXTRA+=(--resume)
set +e
"$HB_ENV/bin/torchrun" --nnodes 4 --node_rank "$NODE" --nproc_per_node 8 \
  --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_train.py" --task "$TASK" --exp-name "$(basename "$OUT")" --per-gpu-batch "$BS" \
  --ckpt-base "$OUT/ckpt" "${EXTRA[@]}"
rc=$?
echo "EXIT $rc"
exit $rc
