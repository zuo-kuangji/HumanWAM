#!/bin/bash
# pi0.5 20-epoch run on a PAIR of nodes, global batch 512 = 2 x 8 x 32 (user 2026-11-11 20:27).
# Usage: pi05_pair.sh <task> <rank 0|1> <master_addr> <port>; output pi05-runs/b512/<task>/run
set -uo pipefail
TASK=$1 R=$2 MASTER=$3 PORT=$4
H0=$(dirname "$(realpath "$0")"); G=$(dirname "$H0"); H=$G/hb_pi05
source "$H/hb_env.sh"
OUT=$G/pi05-runs/b512/$TASK/run
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/train-node-$R.log") 2>&1
date -u; hostname; echo "PI05_PAIR task=$TASK rank=$R master=$MASTER per_gpu=32 world=16 global=512 out=$OUT"
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
"$HB_ENV/bin/torchrun" --nnodes 2 --node_rank "$R" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_train.py" --task "$TASK" --exp-name run --per-gpu-batch 32 --ckpt-base "$OUT/ckpt"
rc=$?
echo "$rc" > "$OUT/logs/exit-node-$R"; echo "EXIT $rc"; exit $rc
