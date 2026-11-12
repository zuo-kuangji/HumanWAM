#!/bin/bash
# pi0.5 20-epoch run on NN nodes, global batch 256 = NN x 8 x BS (user 2026-11-12), augmentation aligned with GR00T A20.
# Usage: pi05_multi.sh <task> <nnodes> <rank> <master_addr> <port> <per_gpu>; output pi05-runs/b256/<task>/run
set -uo pipefail
TASK=$1 NN=$2 R=$3 MASTER=$4 PORT=$5 BS=$6
H0=$(dirname "$(realpath "$0")"); G=$(dirname "$H0"); H=$G/hb_pi05
source "$H/hb_env.sh"
OUT=$G/pi05-runs/b256/$TASK/run
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/train-node-$R.log") 2>&1
date -u; hostname; echo "PI05_MULTI task=$TASK nnodes=$NN rank=$R master=$MASTER per_gpu=$BS global=$((NN * 8 * BS)) out=$OUT"
[ $((NN * 8 * BS)) = 256 ] || { echo "global batch must be 256"; exit 2; }
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
for i in $(seq 720); do [ -s "$HB_PI05_DATA/$TASK/meta.json" ] && break; sleep 15; done
if [ "$NN" = 1 ]; then
  "$HB_ENV/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 8 "$H/hb_train.py" --task "$TASK" --exp-name run --per-gpu-batch "$BS" --ckpt-base "$OUT/ckpt"
else
  "$HB_ENV/bin/torchrun" --nnodes "$NN" --node_rank "$R" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
    "$H/hb_train.py" --task "$TASK" --exp-name run --per-gpu-batch "$BS" --ckpt-base "$OUT/ckpt"
fi
rc=$?
echo "$rc" > "$OUT/logs/exit-node-$R"; echo "EXIT $rc"; exit $rc
