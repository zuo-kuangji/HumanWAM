#!/bin/bash
# Single-node pi0.5 20-epoch production on THIS node's 8 GPUs (torchrun --standalone), for the task pool (hb_pool.sh).
# Usage: pi05_single.sh <task> <per_gpu_batch> <out_dir>      e.g. pi05_single.sh b6 87 $G/pi05-runs/b696/b6/run
# Same recipe as pi05_node.sh / hb_train.py (20 epochs, saves at epoch 10 and 20, HB aug v2, peak lr 5e-5, warmup 5 %)
# except 1 node x 8 GPUs and the given per-GPU batch (user 2026-11-10: global batch 696 = 87 x 8). HB_RESUME=1 resumes.
set -euo pipefail
TASK=$1 BS=$2 OUT=$3
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/train.log") 2>&1
date -u; hostname; echo "PI05_SINGLE task=$TASK per_gpu=$BS global=$((BS * 8)) out=$OUT"
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXTRA=()
[ "${HB_RESUME:-0}" = 1 ] && EXTRA+=(--resume)
set +e
"$HB_ENV/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 8 \
  "$H/hb_train.py" --task "$TASK" --exp-name run --per-gpu-batch "$BS" --ckpt-base "$OUT/ckpt" "${EXTRA[@]}"
rc=$?
echo "EXIT $rc"
exit $rc
