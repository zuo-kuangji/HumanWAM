#!/bin/bash
# B7 stage 1 (2026-11-11, user): nodes 0-2 = pi0.5 B7 on 3 nodes x 8 GPUs x 29 = global 696 (same global batch, steps and
# lr schedule as the single-node 696 runs); node 3 = GR00T B7 single node, 8 x 87 = 696. Started by b7_s1_wait.sh.
G=$(dirname "$(dirname "$(realpath "$0")")")
NODE=${RANK:?DLC RANK missing}
MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}
gpu_free() { for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 5; done; return 1; }
if [ "$NODE" = 3 ]; then
  gpu_free || { echo GPU_NODE_BUSY; exit 65; }
  bash "$G/hb_groot/groot_single.sh" b7 87 "$G/groot-runs-b696"
  exit $?
fi
H=$G/hb_pi05
source "$H/hb_env.sh"
OUT=$G/pi05-runs/b696/b7/run
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/train-node-$NODE.log") 2>&1
date -u; hostname; echo "PI05_3NODE task=b7 node=$NODE per_gpu=29 world=24 global=696 out=$OUT"
gpu_free || { echo GPU_NODE_BUSY; exit 65; }
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXTRA=()
[ "${HB_RESUME:-0}" = 1 ] && EXTRA+=(--resume)
"$HB_ENV/bin/torchrun" --nnodes 3 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port 30470 \
  "$H/hb_train.py" --task b7 --exp-name run --per-gpu-batch 29 --ckpt-base "$OUT/ckpt" "${EXTRA[@]}"
rc=$?
echo "$rc" > "$OUT/logs/exit-node-$NODE"
echo "EXIT $rc"
exit $rc
