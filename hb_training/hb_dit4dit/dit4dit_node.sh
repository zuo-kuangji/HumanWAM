#!/bin/bash
# Per-node DiT4DiT launcher for ONE task: 4 nodes x 8 GPUs, `accelerate launch` (DeepSpeed ZeRO-2, bf16, multinode
# launcher "standard" = torchrun static rendezvous on the DLC MASTER_ADDR), exactly like the native run scripts.
# Usage (on every node, same args): dit4dit_node.sh <task> <per_gpu_batch> <master_port> <out_dir>
# Env: HB_EPOCHS (20), HB_SAVE_EPOCHS (10), HB_NUM_WORKERS (6). Resume is automatic when <out_dir>/run/checkpoints exists.
set -euo pipefail
TASK=$1 BS=$2 PORT=$3 OUT=$4
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
NODE=${RANK:?cluster RANK (node rank) missing}
MASTER=${MASTER_ADDR:?cluster MASTER_ADDR missing}
[ "${WORLD_SIZE:?}" = 4 ] || { echo "WORLD_SIZE=$WORLD_SIZE, expected 4 nodes"; exit 64; }
mkdir -p "$OUT/logs"
exec > >(tee -a "$OUT/logs/train-node-$NODE.log") 2>&1
date -u; hostname; echo "task=$TASK bs=$BS port=$PORT out=$OUT node=$NODE master=$MASTER"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo GPU_NODE_BUSY; exit 65; }
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
# DLC PET_* (PET_NNODES, PET_NODE_RANK, PET_MASTER_ADDR/PORT, PET_NPROC_PER_NODE ...) are torch.distributed.run argument
# defaults; the explicit accelerate arguments below must be the only source of the rendezvous settings.
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HB_AUG_LOG="$OUT/logs/aug-node$NODE"
LF=$OUT/launch/node-$NODE   # per node: the 4 nodes write the same files, never the same path
"$HB_ENV/bin/python" "$H/hb_d4d_config.py" launch-files --task "$TASK" --batch "$BS" --world 32 --machines 4 --out "$LF"
cd "$HB_D4D_SRC"
set +e
"$HB_ENV/bin/python" -m accelerate.commands.launch --config_file "$LF/accelerate.yaml" \
  --num_machines 4 --machine_rank "$NODE" --num_processes 32 --main_process_ip "$MASTER" --main_process_port "$PORT" \
  --max_restarts 0 \
  "$H/hb_train.py" --task "$TASK" --per-gpu-batch "$BS" --run-root "$OUT" --ds-config "$LF/ds_config.json"
rc=$?
echo "EXIT $rc"
exit $rc
