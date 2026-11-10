#!/usr/bin/env bash
# One 32-GPU accelerate/DeepSpeed ZeRO-1 launch of the Delta-flow production run on this node
# (same launcher, env and NCCL settings as the previous B6/A3/A7 launch_attempt.sh).
# Usage: deltaflow_attempt.sh <run_root R> <master_port> <resume_state_dir|none>   (DLC env RANK/MASTER_ADDR)
set -euo pipefail
R=$1
PORT=$2
RESUME=$3
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
P=$(dirname "$G")
C=$G/deltaflow-runs/HumanWAM
[ "$(basename "$R")" != a18 ] || C=$G/deltaflow-runs/HumanWAM-h2   # H2 code tree (own imagewam package)
PY=$P/venv/bin/python
node_rank="${RANK:?DLC node rank missing}"
num_nodes="${WORLD_SIZE:?DLC node count missing}"
master_addr="${MASTER_ADDR:?DLC master address missing}"
case "$node_rank" in 0|1|2|3) ;; *) echo INVALID_NODE_RANK; exit 64;; esac
[ "$num_nodes" = 4 ] || { echo "WORLD_SIZE=$num_nodes, need 4"; exit 64; }
test -f "$R/run/production.yaml"
test -f "$R/data-verified.json"
test -f "$C/scripts/wholebody/train_deltaflow.py"
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo GPU_NODE_BUSY; exit 65; }
date -u
hostname
echo "DELTAFLOW run=$R node=$node_rank/4 master=$master_addr:$PORT resume=$RESUME"
export PATH="$P/venv/bin:$PATH"
export PYTHONPATH="$C/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 WANDB_MODE=offline
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE
cd "$C"
args=(--config "$R/run/production.yaml")
[ "$RESUME" = none ] || args+=(--resume "$RESUME")
exec "$PY" -m accelerate.commands.launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --deepspeed_multinode_launcher standard --num_processes 32 --num_machines 4 --machine_rank "$node_rank" \
  --main_process_ip "$master_addr" --main_process_port "$PORT" --same_network \
  scripts/wholebody/train_deltaflow.py "${args[@]}"
