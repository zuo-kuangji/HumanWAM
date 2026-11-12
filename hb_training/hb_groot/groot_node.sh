#!/bin/bash
# Per-node GR00T launcher (4 nodes x 8 GPUs, torchrun static rendezvous on DLC MASTER_ADDR).
# Usage: groot_node.sh <task> <phase> <per_gpu_batch> <max_steps> <save_steps> <port>
set -euo pipefail
TASK=$1 PHASE=$2 BS=$3 STEPS=$4 SAVE=$5 PORT=$6
CFG=g1_feedback76_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm
if [[ $TASK = a1[78] ]]; then CFG=h2_body31_hand44_h16.py; export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm,head; fi
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T
ENV=$G/envs/groot-py312
NODE=${RANK:?DLC RANK missing}
MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}
[ "${WORLD_SIZE:?}" = 4 ]
GRUNS=${HB_GROOT_RUNS:-$G/groot-runs-bmax}
OUT=$GRUNS/$TASK/$PHASE
LOGD=$GRUNS/$TASK/logs
mkdir -p "$OUT" "$LOGD"
exec > >(tee -a "$LOGD/$PHASE-node-$NODE.log") 2>&1
date -u; hostname
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  echo "installing FFmpeg 4.4 runtime for torchcodec"
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7
fi
for _ in $(seq 36); do  # up to 3 min for processes of a previous (e.g. OOM) probe to release the GPUs
  test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" && break
  sleep 5
done
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" || { echo GPU_NODE_BUSY; echo 65 > "$LOGD/$PHASE-node-$NODE.exit"; exit 65; }
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE
# /usr/local/bin/python3.12 is a static build (no libpython3.12.so); torchcodec's custom-ops lib has it as DT_NEEDED,
# so an empty stub with that soname satisfies the loader (symbols come from the executable).
export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$GR:$H" GROOT_SRC="$GR" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HB_AUG_LOG="$LOGD/aug-$PHASE-node-$NODE" HB_PEAK_FILE="$LOGD/peak-$PHASE" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$GR"
set +e
"$ENV/bin/torchrun" --nnodes 4 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_launch.py" \
  --base_model_path "$G/models/groot-n17-base" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
  --num_gpus 32 --output_dir "$OUT" --save_steps "$SAVE" --save_total_limit 2 --max_steps "$STEPS" \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 --global_batch_size $((BS * 32)) \
  --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1
rc=$?
echo "$rc" > "$LOGD/$PHASE-node-$NODE.exit"
echo "EXIT $rc"
exit $rc
