#!/bin/bash
# 2026-11-14 user: GR00T convergence probe on basket (full fine-tune, 4000-step LR schedule, stopped at step 500,
# no checkpoints). Usage: groot_probe.sh <name> <nnodes 1|4> <per_gpu> <accum> [port]
set -uo pipefail
NAME=$1 NN=$2 BS=$3 ACC=$4 PORT=${5:-29670}
TASK=basket STOP=${HB_PROBE_STOP:-500} CFG=g1_feedback76_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T; ENV=$G/envs/groot-py312
NODE=${RANK:?DLC RANK missing}; MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}
R=$G/groot-runs-probe/$NAME; OUT=$R/out; mkdir -p "$OUT"
LOGF=$R/train-node-$NODE.log; WATCH=$R/train-node-0.log; [ "$NN" = 1 ] && WATCH=$LOGF
NG=$((NN * 8)); GB=$((BS * NG)); EFF=$((GB * ACC))
echo "$(date) PROBE name=$NAME nnodes=$NN per_gpu=$BS accum=$ACC micro_global=$GB global=$EFF stop=$STOP node=$NODE" >> "$LOGF"
if [ -f "$R/PROBE_DONE" ]; then echo "already done" >> "$LOGF"; exit 0; fi
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7
fi
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$GR:$H" GROOT_SRC="$GR" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HB_AUG_LOG="$R/aug-node-$NODE" HB_PEAK_FILE="$R/peak-node-$NODE" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ "$NN" = 1 ]; then LAUNCH=(--standalone --nnodes 1 --nproc_per_node 8)
else LAUNCH=(--nnodes 4 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT"); fi
cd "$GR"
"$ENV/bin/torchrun" "${LAUNCH[@]}" "$H/hb_launch.py" \
  --base_model_path "$G/models/groot-n17-base" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
  --tune_llm --tune_visual \
  --num_gpus "$NG" --output_dir "$OUT" --save_steps 1000000 --save_total_limit 1 --max_steps 4000 \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size "$GB" --gradient_accumulation_steps "$ACC" \
  --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1 \
  >> "$LOGF" 2>&1 < /dev/null &
TP=$!
T0=$(date +%s)
while kill -0 "$TP" 2>/dev/null; do
  s=$(tr '\r' '\n' < "$WATCH" 2>/dev/null | grep -aoE '[0-9]+/4000 ' | tail -n 1 | cut -d/ -f1)
  if [ -n "$s" ] && [ "$s" -ge "$STOP" ]; then echo "$(date) PROBE_STOP at step $s" >> "$LOGF"; break; fi
  if [ $(( $(date +%s) - T0 )) -gt 7200 ]; then echo "$(date) PROBE_TIMEOUT" >> "$LOGF"; break; fi
  sleep 10
done
pkill -TERM -f "output_dir $OUT" 2>/dev/null; sleep 20; pkill -KILL -f "output_dir $OUT" 2>/dev/null
wait "$TP" 2>/dev/null
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
rm -rf "$OUT"/checkpoint-* 2>/dev/null
[ "$NODE" = 0 ] && touch "$R/PROBE_DONE"
echo "$(date) PROBE_END $NAME" >> "$LOGF"
exit 0
