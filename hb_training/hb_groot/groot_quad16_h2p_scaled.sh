#!/bin/bash
# 2026-11-28 scaled copy of the r80 launcher: STEPS / SAVE from HB_STEPS / HB_SAVE (user 11-28: steps scale with the mean episode length vs A20 basket v2, ceil to 1000; ckpt ~80 % + last).
# 2026-11-27 r80 copy (big-model checkpoint rule, protocol.yaml interface.save_rule): checkpoints 3200 and 4000 (HF Trainer also saves the last step) only.
# 2026-11-20 copy of groot_quad16_h2.sh for the H2 action82 (+ planner) tasks (A14 cropped-v3): GR00T H2 2-camera task on all 4 nodes, 512 = 4 x 8 x 16, no accumulation, same recipe
# as groot_pair2x32_h2.sh. Use only if per-GPU 32 does not fit.   groot_quad16_h2.sh <a18c2|a17c2> <port>
set -uo pipefail
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T; ENV=$G/envs/groot-py312; CFG=h2_body31_hand44_planner7_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm,head
NODE=${RANK:?}; MASTER=${MASTER_ADDR:?}; [ "${WORLD_SIZE:?}" = 4 ]
TASK=$1 PORT=$2 BS=16 ACCUM=1 SAVE=${HB_SAVE:?} STEPS=${HB_STEPS:?} LR=1e-4 KEEP=2 SAVE_LIST="[$SAVE, $STEPS]" SRC=$G/models/groot-n17-base
ROOT=$G/groot-runs-2x32/$TASK; OUT=$ROOT/production; LOGD=$ROOT/logs; mkdir -p "$OUT" "$LOGD"
LOGF=$LOGD/production.log; [ "$NODE" = 0 ] || LOGF=$LOGD/production-node-$NODE.log
exec > >(tee -a "$LOGF") 2>&1
date -u; hostname
if [ -f "$ROOT/production-complete.json" ]; then echo "already complete"; exit 0; fi
REC=$G/groot-data/$TASK/hb-dataset-receipt.json; [ -f "$REC" ] || { echo "dataset $TASK not prepared"; exit 3; }
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$REC")
GB=$((BS * 32)); [ "$GB" = 512 ] || exit 2
echo "GROOT_QUAD16_H2 task=$TASK node=$NODE frames=$FRAMES per_gpu=$BS accum=1 global=$GB steps=$STEPS"
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$GR:$H" GROOT_SRC="$GR" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HB_AUG_LOG="$LOGD/aug-node-$NODE" HB_PEAK_FILE="$LOGD/peak-production-n$NODE" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$GR"
"$ENV/bin/torchrun" --nnodes 4 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_launch.py" --base_model_path "$SRC" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" --tune_llm --tune_visual \
  --num_gpus 32 --output_dir "$OUT" --save_steps "$SAVE" --save_total_limit "$KEEP" --max_steps "$STEPS" \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate "$LR" --global_batch_size "$GB" --gradient_accumulation_steps 1 \
  --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1
rc=$?
echo "$rc" > "$LOGD/exit-node-$NODE"
if [ "$rc" = 0 ] && [ "$NODE" = 0 ]; then
  python3 - "$ROOT/production-complete.json" <<EOF
import json, sys
d = {"task": "$TASK", "frames": $FRAMES, "source": "$SRC", "recipe": "full fine-tune, fixed steps, 4 nodes x 8 x 16, no accumulation (fallback)",
     "per_gpu_batch": $BS, "gradient_accumulation_steps": 1, "num_nodes": 4, "world_size": 32, "global_batch": $GB,
     "max_steps": $STEPS, "learning_rate": $LR, "warmup_ratio": 0.05, "save_steps": $SAVE, "save_list": $SAVE_LIST,
     "epochs_equivalent": round($STEPS * $GB / $FRAMES, 2)}
json.dump(d, open(sys.argv[1], "w"), indent=1)
EOF
fi
echo "EXIT $rc"; exit $rc
