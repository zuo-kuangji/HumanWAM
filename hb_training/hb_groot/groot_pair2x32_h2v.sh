#!/bin/bash
# 2026-11-15 GR00T variant (dropout 0, horizon 32, 16 flow steps, layer-wise lr; env set by the relay wrapper) copy of groot_pair2x32_h2.sh. Relay cmd helper (2026-11-14 user rule): two GR00T trainings in parallel, each global batch 512 = 2 nodes x 8 GPUs x 32
# per GPU, NO gradient accumulation, full fine-tune from the GR00T N1.7 base, 4000 steps, lr 1e-4, warmup 5 %,
# wd 1e-5, checkpoints 2000..4000 every 500, HB aug v2 + A20 proprio (hb_launch.py).
#   groot_pair2x32.sh <task for nodes 0,1> <task for nodes 2,3> <port A> <port B>
# Waits (up to 3 h) for groot-data/<task>. Output groot-runs-2x32/<task> (auto-upload job groot2x32 -> groot/<task dir>/b512/).
set -uo pipefail
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T; ENV=$G/envs/groot-py312; CFG=h2_body31_hand44_h32.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm,head
RK=${RANK:?DLC RANK missing}; MASTER0=${MASTER_ADDR:?DLC MASTER_ADDR missing}; [ "${WORLD_SIZE:?}" = 4 ]
BS=32 ACCUM=1 SAVE=500 STEPS=4000 LR=1e-4 KEEP=5 SAVE_LIST="[2000, 2500, 3000, 3500, 4000]" SRC=$G/models/groot-n17-base
case "$RK" in
  0|1) TASK=$1 NR=$RK PORT=$3 ;;
  2|3) TASK=$2 NR=$((RK - 2)) PORT=$4 ;;
esac
ROOT=$G/groot-runs-h32/$TASK; OUT=$ROOT/production; LOGD=$ROOT/logs; mkdir -p "$OUT" "$LOGD"
LOGF=$LOGD/production.log; [ "$NR" = 0 ] || LOGF=$LOGD/production-node-$NR.log
exec > >(tee -a "$LOGF") 2>&1
date -u; hostname
if [ -f "$ROOT/production-complete.json" ]; then echo "already complete"; exit 0; fi
REC=$G/groot-data/$TASK/hb-dataset-receipt.json
for i in $(seq 1080); do [ -f "$REC" ] && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $REC"; sleep 10; done
[ -f "$REC" ] || { echo "dataset $TASK not prepared"; exit 3; }
MF=$ROOT/master-addr
if [ "$RK" = 0 ] || [ "$RK" = 1 ]; then MASTER=$MASTER0
elif [ "$RK" = 2 ]; then MASTER=$(hostname -I | awk '{print $1}'); echo "$MASTER $(date +%s)" > "$MF"
else
  for i in $(seq 120); do [ -s "$MF" ] && [ $(( $(date +%s) - $(cut -d' ' -f2 "$MF") )) -lt 3600 ] && break; sleep 5; done
  MASTER=$(cut -d' ' -f1 "$MF")
fi
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$REC")
GB=$((BS * 16)); EFF=$((GB * ACCUM))
echo "GROOT_PAIR2X32 task=$TASK rank=$RK node_rank=$NR master=$MASTER:$PORT frames=$FRAMES per_gpu=$BS accum=$ACCUM global=$EFF steps=$STEPS lr=$LR out=$OUT"
[ "$EFF" = 512 ] || { echo "global batch must be 512"; exit 2; }
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7
fi
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$GR:$H" GROOT_SRC="$GR" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HB_AUG_LOG="$LOGD/aug-node-$NR" HB_PEAK_FILE="$LOGD/peak-production-n$NR" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$GR"
"$ENV/bin/torchrun" --nnodes 2 --node_rank "$NR" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_launch_v.py" \
  --base_model_path "$SRC" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
  --tune_llm --tune_visual --state_dropout_prob 0.0 \
  --num_gpus 16 --output_dir "$OUT" --save_steps "$SAVE" --save_total_limit "$KEEP" --max_steps "$STEPS" \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate "$LR" \
  --global_batch_size "$GB" --gradient_accumulation_steps "$ACCUM" \
  --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1
rc=$?
echo "$rc" > "$LOGD/exit-node-$NR"
if [ "$rc" = 0 ] && [ "$NR" = 0 ]; then
  python3 - "$OUT" <<'EOF'
import pathlib, shutil, sys
out = pathlib.Path(sys.argv[1])
ck = sorted((p for p in out.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()), key=lambda p: int(p.name.split("-")[-1]))
for c in ck[:-1]:  # resume state only for the latest checkpoint (weights kept)
    for p in list(c.glob("global_step*")) + list(c.glob("optimizer*.pt")) + list(c.glob("scheduler.pt")) + list(c.glob("rng_state*.pth")) + list(c.glob("latest")) + list(c.glob("zero_to_fp32.py")):
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    print("resume state dropped:", c.name)
EOF
  python3 - "$ROOT/production-complete.json" <<EOF
import json, sys
d = {"task": "$TASK", "frames": $FRAMES, "source": "$SRC",
     "recipe": "full fine-tune (tune_llm, tune_visual, projector, diffusion, vlln), fixed steps, 2 nodes x 8 x 32, no accumulation; variant 2026-11-15: all dropouts 0, action horizon 32, 16 flow sampling steps, layer-wise lr", "model_overrides": ${HB_GROOT_MODEL_OVERRIDES}, "lr_groups": ${HB_GROOT_LR_GROUPS},
     "per_gpu_batch": $BS, "gradient_accumulation_steps": $ACCUM, "num_nodes": 2, "world_size": 16, "global_batch": $EFF,
     "max_steps": $STEPS, "learning_rate": $LR, "warmup_ratio": 0.05, "lr_schedule": "linear warmup 5 %, cosine to 0",
     "save_steps": $SAVE, "save_list": $SAVE_LIST, "epochs_equivalent": round($STEPS * $EFF / $FRAMES, 2)}
json.dump(d, open(sys.argv[1], "w"), indent=1); print("COMPLETE", d)
EOF
fi
echo "EXIT $rc"; exit $rc
