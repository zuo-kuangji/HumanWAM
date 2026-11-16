#!/bin/bash
# 2026-11-15 17:xx user: GR00T B6 continuation on the v6 data (HF b6-pedal-bin @ad2f5bf0, repo root = v6, all 542
# episodes, task key b6v6) from the last GR00T B6 v5 checkpoint (groot-runs-ft4k/b6 checkpoint-4000: full fine-tune,
# first 100 v5 episodes, 256 x 4000 steps). 4000 steps (6.2 epochs of v6), 2 nodes x 8 GPUs x 32 = 512, no gradient
# accumulation, weights only (fresh AdamW), warmup 5 % to a REDUCED peak lr 3e-5 (= 0.3 x the original 1e-4, user
# rule), cosine to 0, wd 1e-5, HB aug v2 + A20 proprio (hb_launch.py), checkpoints 2000..4000 every 500. Normalization
# statistics are recomputed from the v6 training data (GR00T default override_pretraining_statistics=True; the
# basketv2 continuation did the same: its checkpoint statistics differ from the source checkpoint's).
# Runs on DLC ranks 0,1 (called by groot_b6v6cont_b5.sh). Output groot-runs-cont/b6v6-from-ft4k.
set -uo pipefail
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T; ENV=$G/envs/groot-py312; CFG=g1_feedback76_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm
RK=${RANK:?DLC RANK missing}; MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}; [ "${WORLD_SIZE:?}" = 4 ]
case "$RK" in 0|1) NR=$RK ;; *) echo "groot_cont_b6v6.sh runs on ranks 0,1 only"; exit 2 ;; esac
BS=32 ACCUM=1 SAVE=500 TASK=b6v6 MODE=cont
ROOT=$G/groot-runs-cont/b6v6-from-ft4k SRC=$G/groot-runs-ft4k/b6/production/checkpoint-4000
STEPS=4000 LR=3e-5 KEEP=5 SAVE_LIST="[2000, 2500, 3000, 3500, 4000]" PORT=29750
DATASET="humanoid-benchmark/b6-pedal-bin@ad2f5bf0 (v6 at the repo root, all 542 episodes)"
RECIPE="full fine-tune continuation of the GR00T B6 v5 run (groot-runs-ft4k/b6 checkpoint-4000, first 100 v5 episodes, 256 x 4000): weights only, fresh AdamW, warmup 5 % to peak lr 3e-5 = 0.3 x the original 1e-4, cosine to 0 over 4000 steps; normalization statistics recomputed from the v6 data"
OUT=$ROOT/production; LOGD=$ROOT/logs; mkdir -p "$OUT" "$LOGD"
LOGF=$LOGD/production.log; [ "$NR" = 0 ] || LOGF=$LOGD/production-node-$NR.log
exec > >(tee -a "$LOGF") 2>&1
date -u; hostname
if [ -f "$ROOT/production-complete.json" ]; then echo "already complete"; exit 0; fi
[ -f "$SRC/model.safetensors.index.json" ] && [ -f "$SRC/statistics.json" ] || { echo "missing source checkpoint $SRC"; exit 4; }
REC=$G/groot-data/$TASK/hb-dataset-receipt.json
[ -f "$REC" ] || { echo "dataset $TASK not prepared"; exit 3; }
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$REC")
GB=$((BS * 16)); EFF=$((GB * ACCUM))
echo "GROOT_CONT mode=$MODE task=$TASK rank=$RK node_rank=$NR master=$MASTER:$PORT src=$SRC frames=$FRAMES per_gpu=$BS accum=$ACCUM global=$EFF steps=$STEPS lr=$LR save_every=$SAVE keep=$KEEP out=$OUT"
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
  "$H/hb_launch.py" \
  --base_model_path "$SRC" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
  --tune_llm --tune_visual \
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
d = {"task": "$TASK", "mode": "$MODE", "dataset": "$DATASET", "frames": $FRAMES, "source": "$SRC",
     "recipe": "$RECIPE", "per_gpu_batch": $BS, "gradient_accumulation_steps": $ACCUM, "num_nodes": 2,
     "world_size": 16, "global_batch": $EFF, "max_steps": $STEPS, "learning_rate": $LR, "warmup_ratio": 0.05,
     "lr_schedule": "linear warmup 5 % to the peak, then cosine to 0", "save_steps": $SAVE, "save_list": $SAVE_LIST,
     "epochs_equivalent": round($STEPS * $EFF / $FRAMES, 2), "continued_from": "$SRC", "peak_lr_ratio_to_original": 0.3,
     "source_run": {"frames": 65127, "episodes": "v5 0-99", "global_batch": 256, "max_steps": 4000, "learning_rate": 1e-4}}
json.dump(d, open(sys.argv[1], "w"), indent=1); print("COMPLETE", d)
EOF
fi
echo "EXIT $rc"; exit $rc
