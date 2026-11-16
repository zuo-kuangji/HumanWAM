#!/bin/bash
# 2026-11-14 user: GR00T basket continuation on a20-basket-transfer-pre-reach-trim-expert-v2 (task key basketv2),
# 4000 more steps from the final checkpoint, 2 nodes x 8 GPUs x 32 per GPU, no gradient accumulation = 512 (user rule
# 2026-11-14), full fine-tune, lr 1e-4, warmup 5 %, wd 1e-5, HB aug v2 + A20 proprio (hb_launch.py), checkpoints every
# 500 from step 2000. Weights only (fresh optimizer / LR schedule); the checkpoint's normalization statistics are kept
# (Gr00t processor loaded from the checkpoint, set_statistics without override).
#   nodes 0,1 -> run from-b512 (groot-runs-ft512/basket checkpoint-4000)
#   nodes 2,3 -> run from-b256 (groot-runs-ft4k/basket  checkpoint-4000)
# Output groot-runs-cont/<run>/production.
set -uo pipefail
TASK=basketv2 BS=32 ACCUM=1 STEPS=4000 SAVE=500 KEEP=5 CFG=g1_feedback76_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T; ENV=$G/envs/groot-py312
RK=${RANK:?DLC RANK missing}; MASTER0=${MASTER_ADDR:?DLC MASTER_ADDR missing}; [ "${WORLD_SIZE:?}" = 4 ]
case "$RK" in
  0|1) RUN=from-b512 SRC=$G/groot-runs-ft512/basket/production/checkpoint-4000 NR=$RK PORT=29680 ;;
  2|3) RUN=from-b256 SRC=$G/groot-runs-ft4k/basket/production/checkpoint-4000 NR=$((RK - 2)) PORT=29681 ;;
esac
GRUNS=$G/groot-runs-cont; ROOT=$GRUNS/$RUN; OUT=$ROOT/production; LOGD=$ROOT/logs; mkdir -p "$OUT" "$LOGD"
LOGF=$LOGD/production.log; [ "$NR" = 0 ] || LOGF=$LOGD/production-node-$NR.log
exec > >(tee -a "$LOGF") 2>&1
date -u; hostname
if [ -f "$ROOT/production-complete.json" ]; then echo "already complete"; exit 0; fi
[ -f "$SRC/model.safetensors.index.json" ] && [ -f "$SRC/statistics.json" ] || { echo "missing source checkpoint $SRC"; exit 4; }
MF=$ROOT/master-addr
if [ "$RK" = 0 ]; then MASTER=$MASTER0
elif [ "$RK" = 2 ]; then MASTER=$(hostname -I | awk '{print $1}'); echo "$MASTER $(date +%s)" > "$MF"
else
  if [ "$RK" = 1 ]; then MASTER=$MASTER0
  else for i in $(seq 120); do [ -s "$MF" ] && [ $(( $(date +%s) - $(cut -d' ' -f2 "$MF") )) -lt 3600 ] && break; sleep 5; done
       MASTER=$(cut -d' ' -f1 "$MF"); fi
fi
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$G/groot-data/$TASK/hb-dataset-receipt.json")
GB=$((BS * 16)); EFF=$((GB * ACCUM))
echo "GROOT_CONT run=$RUN task=$TASK rank=$RK node_rank=$NR master=$MASTER:$PORT src=$SRC frames=$FRAMES per_gpu=$BS accum=$ACCUM global=$EFF steps=$STEPS save_every=$SAVE keep=$KEEP out=$OUT"
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
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
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
  printf '{"task": "basketv2", "dataset": "humanoid-benchmark/a20-basket-transfer-pre-reach-trim-expert-v2@c4df4d40", "frames": %s, "continued_from": "%s", "recipe": "full fine-tune continuation (weights only, fresh optimizer and LR schedule, source normalization statistics kept)", "per_gpu_batch": %s, "gradient_accumulation_steps": %s, "world_size": 16, "global_batch": %s, "max_steps": %s, "save_steps": %s, "save_list": [2000, 2500, 3000, 3500, 4000], "epochs_equivalent": %s}\n' \
    "$FRAMES" "$SRC" "$BS" "$ACCUM" "$EFF" "$STEPS" "$SAVE" "$(python3 -c "print(round($STEPS*$EFF/$FRAMES,2))")" > "$ROOT/production-complete.json"
fi
echo "EXIT $rc"; exit $rc
