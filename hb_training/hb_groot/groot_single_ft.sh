#!/bin/bash
# Single-node GR00T N1.7 production, FULL fine-tuning, fixed 4000 optimizer steps (user 2026-11-12 19:4x):
#   - every module trainable: --tune_llm --tune_visual (projector, diffusion action head and VLLN are trainable by default)
#   - global batch 256 = 8 GPUs x 32; if that does not fit, 8 GPUs x 16 x gradient accumulation 2 (still 256)
#   - 4000 steps, checkpoints at step 2000 and 4000 (HF: <task>/b256/step-002000, step-004000)
#   - lr 1e-4, warmup 5 %, weight decay 1e-5, HB augmentation v2 + GR00T A20 proprio (hb_launch.py), as before
# Usage: groot_single_ft.sh <task> <per_gpu_batch> <runs_root>      e.g. groot_single_ft.sh b6 32 $G/groot-runs-ft4k
set -euo pipefail
TASK=$1 BS=$2 GRUNS=$3
STEPS=${HB_GROOT_STEPS:-4000}
SAVE=$((STEPS / 2))
CFG=g1_feedback76_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm
if [[ $TASK = a1[478] ]]; then CFG=h2_body31_hand44_h16.py; export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm,head; fi
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
if [[ $TASK = a1[478] ]]; then  # every H2 task trains with head + chest + wrist (user 2026-11-12 22:0x)
  CFG=h2w3_body31_hand44_h16.py
  [ -f "$H/$CFG" ] || { echo "H2 3-camera modality config $H/$CFG missing"; exit 3; }
  python3 -c "import json,sys; v=list(json.load(open(sys.argv[1]))['video']); sys.exit(0 if v==['head','chest','wrist'] else 4)" "$G/groot-data/$TASK/meta/modality.json" \
    || { echo "groot-data/$TASK is not the 3-camera view"; exit 4; }
fi
GR=$G/HumanWAM/third_party/Isaac-GR00T
ENV=$G/envs/groot-py312
OUT=$GRUNS/$TASK/production
LOGD=$GRUNS/$TASK/logs
mkdir -p "$OUT" "$LOGD"
exec > >(tee -a "$LOGD/production.log") 2>&1
date -u; hostname
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$G/groot-data/$TASK/hb-dataset-receipt.json")
GB=$((BS * 8))
echo "GROOT_SINGLE_FT task=$TASK frames=$FRAMES per_gpu=$BS global=$GB steps=$STEPS save=$SAVE full_finetune=1 out=$OUT"
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7
fi
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$GR:$H" GROOT_SRC="$GR" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HB_AUG_LOG="$LOGD/aug" HB_PEAK_FILE="$LOGD/peak-production" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$GR"
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
run() {  # $1 = per-GPU micro batch, $2 = gradient accumulation; global = 8 x $1 x $2
  "$ENV/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 8 "$H/hb_launch.py" \
    --base_model_path "$G/models/groot-n17-base" --dataset_path "$G/groot-data/$TASK" \
    --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
    --tune_llm --tune_visual \
    --num_gpus 8 --output_dir "$OUT" --save_steps "$SAVE" --save_total_limit 2 --max_steps "$STEPS" \
    --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
    --global_batch_size "$((8 * $1))" --gradient_accumulation_steps "$2" \
    --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1
}
MB=$BS ACC=1
set +e
run "$MB" "$ACC"; rc=$?
if [ "$rc" != 0 ] && grep -qi "out of memory" "$LOGD/production.log" && [ "$BS" -ge 2 ]; then
  echo "$(date -u) OOM at per-GPU $BS: retrying with per-GPU $((BS / 2)) x gradient accumulation 2 (global batch unchanged)"
  rm -rf "$OUT"; mkdir -p "$OUT"; sleep 30
  MB=$((BS / 2)) ACC=2
  run "$MB" "$ACC"; rc=$?
fi
set -e
if [ "$rc" = 0 ]; then
  python3 - "$OUT" <<'EOF'
import pathlib, shutil, sys
out = pathlib.Path(sys.argv[1])
ck = sorted((p for p in out.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()), key=lambda p: int(p.name.split("-")[-1]))
for c in ck[:-1]:  # resume state only for the latest checkpoint (weights kept)
    for p in list(c.glob("global_step*")) + list(c.glob("optimizer*.pt")) + list(c.glob("scheduler.pt")) + list(c.glob("rng_state*.pth")) + list(c.glob("latest")) + list(c.glob("zero_to_fp32.py")):
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    print("resume state dropped:", c.name)
EOF
  printf '{"task": "%s", "frames": %s, "recipe": "full fine-tune (tune_llm, tune_visual, projector, diffusion, vlln), fixed steps", "per_gpu_batch": %s, "gradient_accumulation_steps": %s, "world_size": 8, "global_batch": %s, "max_steps": %s, "save_steps": %s, "epochs_equivalent": %s}\n' \
    "$TASK" "$FRAMES" "$MB" "$ACC" "$GB" "$STEPS" "$SAVE" "$(python3 -c "print(round($STEPS*$GB/$FRAMES,2))")" > "$GRUNS/$TASK/production-complete.json"
fi
echo "EXIT $rc"
exit $rc
