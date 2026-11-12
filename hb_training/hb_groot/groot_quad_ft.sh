#!/bin/bash
# 2026-11-13: GR00T N1.7 FULL fine-tune on all 4 nodes (relay cmd item; G1 tasks), same recipe as groot_single_ft.sh:
#   --tune_llm --tune_visual (nothing frozen), 4000 steps, checkpoints 2000 / 4000, lr 1e-4, warmup 5 %, wd 1e-5,
#   HB augmentation v2 + GR00T A20 proprio (hb_launch.py); global batch 256 = 4 nodes x 8 GPUs x 8.
# Output groot-runs-ft4k/<task> (auto-upload job "grootft"). Usage: groot_quad_ft.sh <task> [per_gpu=8] [port=29640]
set -uo pipefail
TASK=$1 BS=${2:-8} PORT=${3:-29640}
STEPS=4000 SAVE=2000
CFG=g1_feedback76_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm
[[ $TASK = a1[478] ]] && { echo "H2 task: use groot_ft_guard2.sh (single node)"; exit 3; }
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T; ENV=$G/envs/groot-py312
NODE=${RANK:?DLC RANK missing}; MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}; [ "${WORLD_SIZE:?}" = 4 ]
GRUNS=$G/groot-runs-ft4k; OUT=$GRUNS/$TASK/production; LOGD=$GRUNS/$TASK/logs
mkdir -p "$OUT" "$LOGD"
LOGF=$LOGD/production.log; [ "$NODE" = 0 ] || LOGF=$LOGD/production-node-$NODE.log
exec > >(tee -a "$LOGF") 2>&1
date -u; hostname
if [ -f "$GRUNS/$TASK/production-complete.json" ]; then echo "already complete"; exit 0; fi
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$G/groot-data/$TASK/hb-dataset-receipt.json")
GB=$((BS * 32))
echo "GROOT_QUAD_FT task=$TASK node=$NODE frames=$FRAMES per_gpu=$BS global=$GB steps=$STEPS save=$SAVE full_finetune=1 out=$OUT"
[ "$GB" = 256 ] || { echo "global batch must be 256"; exit 2; }
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7
fi
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$GR:$H" GROOT_SRC="$GR" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HB_AUG_LOG="$LOGD/aug-node-$NODE" HB_PEAK_FILE="$LOGD/peak-production" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$GR"
"$ENV/bin/torchrun" --nnodes 4 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_launch.py" \
  --base_model_path "$G/models/groot-n17-base" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
  --tune_llm --tune_visual \
  --num_gpus 32 --output_dir "$OUT" --save_steps "$SAVE" --save_total_limit 2 --max_steps "$STEPS" \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size "$GB" --gradient_accumulation_steps 1 \
  --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1
rc=$?
echo "$rc" > "$LOGD/exit-node-$NODE"
if [ "$rc" = 0 ] && [ "$NODE" = 0 ]; then
  python3 - "$OUT" <<'EOF'
import pathlib, shutil, sys
out = pathlib.Path(sys.argv[1])
ck = sorted((p for p in out.glob("checkpoint-*") if p.name.split("-")[-1].isdigit()), key=lambda p: int(p.name.split("-")[-1]))
for c in ck[:-1]:  # resume state only for the latest checkpoint (weights kept)
    for p in list(c.glob("global_step*")) + list(c.glob("optimizer*.pt")) + list(c.glob("scheduler.pt")) + list(c.glob("rng_state*.pth")) + list(c.glob("latest")) + list(c.glob("zero_to_fp32.py")):
        shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink(missing_ok=True)
    print("resume state dropped:", c.name)
EOF
  printf '{"task": "%s", "frames": %s, "recipe": "full fine-tune (tune_llm, tune_visual, projector, diffusion, vlln), fixed steps", "per_gpu_batch": %s, "gradient_accumulation_steps": 1, "world_size": 32, "global_batch": %s, "max_steps": %s, "save_steps": %s, "epochs_equivalent": %s}\n' \
    "$TASK" "$FRAMES" "$BS" "$GB" "$STEPS" "$SAVE" "$(python3 -c "print(round($STEPS*$GB/$FRAMES,2))")" > "$GRUNS/$TASK/production-complete.json"
fi
echo "EXIT $rc"; exit $rc
