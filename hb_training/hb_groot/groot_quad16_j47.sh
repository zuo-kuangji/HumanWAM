#!/bin/bash
# hb_relay cmd item (2026-11-18 user): GR00T joint47 action experiment on B6 v5 (all 150 episodes, HF b6-pedal-bin v5/),
# aligned with HumanWAM joint47 v2: action = body29 | Inspire12 | root rotation6D (current heading), 40 rows at source
# offsets 0,2,...,78 (stride 2). Same GR00T recipe as the B6 runs otherwise: full fine-tune, global batch 512 =
# 4 nodes x 8 GPUs x 16, no gradient accumulation, 4000 steps, lr 1e-4 cosine, warmup 5 %, wd 1e-5, HB aug v2 +
# proprio perturbation (hb_launch.py), state44 unchanged. Checkpoints 2000..4000 every 500.
# Output groot-runs-j47/b6v5j47/production.
set -uo pipefail
TASK=b6v5j47 BS=16 ACCUM=1 PORT=${1:-29680}
STEPS=4000 SAVE=500
KEEP=$(( STEPS / 2 / SAVE + 1 )); SAVE_LIST=$(python3 -c "print(list(range($STEPS // 2, $STEPS + 1, $SAVE)))")
CFG=g1_joint47_h40.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm HB_J47_ROOT_KEY=root_rot6d HB_INLOOP_EVERY=0
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T; ENV=$G/envs/groot-py312
NODE=${RANK:?DLC RANK missing}; MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}; [ "${WORLD_SIZE:?}" = 4 ]
for i in $(seq 180); do [ -f "$G/groot-data/$TASK/hb-dataset-receipt.json" ] && grep -q J47_CPU_VERIFY_OK "$G/groot-data/prep-$TASK.log" 2>/dev/null && break; sleep 10; done  # data view + CPU check (<= 30 min)
[ -f "$G/groot-data/$TASK/hb-dataset-receipt.json" ] && grep -q J47_CPU_VERIFY_OK "$G/groot-data/prep-$TASK.log" || { echo "SKIP groot $TASK: dataset not prepared / CPU check failed"; exit 3; }
GRUNS=$G/groot-runs-j47; OUT=$GRUNS/$TASK/production; LOGD=$GRUNS/$TASK/logs
mkdir -p "$OUT" "$LOGD"
LOGF=$LOGD/production.log; [ "$NODE" = 0 ] || LOGF=$LOGD/production-node-$NODE.log
exec > >(tee -a "$LOGF") 2>&1
date -u; hostname
if [ -f "$GRUNS/$TASK/production-complete.json" ]; then echo "already complete"; exit 0; fi
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$G/groot-data/$TASK/hb-dataset-receipt.json")
GB=$((BS * 32)); EFF=$((GB * ACCUM))
echo "GROOT_J47 task=$TASK node=$NODE frames=$FRAMES per_gpu=$BS global=$EFF accum=$ACCUM steps=$STEPS save_every=$SAVE keep=$KEEP out=$OUT"
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
export HB_AUG_LOG="$LOGD/aug-node-$NODE" HB_PEAK_FILE="$LOGD/peak-production" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$GR"
"$ENV/bin/torchrun" --nnodes 4 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" \
  "$H/hb_launch.py" \
  --base_model_path "$G/models/groot-n17-base" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
  --tune_llm --tune_visual \
  --num_gpus 32 --output_dir "$OUT" --save_steps "$SAVE" --save_total_limit "$KEEP" --max_steps "$STEPS" \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 \
  --global_batch_size "$GB" --gradient_accumulation_steps "$ACCUM" \
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
  printf '{"task": "%s", "frames": %s, "recipe": "joint47 (body29 | Inspire12 | root rot6D current heading), H40 stride 2, full fine-tune, fixed steps", "per_gpu_batch": %s, "gradient_accumulation_steps": %s, "world_size": 32, "global_batch": %s, "max_steps": %s, "save_steps": %s, "save_list": %s, "epochs_equivalent": %s}\n' \
    "$TASK" "$FRAMES" "$BS" "$ACCUM" "$EFF" "$STEPS" "$SAVE" "$SAVE_LIST" "$(python3 -c "print(round($STEPS*$EFF/$FRAMES,2))")" > "$GRUNS/$TASK/production-complete.json"
fi
echo "EXIT $rc"; exit $rc
