#!/bin/bash
# 2026-11-11: pool jobs "groot act:<task>" (hb_pool.sh knows only groot|pi05) run ACT single-node via hb_act/act_single.sh
case ${1:-} in act:*) HB_G0=$(dirname "$(dirname "$(realpath "$0")")"); exec bash "$HB_G0/hb_act/act_single.sh" "${1#act:}" "$2" "$HB_G0/act-runs-b696" ;; esac
case ${1:-} in psi0probe:*) HB_G0=$(dirname "$(dirname "$(realpath "$0")")"); exec bash "$HB_G0/hb_psi0/psi0_probe_single.sh" "${1#psi0probe:}" ;; esac
# Single-node GR00T 20-epoch production on THIS node's 8 GPUs (torchrun --standalone), for the task pool (hb_pool.sh).
# Usage: groot_single.sh <task> <per_gpu_batch> <runs_root>      e.g. groot_single.sh b6 87 $G/groot-runs-b696
# Same recipe as groot_node.sh (lr 1e-4, warmup 5 %, wd 1e-5, HB aug v2, 20 epochs, saves at epoch 10 and 20) except
# 1 node x 8 GPUs and the given per-GPU batch (user 2026-11-10: global batch 696 = 87 x 8 for GR00T and pi0.5).
set -euo pipefail
TASK=$1 BS=$2 GRUNS=$3
CFG=g1_feedback76_h16.py
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm
if [[ $TASK = a1[78] ]];then CFG=h2_body31_hand44_h16.py;export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm,head;fi
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
GR=$G/HumanWAM/third_party/Isaac-GR00T
ENV=$G/envs/groot-py312
OUT=$GRUNS/$TASK/production
LOGD=$GRUNS/$TASK/logs
mkdir -p "$OUT" "$LOGD"
exec > >(tee -a "$LOGD/production.log") 2>&1
date -u; hostname
FRAMES=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["frames"])' "$G/groot-data/$TASK/hb-dataset-receipt.json")
GB=$((BS * 8))
STEPS=$(( (20 * FRAMES + GB - 1) / GB ))
SAVE=$(( (STEPS + 1) / 2 ))
echo "GROOT_SINGLE task=$TASK frames=$FRAMES per_gpu=$BS global=$GB steps=$STEPS save=$SAVE out=$OUT"
if ! ldconfig -p | grep -q 'libavutil.so.56'; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
    libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 libavdevice58 libavfilter7
fi
unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}"
export PATH="$ENV/bin:$PATH" PYTHONPATH="$GR:$H" GROOT_SRC="$GR" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HB_AUG_LOG="$LOGD/aug" HB_PEAK_FILE="$LOGD/peak-production" OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$GR"
set +e
"$ENV/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 8 "$H/hb_launch.py" \
  --base_model_path "$G/models/groot-n17-base" --dataset_path "$G/groot-data/$TASK" \
  --embodiment_tag NEW_EMBODIMENT --modality_config_path "$H/$CFG" \
  --num_gpus 8 --output_dir "$OUT" --save_steps "$SAVE" --save_total_limit 2 --max_steps "$STEPS" \
  --warmup_ratio 0.05 --weight_decay 1e-5 --learning_rate 1e-4 --global_batch_size "$GB" \
  --dataloader_num_workers 8 --shard_size 1024 --num_shards_per_epoch 100000 --episode_sampling_rate 0.1
rc=$?
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
  printf '{"task": "%s", "frames": %s, "epochs": 20, "per_gpu_batch": %s, "world_size": 8, "global_batch": %s, "max_steps": %s, "save_steps": %s}\n' \
    "$TASK" "$FRAMES" "$BS" "$GB" "$STEPS" "$SAVE" > "$GRUNS/$TASK/production-complete.json"
fi
echo "EXIT $rc"
exit $rc
