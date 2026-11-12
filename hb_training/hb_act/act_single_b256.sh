#!/bin/bash
# Single-node ACT 20-epoch production on THIS node's 8 GPUs (torchrun --standalone), for the task pool (hb_pool.sh).
# Usage: act_single.sh <task> <per_gpu_batch> <runs_root>      e.g. act_single.sh b6 87 $G/act-runs-b696
# User 2026-11-12: ACT back, global batch 256 = 32 x 8, one node per job, every task, after the aligned re-trains; proprio aligned with GR00T A20 (no delay / history drop).
# Recipe otherwise = act_node.sh (Psi0 ACT, chunk 16, VAE kl 10, lr 1e-4 cosine, warmup 2.5 %, HB aug v2, native
# ColorJitter off, saves at epoch 10 and 20, resume state only for the last checkpoint).
# GPU trial first: HB_TRIAL_STEPS steps of the real 8-GPU DDP job at the production batch (probe mode: peak memory and
# step time per rank); production starts only when every rank finished OK and peak reserved memory <= 92 %.
set -euo pipefail
TASK=$1 BS=$2 RUNS=$3
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
export G HB_WORLD=8
PSI=$G/HumanWAM/third_party/Psi0
ENV=$G/envs/psi0
PY=$ENV/bin/python
ROOT=$RUNS/$TASK
LOGD=$ROOT/logs
TRIAL=$ROOT/trial
DEC=$ROOT/decision.json
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/production.log") 2>&1
date -u; hostname; echo "ACT_SINGLE task=$TASK per_gpu=$BS world=8 root=$ROOT"
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export PATH="$ENV/bin:$PATH" PYTHONPATH="$PSI/src:$H:$G/hb_psi0" PSI_SRC="$PSI" HB_AUG_DIR="${HB_AUG_DIR:-$G/hb_groot}"
export TORCH_HOME="$G/torch-home" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE="/tmp/hb_act_hfds_$(id -u)"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TQDM_MININTERVAL=30
export NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -n 65535 || true
cd "$PSI"
test -s "$G/torch-home/hub/checkpoints/resnet18-f37072fd.pth" || { echo "ResNet18 ImageNet weights missing in $G/torch-home"; exit 2; }

if [ -s "$DEC" ]; then
  echo "reusing $DEC"
else
  rm -rf "$TRIAL"; mkdir -p "$TRIAL/bs$BS"
  TS=${HB_TRIAL_STEPS:-20}
  mapfile -t PA < <("$PY" "$H/hb_act_args.py" args "$TASK" --batch "$BS" --steps "$TS" --save "$TS" --warmup 0 \
      --out "$TRIAL/bs$BS/run" --timestamp "trial$BS" --no-resume)
  [ "${#PA[@]}" -gt 50 ] || { echo "hb_act_args failed for the trial"; exit 5; }
  echo "GPU trial: $TS steps, 8 GPUs x $BS"
  set +e
  HB_PROBE_OUT="$TRIAL/bs$BS" HB_AUG_LOG="$TRIAL/bs$BS/aug" \
    "$ENV/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 8 "$H/hb_act_train.py" "${PA[@]}" > "$TRIAL/bs$BS.log" 2>&1
  echo $? > "$TRIAL/bs$BS.rc"
  set -e
  echo "trial rc=$(cat "$TRIAL/bs$BS.rc")"
  "$PY" "$H/hb_act_args.py" decide "$TASK" "$TRIAL" "$DEC" "$BS" || { echo "GPU_TRIAL_FAILED (see $TRIAL/bs$BS.log)"; tail -n 30 "$TRIAL/bs$BS.log"; exit 4; }
fi
cat "$DEC"
"$PY" - "$DEC" "$BS" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["per_gpu_batch"] == int(sys.argv[2]) and d["global_batch"] == 8 * int(sys.argv[2]), d
assert d["state_dim"] in (44, 78), d["state_dim"]
print("DECISION_OK per_gpu", d["per_gpu_batch"], "global", d["global_batch"], "steps", d["steps"], "save", d["save_steps"], "est_hours", d.get("est_hours"))
EOF
mapfile -t ARGS < <("$PY" "$H/hb_act_args.py" args "$TASK" --decision "$DEC" --out "$ROOT/train" --timestamp b256)
[ "${#ARGS[@]}" -gt 50 ] || { echo "hb_act_args failed"; exit 5; }
printf '%s\n' "${ARGS[@]}" > "$ROOT/train_args.txt"
export HB_AUG_LOG="$LOGD/aug"
set +e
"$ENV/bin/torchrun" --standalone --nnodes 1 --nproc_per_node 8 "$H/hb_act_train.py" "${ARGS[@]}"
rc=$?
set -e
[ "$rc" != 0 ] || cp "$DEC" "$ROOT/production-complete.json"
echo "EXIT $rc"
exit $rc
