#!/bin/bash
# 2026-11-15 17:xx user: Psi-Zero B6 continuation on the v6 data (HF b6-pedal-bin @ad2f5bf0, repo root = v6, all 542
# episodes, task key b6v6) from the last Psi-Zero B6 v5 checkpoint (psi0-runs/b6/aligned ckpt_4000: first 100 v5
# episodes, 4 x 8 x 16 = 512, 4000 steps). 4000 steps (6.2 epochs of v6), weights only (fresh optimizer), warmup 200
# steps (5 %) to a REDUCED peak = 0.3 x every original learning rate (user rule): action header / mm projector
# 1e-4 -> 3e-5, vision tower 1e-5 -> 3e-6, language backbone 1e-6 -> 3e-7; cosine to 0; checkpoints 2000 and 4000;
# 2 nodes x 8 x 32 = 512 (no accumulation), HB aug v2 + A20 proprio, normalization statistics of the v6 data (as in
# the basketv2 continuation). Runs on DLC ranks 0,1 (called by psi0_b6v6cont_b5.sh). Output psi0-runs/b6v6/cont-aligned.
set -uo pipefail
TASK=b6v6 NN=2 BS=32 PORT=30900 STEPS=4000 SAVE=2000 WARM=200
R=${RANK:?}; MASTER=${MASTER_ADDR:?}
case "$R" in 0|1) ;; *) echo "psi0_cont_b6v6.sh runs on ranks 0,1 only"; exit 2 ;; esac
H0=$(dirname "$(realpath "$0")"); G=$(dirname "$H0"); export G
H=$G/hb_psi0; PSI=$G/HumanWAM/third_party/Psi0; ENV=$G/envs/psi0; PY=$ENV/bin/python
SRCRUN=$G/psi0-runs/b6/aligned/train/finetune/hb-b6.hb.flow1000.cosine.lr1.0e-04.b512.gpus32.aligned
export HB_INIT_WEIGHTS=$SRCRUN/checkpoints/ckpt_4000/model.safetensors
export HB_STATS_PATH=$G/psi0-data/$TASK/meta/stats_psi0.json
unset HB_DELAY_PROB
ROOT=$G/psi0-runs/$TASK/cont-aligned; LOGD=$ROOT/logs; DEC=$ROOT/decision.json
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/node-$R.log") 2>&1
date -u; hostname; echo "PSI0_CONT task=$TASK rank=$R master=$MASTER port=$PORT per_gpu=$BS nnodes=$NN global=512 steps=$STEPS init=$HB_INIT_WEIGHTS stats=$HB_STATS_PATH"
[ -f "$ROOT/production-complete.json" ] && { echo "already complete"; exit 0; }
[ -s "$HB_INIT_WEIGHTS" ] && [ -s "$HB_STATS_PATH" ] || { echo "missing init weights / stats"; exit 4; }
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export HB_WORLD=$((NN * 8)) PATH="$ENV/bin:$PATH" PYTHONPATH="$PSI/src:$H" PSI_SRC="$PSI" HB_AUG_DIR="$G/hb_groot"
export HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE="/tmp/hb_psi0_hfds_$(id -u)"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TQDM_MININTERVAL=30
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -n 65535 || true
cd "$PSI"
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
export HB_BS=$BS HB_NN=$NN HB_STEPS=$STEPS HB_SAVE=$SAVE HB_WARM=$WARM
if [ "$R" = 0 ] && [ ! -s "$DEC" ]; then
  "$PY" - "$TASK" "$DEC" <<'PY'
import json, os, sys
import hb_args as HA
t, out = sys.argv[1], sys.argv[2]
r = HA.receipt(t); d = HA.plan(r["frames"], int(os.environ["HB_BS"]))
assert d["global_batch"] == 512, d
st, sv, wu = int(os.environ["HB_STEPS"]), int(os.environ["HB_SAVE"]), int(os.environ["HB_WARM"])
d.update(steps=st, save_steps=sv, warmup_steps=wu, epochs_equivalent=round(st * 512 / r["frames"], 2),
         lr_scale=0.3, learning_rates={"action_header": 3e-5, "mm_projector": 3e-5, "vision_tower": 3e-6, "lang_backbone": 3e-7})
d.update(task=t, robot=r.get("robot"), state_dim=r["state_dim"], action_dim=r["action_dim"], image_keys=r["image_keys"],
         num_nodes=int(os.environ["HB_NN"]), continued_from=os.environ["HB_INIT_WEIGHTS"], stats_path=os.environ["HB_STATS_PATH"],
         dataset="humanoid-benchmark/b6-pedal-bin@ad2f5bf0 (v6 at the repo root, all 542 episodes)",
         source_run={"frames": 65127, "episodes": "v5 0-99", "global_batch": 512, "steps": 4000},
         note="continuation, weights only; 2 nodes x 8 GPUs x 32 = 512")
json.dump(d, open(out, "w"), indent=2); print("DECISION", d)
PY
fi
for i in $(seq 240); do [ -s "$DEC" ] && break; sleep 5; done
mapfile -t ARGS < <("$PY" "$H/hb_args.py" args "$TASK" --decision "$DEC" --out "$ROOT/train" --timestamp cont-aligned)
[ "${#ARGS[@]}" -gt 50 ] || { echo "hb_args failed"; exit 5; }
export HB_CONT_LR_SCALE=0.3
for i in "${!ARGS[@]}"; do
  case "${ARGS[$i]}" in
    --train.learning_rate=1e-4) ARGS[$i]=--train.learning_rate=3e-5 ;;
    --model.mm-projector-lr=1e-4) ARGS[$i]=--model.mm-projector-lr=3e-5 ;;
    --model.vision-tower-lr=1e-5) ARGS[$i]=--model.vision-tower-lr=3e-6 ;;
    --model.lang-backbone-lr=1e-6) ARGS[$i]=--model.lang-backbone-lr=3e-7 ;;
  esac
done
for a in --train.learning_rate=3e-5 --model.mm-projector-lr=3e-5 --model.vision-tower-lr=3e-6 --model.lang-backbone-lr=3e-7 \
         --train.max_training_steps=$STEPS --train.warmup_steps=$WARM --train.checkpointing_steps=$SAVE; do
  printf '%s\n' "${ARGS[@]}" | grep -qx -- "$a" || { echo "continuation arg not applied: $a"; exit 7; }
done
printf '%s\n' "${ARGS[@]}" | grep -q -- "--data.transform.field.stat-path=$HB_STATS_PATH" || { echo "stats path not applied"; exit 6; }
[ "$R" != 0 ] || printf '%s\n' "${ARGS[@]}" > "$ROOT/train_args.txt"
export HB_AUG_LOG="$LOGD/aug-node$R"
torchrun --nnodes $NN --node_rank "$R" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" "$H/hb_train.py" "${ARGS[@]}"
rc=$?
echo "$rc" > "$LOGD/prod-node-$R.exit"
[ "$R" = 0 ] && [ "$rc" = 0 ] && cp "$DEC" "$ROOT/production-complete.json"
echo "EXIT $rc"; exit $rc
