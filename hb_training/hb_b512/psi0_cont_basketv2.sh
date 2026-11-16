#!/bin/bash
# 2026-11-14 user: Psi-Zero basket continuation on a20-basket-transfer-pre-reach-trim-expert-v2 (task key basketv2),
# 2000 more steps (user 16:xx: 2000, not 4000) from psi0-runs/basket/aligned ckpt_4000 (weights only, fresh optimizer),
# warmup 100 steps (5 %) to a REDUCED peak = 0.3 x every original learning rate (user rule: a continuation never
# re-warms to the full peak): action header / mm projector 1e-4 -> 3e-5, vision tower 1e-5 -> 3e-6, language
# backbone 1e-6 -> 3e-7; cosine to 0; checkpoints 1000 and 2000; same
# recipe as the aligned run: 4 nodes x 8 x 16 = 512, HB aug v2 + A20 proprio, normalization stats of the basketv2 data.
# Output psi0-runs/basketv2/cont-aligned.
set -uo pipefail
TASK=basketv2 NN=4 BS=16 PORT=30741
R=${RANK:?}; MASTER=${MASTER_ADDR:?}
H0=$(dirname "$(realpath "$0")"); G=$(dirname "$H0"); export G
H=$G/hb_psi0; PSI=$G/HumanWAM/third_party/Psi0; ENV=$G/envs/psi0; PY=$ENV/bin/python
SRCRUN=$G/psi0-runs/basket/aligned/train/finetune/hb-basket.hb.flow1000.cosine.lr1.0e-04.b512.gpus32.aligned
export HB_INIT_WEIGHTS=$SRCRUN/checkpoints/ckpt_4000/model.safetensors
# 16:1x: normalization statistics of the NEW data (basketv2), consistent with the GR00T continuation, whose
# trainer recomputes them from the training data by default (override_pretraining_statistics=True)
export HB_STATS_PATH=$G/psi0-data/basketv2/meta/stats_psi0.json
ROOT=$G/psi0-runs/$TASK/cont-aligned; LOGD=$ROOT/logs; DEC=$ROOT/decision.json
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/node-$R.log") 2>&1
date -u; hostname; echo "PSI0_CONT task=$TASK rank=$R master=$MASTER port=$PORT per_gpu=$BS nnodes=$NN global=512 init=$HB_INIT_WEIGHTS stats=$HB_STATS_PATH"
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
export HB_BS=$BS HB_NN=$NN
if [ "$R" = 0 ] && [ ! -s "$DEC" ]; then
  "$PY" - "$TASK" "$DEC" <<'PY'
import json, os, sys
import hb_args as HA
t, out = sys.argv[1], sys.argv[2]
r = HA.receipt(t); d = HA.plan(r["frames"], int(os.environ["HB_BS"]))
assert d["global_batch"] == 512, d
d.update(steps=2000, save_steps=1000, warmup_steps=100, epochs_equivalent=round(2000 * 512 / r["frames"], 2),
         lr_scale=0.3, learning_rates={"action_header": 3e-5, "mm_projector": 3e-5, "vision_tower": 3e-6, "lang_backbone": 3e-7})
d.update(task=t, robot=r.get("robot"), state_dim=r["state_dim"], action_dim=r["action_dim"], image_keys=r["image_keys"],
         num_nodes=int(os.environ["HB_NN"]), continued_from=os.environ["HB_INIT_WEIGHTS"], stats_path=os.environ["HB_STATS_PATH"],
         dataset="humanoid-benchmark/a20-basket-transfer-pre-reach-trim-expert-v2@c4df4d40",
         note="continuation, weights only; 4 nodes x 8 GPUs x 16 = 512")
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
         --train.max_training_steps=2000 --train.warmup_steps=100 --train.checkpointing_steps=1000; do
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
