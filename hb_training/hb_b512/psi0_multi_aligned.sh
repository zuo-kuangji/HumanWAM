#!/bin/bash
# 2026-11-12 ALIGNED RE-TRAIN copy of psi0_multi.sh (output psi0-runs/<task>/aligned; GR00T-A20 proprio: per-sample joint bias, no delay / history drop).
# Psi-Zero 20-epoch run on NN nodes, global batch 512 = NN x 8 x BS (BS = 512 / (8 NN)) (user 2026-11-11 20:27: every not-yet-started
# GR00T / pi0.5 / Psi-Zero run uses batch 512). Usage: psi0_multi.sh <task> <nnodes> <rank> <master_addr> <port> <per_gpu>
set -uo pipefail
TASK=$1 NN=$2 R=$3 MASTER=$4 PORT=$5 BS=$6
H0=$(dirname "$(realpath "$0")"); G=$(dirname "$H0"); export G
H=$G/hb_psi0; PSI=$G/HumanWAM/third_party/Psi0; ENV=$G/envs/psi0; PY=$ENV/bin/python
ROOT=$G/psi0-runs/$TASK/aligned; LOGD=$ROOT/logs; DEC=$ROOT/decision.json
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/node-$R.log") 2>&1
date -u; hostname; echo "PSI0_PAIR task=$TASK rank=$R master=$MASTER port=$PORT per_gpu=$BS nnodes=$NN global=512"
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
import json, sys
import hb_args as HA
t, out = sys.argv[1], sys.argv[2]
r = HA.receipt(t); import os; d = HA.plan(r["frames"], int(os.environ["HB_BS"]))
assert d["global_batch"] == 512, d
d.update(task=t, robot=r.get("robot"), state_dim=r["state_dim"], action_dim=r["action_dim"], image_keys=r["image_keys"],
         num_nodes=int(os.environ['HB_NN']), note="%s nodes x 8 GPUs x %s = 512" % (os.environ["HB_NN"], os.environ["HB_BS"]))
json.dump(d, open(out, "w"), indent=2); print("DECISION", d)
PY
fi
for i in $(seq 240); do [ -s "$DEC" ] && break; sleep 5; done
mapfile -t ARGS < <("$PY" "$H/hb_args.py" args "$TASK" --decision "$DEC" --out "$ROOT/train" --timestamp aligned)
[ "${#ARGS[@]}" -gt 50 ] || { echo "hb_args failed"; exit 5; }
[ "$R" != 0 ] || printf '%s\n' "${ARGS[@]}" > "$ROOT/train_args.txt"
export HB_AUG_LOG="$LOGD/aug-node$R"
torchrun --nnodes $NN --node_rank "$R" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" "$H/hb_train.py" "${ARGS[@]}"
rc=$?
echo "$rc" > "$LOGD/prod-node-$R.exit"
[ "$R" = 0 ] && [ "$rc" = 0 ] && cp "$DEC" "$ROOT/production-complete.json"
echo "EXIT $rc"
echo "EXIT $rc"; exit $rc
