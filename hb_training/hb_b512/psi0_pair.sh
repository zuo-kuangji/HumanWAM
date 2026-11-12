#!/bin/bash
# Psi-Zero 20-epoch run on a PAIR of nodes, global batch 512 = 2 x 8 x 32 (user 2026-11-11 20:27: every not-yet-started
# GR00T / pi0.5 / Psi-Zero run uses batch 512). Usage: psi0_pair.sh <task> <rank 0|1> <master_addr> <port>
set -uo pipefail
TASK=$1 R=$2 MASTER=$3 PORT=$4
H0=$(dirname "$(realpath "$0")"); G=$(dirname "$H0"); export G
H=$G/hb_psi0; PSI=$G/HumanWAM/third_party/Psi0; ENV=$G/envs/psi0; PY=$ENV/bin/python
ROOT=$G/psi0-runs/$TASK/b512; LOGD=$ROOT/logs; DEC=$ROOT/decision.json
mkdir -p "$LOGD"
exec > >(tee -a "$LOGD/node-$R.log") 2>&1
date -u; hostname; echo "PSI0_PAIR task=$TASK rank=$R master=$MASTER port=$PORT per_gpu=32 world=16 global=512"
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export HB_WORLD=16 PATH="$ENV/bin:$PATH" PYTHONPATH="$PSI/src:$H" PSI_SRC="$PSI" HB_AUG_DIR="$G/hb_groot"
export HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE="/tmp/hb_psi0_hfds_$(id -u)"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TQDM_MININTERVAL=30
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
ulimit -n 65535 || true
cd "$PSI"
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
if [ "$R" = 0 ] && [ ! -s "$DEC" ]; then
  "$PY" - "$TASK" "$DEC" <<'PY'
import json, sys
import hb_args as HA
t, out = sys.argv[1], sys.argv[2]
r = HA.receipt(t); d = HA.plan(r["frames"], 32)
assert d["global_batch"] == 512, d
d.update(task=t, robot=r.get("robot"), state_dim=r["state_dim"], action_dim=r["action_dim"], image_keys=r["image_keys"],
         num_nodes=2, note="2 nodes x 8 GPUs x 32 = 512 (user 2026-11-11 20:27)")
json.dump(d, open(out, "w"), indent=2); print("DECISION", d)
PY
fi
for i in $(seq 240); do [ -s "$DEC" ] && break; sleep 5; done
mapfile -t ARGS < <("$PY" "$H/hb_args.py" args "$TASK" --decision "$DEC" --out "$ROOT/train" --timestamp b512)
[ "${#ARGS[@]}" -gt 50 ] || { echo "hb_args failed"; exit 5; }
[ "$R" != 0 ] || printf '%s\n' "${ARGS[@]}" > "$ROOT/train_args.txt"
export HB_AUG_LOG="$LOGD/aug-node$R"
torchrun --nnodes 2 --node_rank "$R" --nproc_per_node 8 --master_addr "$MASTER" --master_port "$PORT" "$H/hb_train.py" "${ARGS[@]}"
rc=$?
echo "$rc" > "$LOGD/prod-node-$R.exit"
[ "$R" = 0 ] && [ "$rc" = 0 ] && cp "$DEC" "$ROOT/production-complete.json"
echo "EXIT $rc"
# 2026-11-12 (user: idle time first): pair B (nodes 2/3) ends Psi-Zero a17 ~1.7 h before pair A ends Psi-Zero B7 and would
# wait at the relay barrier; fill that gap with the GR00T batch-256 single-node runs (8 x 32 = 256, 20 epochs).
if [ "$TASK" = a17 ]; then
  gfree() { for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 5; done; }
  case $R in
    0) gfree; bash "$G/hb_groot/groot_single.sh" a7 32 "$G/groot-runs-b256"; gfree; bash "$G/hb_groot/groot_single.sh" a3 32 "$G/groot-runs-b256" ;;
    1) for i in $(seq 240); do [ -s "$G/groot-data/a6/hb-dataset-receipt.json" ] && break; sleep 30; done
       gfree; bash "$G/hb_groot/groot_single.sh" a6 32 "$G/groot-runs-b256" ;;
  esac
fi
exit $rc
