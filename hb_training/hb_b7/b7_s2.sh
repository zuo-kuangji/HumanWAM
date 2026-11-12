#!/bin/bash
# B7 stage 2 (2026-11-11, user): ACT B7 on nodes 0-2, 3 x 8 GPUs x 29 = global 696 (same recipe as the single-node 696
# ACT runs: chunk 16, kl 10, lr 1e-4 cosine, warmup 2.5 %, HB aug v2, epochs 20, saves at epoch 10 and 20, keep 3).
# Node 3 has no job of its own in this stage.
G=$(dirname "$(dirname "$(realpath "$0")")")
NODE=${RANK:?DLC RANK missing}
MASTER=${MASTER_ADDR:?DLC MASTER_ADDR missing}
[ "$NODE" = 3 ] && { echo "node 3: no ACT rank in stage 2"; exit 0; }
H=$G/hb_act
PSI=$G/HumanWAM/third_party/Psi0
ENV=$G/envs/psi0
PY=$ENV/bin/python
ROOT=$G/act-runs-b696/b7
DEC=$ROOT/decision.json
mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/node-$NODE.log") 2>&1
date -u; hostname; echo "ACT_3NODE task=b7 node=$NODE per_gpu=29 world=24"
export G HB_WORLD=24
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT LOCAL_RANK GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE
unset $(compgen -e | grep -E '^PET_') NPROC_PER_NODE WORLD_SIZE_WITH_STANDBY TORCHELASTIC_RUN_ID
export PATH="$ENV/bin:$PATH" PYTHONPATH="$PSI/src:$H:$G/hb_psi0" PSI_SRC="$PSI" HB_AUG_DIR="$G/hb_groot"
export TORCH_HOME="$G/torch-home" HF_HOME="$G/hf-home" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_CACHE="/tmp/hb_act_hfds_$(id -u)"
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled TQDM_MININTERVAL=30
export NCCL_SOCKET_IFNAME=eth GLOO_SOCKET_IFNAME=eth0 NCCL_DEBUG=WARN CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
cd "$PSI"
for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
if [ "$NODE" = 0 ] && [ ! -s "$DEC" ]; then
  "$PY" - "$DEC" <<'PY'
import json, sys
import hb_args as HA
r = HA.receipt("b7")
d = HA.plan(r["frames"], 29)
assert d["global_batch"] == 696, d
d.update(task="b7", robot=r.get("robot"), state_dim=r["state_dim"], action_dim=r["action_dim"], image_keys=r["image_keys"],
         num_nodes=3, note="3 nodes x 8 GPUs x 29 = 696 (user 2026-11-11)")
json.dump(d, open(sys.argv[1], "w"), indent=2)
print("DECISION", d)
PY
fi
for i in $(seq 120); do [ -s "$DEC" ] && break; sleep 5; done
mapfile -t ARGS < <("$PY" "$H/hb_act_args.py" args b7 --decision "$DEC" --out "$ROOT/train" --timestamp b696)
[ "${#ARGS[@]}" -gt 50 ] || { echo "hb_act_args failed"; exit 5; }
[ "$NODE" != 0 ] || printf '%s\n' "${ARGS[@]}" > "$ROOT/train_args.txt"
export HB_AUG_LOG="$ROOT/logs/aug-node$NODE"
"$ENV/bin/torchrun" --nnodes 3 --node_rank "$NODE" --nproc_per_node 8 --master_addr "$MASTER" --master_port 30480 \
  "$H/hb_act_train.py" "${ARGS[@]}"
rc=$?
echo "$rc" > "$ROOT/logs/prod-node-$NODE.exit"
[ "$NODE" = 0 ] && [ "$rc" = 0 ] && cp "$DEC" "$ROOT/production-complete.json"
echo "EXIT $rc"
exit $rc
