#!/bin/bash
# 2026-11-27: copy of act_b5ss.sh on GPU 7, started by hb_ode/ode_mlp_pool.sh on rank 0 next to the ODE runs (GPUs 0-4); the queue item act_b5ss then finds it complete
# Relay cmd item (2026-11-26 user: B5 standing start, all models, end of the queue): ACT B5 standing start on rank 0 GPU 0
# (hb_baselines ACT b512 recipe as ACT A21/A29: 4000 steps, warmup 200, seed 0, 32 workers, save every 2000, seeded-aug code
# Humanoid_benchmark bai @2c91a514). A800_DONE marker -> HF act/b5-chair-push-standing-start/b512/ by the hb_baselines staging session.
# GR00T / Psi-Zero / FLASH b5ss were already trained on the same data (HF */b5-chair-push-standing-start/b512) -> not repeated.
G=$(dirname "$(dirname "$(realpath "$0")")"); P=$(dirname "$G"); DS=$P/b5-chair-push-standing-start-v1-913f5a1/data
RK=${RANK:?DLC RANK missing}; [ "$RK" = 0 ] || { echo "act b5ss: rank $RK has nothing to do"; exit 0; }
NOENV="env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u MASTER_ADDR -u MASTER_PORT"
T=b5ss OUT=$P/act-runs/b5ss-b512-s4000 W=$P/Humanoid_benchmark-act-2c91a51
[ -f $OUT/checkpoints/step-004000/protocol.json ] && { echo "ACT $T already complete"; exit 0; }
[ -f $G/eval-action-mse/cache/b5ss/train10/meta.json ] || { echo "SKIP ACT $T: in-loop MSE cache missing"; exit 3; }
for i in $(seq 60); do [ -z "$(nvidia-smi -i 7 --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done
mkdir -p $OUT; git -C $W rev-parse HEAD > $OUT/code_commit.txt; echo "$(date -u) ACT $T on GPU 7 -> $OUT"
( cd $W && $NOENV CUDA_VISIBLE_DEVICES=7 TORCH_HOME=$G/torch-home $G/envs/hb-baselines/bin/python -I evaluation/baselines/train.py \
    --policy act --contract g1_sonic_dfx6 --dataset $DS --episodes 0:100 --out $OUT --steps 4000 --batch-size 512 --warmup 200 --seed 0 \
    --workers 32 --save-every 2000 --log-every 50 --cache-dir /dev/shm/hb_act_cache_$T --inloop-task $T --inloop-code $G/eval-action-mse \
    > $OUT/train-a800.log 2>&1 ); rc=$?
echo "$(date -u) ACT $T exit $rc"; [ $rc = 0 ] && touch $OUT/A800_DONE; exit $rc
