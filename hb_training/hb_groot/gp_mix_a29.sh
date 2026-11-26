#!/bin/bash
# Relay cmd item (2026-11-25 user): A29 drawer close, black-marble kitchen 30 Hz (HF a29-drawer-close-black-marble-30hz-v1 @d03c19e6,
# 100 ep, ego_view + chest), each model from its base, no accumulation.
#   ranks 0,1 -> GR00T 2 x 8 x 32 = 512, 4000 steps, ckpt 2000..4000 every 500 -> HF groot/a29-drawer-close/b512/
#   ranks 2,3 -> Psi-Zero 2 x 8 x 32 = 512, G1 default recipe (no delay), 4000 steps -> HF psi0/a29-drawer-close/b512/
#   Psi-Zero runs ~2 h longer than GR00T, so the GR00T nodes then train the single-GPU baselines (user 11-25: place ACT where GPUs idle):
#     rank 0 GPU 0 -> FLASH A29 (A800 FLASH recipe: 12000 steps, batch 512, save every 4000, 40 workers, seed 42)
#     rank 1 GPU 0 -> ACT A29 (hb_baselines ACT b512 recipe as ACT B1 v2 / B4: 4000 steps, warmup 200, seed 0, 32 workers,
#     save every 2000, seeded-aug code Humanoid_benchmark bai @2c91a514); HF upload of ACT by the hb_baselines staging.
#   (ACT B4 was trained on the 4090 side, HF act/b4-cushion/b512 @cff510cf.)
G=$(dirname "$(dirname "$(realpath "$0")")"); P=$(dirname "$G"); L=$G/a29-20261125
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
gpus_free() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done; }
NOENV="env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u MASTER_ADDR -u MASTER_PORT"
act() {  # act <task> <dataset> <episodes a:b> <gpu> <cache marker>
  local T=$1 OUT=$P/act-runs/$1-b512-s4000 W=$P/Humanoid_benchmark-act-2c91a51
  [ -f $OUT/checkpoints/step-004000/protocol.json ] && { echo "ACT $T already complete"; return 0; }
  [ -f "$5" ] || { echo "SKIP ACT $T: in-loop MSE cache missing"; return 3; }
  mkdir -p $OUT; git -C $W rev-parse HEAD > $OUT/code_commit.txt; echo "$(date -u) ACT $T on GPU $4 -> $OUT"
  ( cd $W && $NOENV CUDA_VISIBLE_DEVICES=$4 TORCH_HOME=$G/torch-home $G/envs/hb-baselines/bin/python -I evaluation/baselines/train.py \
      --policy act --contract g1_sonic_dfx6 --dataset $2 --episodes $3 --out $OUT --steps 4000 --batch-size 512 --warmup 200 --seed 0 \
      --workers 32 --save-every 2000 --log-every 50 --cache-dir /dev/shm/hb_act_cache_$T --inloop-task $T --inloop-code $G/eval-action-mse \
      > $OUT/train-a800.log 2>&1 ); local rc=$?
  echo "$(date -u) ACT $T exit $rc"; [ $rc = 0 ] && touch $OUT/A800_DONE; return $rc
}
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.a29mix-master-addr
wait_ok $L/cache-a29.ok $L/cache-a29.failed || { echo "SKIP a29: in-loop MSE cache not built"; exit 3; }
case "$RK" in
  0) wait_ok $L/groot-a29.ok $L/groot-a29.failed || { echo "SKIP groot a29: data view not verified"; exit 3; }
     bash "$G/hb_groot/groot_pair2x32.sh" a29 none 29810 29812; rc=$?; echo "$(date -u) GR00T a29 rc=$rc"
     OUT=$P/flash-runs/a29-b512-s12000
     if [ -f $OUT/checkpoints/step-012000/protocol.json ]; then echo "FLASH a29 already complete"
     elif wait_ok $L/flash-a29.ok $L/flash-a29.failed; then
       gpus_free; mkdir -p $OUT/logs; touch $OUT/.r2; echo "$(date -u) FLASH a29 on GPU 0 -> $OUT"
       ( cd $P/Humanoid_benchmark-bai && $NOENV CUDA_VISIBLE_DEVICES=0 $G/envs/hb-baselines/bin/python -I evaluation/baselines/train.py \
           --policy flash --contract g1_sonic_dfx6 --dataset $P/a29-drawer-close-black-marble-30hz-d03c19e/data --out $OUT --steps 12000 \
           --batch-size 512 --workers 40 --save-every 4000 --log-every 50 --cache-dir $G/flash-cache/a29 \
           --inloop-task a29 --inloop-code $G/eval-action-mse >> $OUT/logs/train-a800.log 2>&1 ); echo "$(date -u) FLASH a29 exit $?"
     else echo "SKIP FLASH a29: preflight failed"; fi
     exit $rc ;;
  1) wait_ok $L/groot-a29.ok $L/groot-a29.failed || { echo "SKIP groot a29: data view not verified"; exit 3; }
     bash "$G/hb_groot/groot_pair2x32.sh" a29 none 29810 29812; rc=$?; echo "$(date -u) GR00T a29 rc=$rc"
     gpus_free
     act a29 $P/a29-drawer-close-black-marble-30hz-d03c19e/data 0:100 0 $L/cache-a29.ok
     exit $rc ;;
  2) wait_ok $L/psi0-a29.ok $L/psi0-a29.failed || { echo "SKIP psi0 a29: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" a29 2 0 "$IP" 32080 32 ;;
  3) wait_ok $L/psi0-a29.ok $L/psi0-a29.failed || { echo "SKIP psi0 a29: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" a29 2 1 "$(cut -d' ' -f1 "$MF")" 32080 32 ;;
esac
