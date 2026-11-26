#!/bin/bash
# Relay cmd item (2026-11-26 user): A21 floor basket transfer (HF a21-floor-transfer-basket-trim-raw-head-expert-v1 @682f2bfc,
# 100 ep, G1, ego_view + chest, idle-trimmed), each model from its base, no accumulation. Same layout as gp_mix_a29.sh:
#   ranks 0,1 -> GR00T 2 x 8 x 32 = 512, 4000 steps, ckpt 2000..4000 every 500 -> HF groot/a21-floor-basket-transfer/b512/
#   ranks 2,3 -> Psi-Zero 2 x 8 x 32 = 512, G1 default recipe (no delay), 4000 steps -> HF psi0/a21-floor-basket-transfer/b512/
#   Psi-Zero runs ~2 h longer than GR00T, so the GR00T nodes then train the single-GPU baselines:
#     rank 0 GPU 0 -> FLASH A21 (A800 FLASH recipe: 12000 steps, batch 512, save every 4000, 40 workers, seed 42)
#     rank 1 GPU 0 -> ACT A21 (hb_baselines ACT b512 recipe: 4000 steps, warmup 200, seed 0, 32 workers, save every 2000,
#     seeded-aug code Humanoid_benchmark bai @2c91a514); A800_DONE marker -> HF upload by the hb_baselines staging session.
G=$(dirname "$(dirname "$(realpath "$0")")"); P=$(dirname "$G"); L=$G/a21-20261126; DS=$P/a21-floor-transfer-basket-682f2bf/data
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
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.a21mix-master-addr
wait_ok $L/cache-a21.ok $L/cache-a21.failed || { echo "SKIP a21: in-loop MSE cache not built"; exit 3; }
case "$RK" in
  0) wait_ok $L/groot-a21.ok $L/groot-a21.failed || { echo "SKIP groot a21: data view not verified"; exit 3; }
     bash "$G/hb_groot/groot_pair2x32.sh" a21 none 29814 29816; rc=$?; echo "$(date -u) GR00T a21 rc=$rc"
     OUT=$P/flash-runs/a21-b512-s12000
     if [ -f $OUT/checkpoints/step-012000/protocol.json ]; then echo "FLASH a21 already complete"
     elif wait_ok $L/flash-a21.ok $L/flash-a21.failed; then
       gpus_free; mkdir -p $OUT/logs; touch $OUT/.r2; echo "$(date -u) FLASH a21 on GPU 0 -> $OUT"
       ( cd $P/Humanoid_benchmark-bai && $NOENV CUDA_VISIBLE_DEVICES=0 $G/envs/hb-baselines/bin/python -I evaluation/baselines/train.py \
           --policy flash --contract g1_sonic_dfx6 --dataset $DS --out $OUT --steps 12000 \
           --batch-size 512 --workers 40 --save-every 4000 --log-every 50 --cache-dir $G/flash-cache/a21 \
           --inloop-task a21 --inloop-code $G/eval-action-mse >> $OUT/logs/train-a800.log 2>&1 ); echo "$(date -u) FLASH a21 exit $?"
     else echo "SKIP FLASH a21: preflight failed"; fi
     exit $rc ;;
  1) wait_ok $L/groot-a21.ok $L/groot-a21.failed || { echo "SKIP groot a21: data view not verified"; exit 3; }
     bash "$G/hb_groot/groot_pair2x32.sh" a21 none 29814 29816; rc=$?; echo "$(date -u) GR00T a21 rc=$rc"
     gpus_free
     act a21 $DS 0:100 0 $L/cache-a21.ok
     exit $rc ;;
  2) wait_ok $L/psi0-a21.ok $L/psi0-a21.failed || { echo "SKIP psi0 a21: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" a21 2 0 "$IP" 32090 32 ;;
  3) wait_ok $L/psi0-a21.ok $L/psi0-a21.failed || { echo "SKIP psi0 a21: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" a21 2 1 "$(cut -d' ' -f1 "$MF")" 32090 32 ;;
esac
