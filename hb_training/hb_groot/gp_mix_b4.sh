#!/bin/bash
# Relay cmd item (2026-11-25 user): B4 cushion bed -> sofa (HF b4-cushion @194e7cde, 302 ep), each model from its base, no accumulation.
#   ranks 0,1 -> GR00T 2 x 8 x 32 = 512, full fine-tune, 4000 steps, ckpt 2000..4000 every 500 -> HF groot/b4-cushion/b512/
#   ranks 2,3 -> Psi-Zero 2 x 8 x 32 = 512, G1 default recipe (no delay), 4000 steps, ckpt 2000/4000 -> HF psi0/b4-cushion/b512/
#   rank 0, after GR00T (Psi-Zero runs ~2 h longer) -> FLASH B4 on GPU 0, the A800 FLASH recipe (12000 steps, batch 512,
#   save every 4000, 40 workers, seed 42, DLC RANK/WORLD_SIZE unset) -> HF flash/b4-cushion/b512/ by the master-0 publish watcher.
# All halves wait for the in-loop MSE cache (cache-b4.ok) and their data view.
G=$(dirname "$(dirname "$(realpath "$0")")"); P=$(dirname "$G"); L=$G/b4-20261125
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.b4mix-master-addr
wait_ok $L/cache-b4.ok $L/cache-b4.failed || { echo "SKIP b4: in-loop MSE cache not built"; exit 3; }
case "$RK" in
  0) wait_ok $L/groot-b4.ok $L/groot-b4.failed || { echo "SKIP groot b4: data view not verified"; exit 3; }
     bash "$G/hb_groot/groot_pair2x32.sh" b4 none 29806 29808; rc=$?; echo "$(date -u) GR00T b4 rc=$rc"
     OUT=$P/flash-runs/b4-b512-s12000
     if [ -f $OUT/checkpoints/step-012000/protocol.json ]; then echo "FLASH b4 already complete"
     elif wait_ok $L/flash-b4.ok $L/flash-b4.failed; then
       for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done
       mkdir -p $OUT/logs; touch $OUT/.r2; echo "$(date -u) FLASH b4 on GPU 0 -> $OUT"
       ( cd $P/Humanoid_benchmark-bai && env -u RANK -u WORLD_SIZE -u LOCAL_RANK -u LOCAL_WORLD_SIZE -u GROUP_RANK -u MASTER_ADDR -u MASTER_PORT CUDA_VISIBLE_DEVICES=0 \
         $G/envs/hb-baselines/bin/python -I evaluation/baselines/train.py --policy flash --contract g1_sonic_dfx6 --dataset $P/b4-cushion-194e7cd/data \
         --out $OUT --steps 12000 --batch-size 512 --workers 40 --save-every 4000 --log-every 50 --cache-dir $G/flash-cache/b4 \
         --inloop-task b4 --inloop-code $G/eval-action-mse >> $OUT/logs/train-a800.log 2>&1 ); echo "$(date -u) FLASH b4 exit $?"
     else echo "SKIP FLASH b4: preflight failed"; fi
     exit $rc ;;
  1) wait_ok $L/groot-b4.ok $L/groot-b4.failed || { echo "SKIP groot b4: data view not verified"; exit 3; }
     exec bash "$G/hb_groot/groot_pair2x32.sh" b4 none 29806 29808 ;;
  2) wait_ok $L/psi0-b4.ok $L/psi0-b4.failed || { echo "SKIP psi0 b4: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" b4 2 0 "$IP" 32070 32 ;;
  3) wait_ok $L/psi0-b4.ok $L/psi0-b4.failed || { echo "SKIP psi0 b4: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" b4 2 1 "$(cut -d' ' -f1 "$MF")" 32070 32 ;;
esac
