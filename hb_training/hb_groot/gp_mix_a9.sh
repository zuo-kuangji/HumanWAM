#!/bin/bash
# Relay cmd item (2026-11-24 user): A9 pedal bin lid open (HF a9-pedal-open-expert-v1 @8d432063, 100 ep), each model from its base, 4000 steps, no accumulation.
#   ranks 0,1 -> GR00T 2 x 8 x 32 = 512, full fine-tune, ckpt 2000..4000 every 500 -> HF groot/a9-pedal-open/b512/
#   ranks 2,3 -> Psi-Zero 2 x 8 x 32 = 512, G1 default recipe (no delay), ckpt 2000/4000 -> HF psi0/a9-pedal-open/b512/
# Both halves also wait for the in-loop MSE cache (cache-a9.ok).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/a9-20261124
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.a9mix-master-addr
wait_ok $L/cache-a9.ok $L/cache-a9.failed || { echo "SKIP a9: in-loop MSE cache not built"; exit 3; }
case "$RK" in
  0|1) wait_ok $L/groot-a9.ok $L/groot-a9.failed || { echo "SKIP groot a9: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" a9 none 29780 29782 ;;
  2) wait_ok $L/psi0-a9.ok $L/psi0-a9.failed || { echo "SKIP psi0 a9: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" a9 2 0 "$IP" 32030 32 ;;
  3) wait_ok $L/psi0-a9.ok $L/psi0-a9.failed || { echo "SKIP psi0 a9: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" a9 2 1 "$(cut -d' ' -f1 "$MF")" 32030 32 ;;
esac
