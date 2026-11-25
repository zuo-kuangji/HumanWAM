#!/bin/bash
# Relay cmd item (2026-11-25 user): B1 v2 (HF b1-riceball-microwave-expert-v2-30hz @b58f0434, 200 ep), each model from its base, 4000 steps, no accumulation.
#   ranks 0,1 -> GR00T 2 x 8 x 32 = 512, full fine-tune, ckpt 2000..4000 every 500 -> HF groot/b1-riceball-microwave/b512-v2/
#   ranks 2,3 -> Psi-Zero 2 x 8 x 32 = 512, G1 default recipe (no delay), ckpt 2000/4000 -> HF psi0/b1-riceball-microwave/b512-v2/
# Both halves also wait for the in-loop MSE cache (cache-b1v2.ok).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/b1v2-20261125
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.b1v2mix-master-addr
wait_ok $L/cache-b1v2.ok $L/cache-b1v2.failed || { echo "SKIP b1v2: in-loop MSE cache not built"; exit 3; }
case "$RK" in
  0|1) wait_ok $L/groot-b1v2.ok $L/groot-b1v2.failed || { echo "SKIP groot b1v2: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" b1v2 none 29796 29798 ;;
  2) wait_ok $L/psi0-b1v2.ok $L/psi0-b1v2.failed || { echo "SKIP psi0 b1v2: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" b1v2 2 0 "$IP" 32060 32 ;;
  3) wait_ok $L/psi0-b1v2.ok $L/psi0-b1v2.failed || { echo "SKIP psi0 b1v2: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" b1v2 2 1 "$(cut -d' ' -f1 "$MF")" 32060 32 ;;
esac
