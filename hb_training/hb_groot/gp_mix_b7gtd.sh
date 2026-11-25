#!/bin/bash
# Relay cmd item (2026-11-24 user): B7 green v2, lead-in still frames dropped (HF b7-green-clothes-hamper @4d2c613c, all 421). Each model from its base, 4000 steps, no accumulation.
#   ranks 0,1 -> GR00T 2 x 8 x 32 = 512, ckpt 2000..4000 every 500 -> HF groot/<B7 dir>/b512-leaddrop/
#   ranks 2,3 -> Psi-Zero 2 x 8 x 32 = 512, G1 default recipe (no delay), ckpt 2000/4000 -> HF psi0/<B7 dir>/b512-leaddrop/
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/b7d-20261124; T=b7gtd
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.${T}mix-master-addr
wait_ok $L/cache-$T.ok $L/cache-$T.failed || { echo "SKIP $T: in-loop MSE cache missing"; exit 3; }
case "$RK" in
  0|1) wait_ok $L/groot-$T.ok $L/groot-$T.failed || { echo "SKIP groot $T: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" $T none 29788 29790 ;;
  2) wait_ok $L/psi0-$T.ok $L/psi0-$T.failed || { echo "SKIP psi0 $T: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" $T 2 0 "$IP" 32050 32 ;;
  3) wait_ok $L/psi0-$T.ok $L/psi0-$T.failed || { echo "SKIP psi0 $T: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" $T 2 1 "$(cut -d' ' -f1 "$MF")" 32050 32 ;;
esac
