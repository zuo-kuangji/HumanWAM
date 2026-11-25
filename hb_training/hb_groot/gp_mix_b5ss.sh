#!/bin/bash
# Relay cmd item (2026-11-22 user): B5 standing start (HF b5-chair-push-standing-start-expert-v1 @913f5a1, all 100)
#   ranks 0,1 -> GR00T, 2 x 8 x 32 = 512, full fine-tune, 4000 steps (groot_pair2x32.sh)
#   ranks 2,3 -> Psi-Zero, 2 x 8 x 32 = 512, 4000 steps, G1 default recipe (no proprio delay / frame drop)
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/gp-20261122
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.b5ssmix-master-addr
case "$RK" in
  0|1) wait_ok $L/groot-b5ss.ok $L/groot-b5ss.failed || { echo "SKIP groot b5ss: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" b5ss none 29764 29766 ;;
  2) wait_ok $L/psi0-b5ss.ok $L/psi0-b5ss.failed || { echo "SKIP psi0 b5ss: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" b5ss 2 0 "$IP" 31950 32 ;;
  3) wait_ok $L/psi0-b5ss.ok $L/psi0-b5ss.failed || { echo "SKIP psi0 b5ss: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" b5ss 2 1 "$(cut -d' ' -f1 "$MF")" 31950 32 ;;
esac
