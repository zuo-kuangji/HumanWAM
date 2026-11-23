#!/bin/bash
# Relay cmd item (2026-11-23 user): basket v2 = A20 final data (HF a20-basket-transfer-pre-reach-trim-expert-v2 @c4df4d40,
# 100 ep), each model from its base (no v1 stage), 4000 steps, no accumulation.
#   ranks 0,1 -> GR00T 2 x 8 x 32 = 512, full fine-tune, ckpt 2000..4000 every 500 -> HF groot/a20-basket-transfer/b512-trimv2-pure/
#   ranks 2,3 -> Psi-Zero 2 x 8 x 32 = 512, G1 default recipe (no delay), ckpt 2000/4000 -> HF psi0/a20-basket-transfer/b512-trimv2-pure/
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/bv2p-20261123
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.basketv2pmix-master-addr
case "$RK" in
  0|1) wait_ok $L/groot-basketv2p.ok $L/groot-basketv2p.failed || { echo "SKIP groot basketv2p: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" basketv2p none 29776 29778 ;;
  2) wait_ok $L/psi0-basketv2p.ok $L/psi0-basketv2p.failed || { echo "SKIP psi0 basketv2p: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" basketv2p 2 0 "$IP" 32020 32 ;;
  3) wait_ok $L/psi0-basketv2p.ok $L/psi0-basketv2p.failed || { echo "SKIP psi0 basketv2p: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" basketv2p 2 1 "$(cut -d' ' -f1 "$MF")" 32020 32 ;;
esac
