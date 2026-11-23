#!/bin/bash
# Relay cmd item (2026-11-23 user): pure B6 v6 (HF b6-pedal-bin @ad2f5bf0 root v6, ALL 542 episodes - user exception to the
# B6 v6 first-500 rule; FastWAM b6v6 used 0-499), each from its base model (no v5 stage), 4000 steps, no accumulation.
#   ranks 0,1 -> GR00T, 2 x 8 x 32 = 512, full fine-tune, ckpt 2000..4000 every 500 (groot_pair2x32.sh) -> HF groot/b6-pedal-bin/b512-v6-pure/
#   ranks 2,3 -> Psi-Zero, 2 x 8 x 32 = 512, G1 default recipe (no proprio delay / frame drop), ckpt 2000/4000 -> HF psi0/b6-pedal-bin/b512-v6-pure/
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/b6v6p-20261123
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s); MF=$G/psi0-runs/.b6v6pmix-master-addr
case "$RK" in
  0|1) wait_ok $L/groot-b6v6p.ok $L/groot-b6v6p.failed || { echo "SKIP groot b6v6p: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" b6v6p none 29772 29774 ;;
  2) wait_ok $L/psi0-b6v6p.ok $L/psi0-b6v6p.failed || { echo "SKIP psi0 b6v6p: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" b6v6p 2 0 "$IP" 32010 32 ;;
  3) wait_ok $L/psi0-b6v6p.ok $L/psi0-b6v6p.failed || { echo "SKIP psi0 b6v6p: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" b6v6p 2 1 "$(cut -d' ' -f1 "$MF")" 32010 32 ;;
esac
