#!/bin/bash
# Relay cmd item (2026-11-22 user): Psi-Zero on the cropped v2 H2 data, head + chest, action75 / state78:
#   ranks 0,1 a18cr2 (HF a18-pour-water-expert-cropped-v2 @44cdd57)   ranks 2,3 a17cr2 (HF a17-unscrew-bottle-expert-cropped-v2 @3520b70)
# each 2 x 8 x 32 = 512, 4000 steps; default Psi-Zero recipe, no proprio delay (user 2026-11-22: align H2 with the other tasks).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/gp-20261122
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB  # 2026-11-22 user: H2 aligned with every other task (CLAUDE.md v2, no proprio delay)
RK=${RANK:?DLC RANK missing}; M0=${MASTER_ADDR:?DLC MASTER_ADDR missing}; T0=$(date +%s); MF=$G/psi0-runs/.h2cr2pair-master-addr
case "$RK" in
  0|1) wait_ok $L/psi0-a18cr2.ok $L/psi0-a18cr2.failed || { echo "SKIP psi0 a18cr2: data view not verified"; exit 3; }
       exec bash "$G/hb_b512/psi0_multi.sh" a18cr2 2 "$RK" "$M0" 31980 32 ;;
  2) wait_ok $L/psi0-a17cr2.ok $L/psi0-a17cr2.failed || { echo "SKIP psi0 a17cr2: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" a17cr2 2 0 "$IP" 31990 32 ;;
  3) wait_ok $L/psi0-a17cr2.ok $L/psi0-a17cr2.failed || { echo "SKIP psi0 a17cr2: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" a17cr2 2 1 "$(cut -d' ' -f1 "$MF")" 31990 32 ;;
esac
