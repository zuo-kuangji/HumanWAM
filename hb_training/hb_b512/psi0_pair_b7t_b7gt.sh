#!/bin/bash
# Relay cmd item (2026-11-22 user): Psi-Zero B7 black trimmed (ranks 0,1; @949da46 first 100 of 107) + B7 green trimmed v2
# (ranks 2,3; @22e3b83 all 421), each 2 x 8 x 32 = 512, 4000 steps, G1 default recipe (no proprio delay / frame drop).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/gp-20261122
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; M0=${MASTER_ADDR:?DLC MASTER_ADDR missing}; T0=$(date +%s); MF=$G/psi0-runs/.b7pair-master-addr
case "$RK" in
  0|1) wait_ok $L/psi0-b7t.ok $L/psi0-b7t.failed || { echo "SKIP psi0 b7t: data view not verified"; exit 3; }
       exec bash "$G/hb_b512/psi0_multi.sh" b7t 2 "$RK" "$M0" 31960 32 ;;
  2) wait_ok $L/psi0-b7gt.ok $L/psi0-b7gt.failed || { echo "SKIP psi0 b7gt: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" b7gt 2 0 "$IP" 31970 32 ;;
  3) wait_ok $L/psi0-b7gt.ok $L/psi0-b7gt.failed || { echo "SKIP psi0 b7gt: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" b7gt 2 1 "$(cut -d' ' -f1 "$MF")" 31970 32 ;;
esac
