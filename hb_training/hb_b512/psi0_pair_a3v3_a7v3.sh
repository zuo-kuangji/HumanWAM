#!/bin/bash
# Relay cmd item (2026-11-15 16:2x user: front of the waiting queue): Psi-Zero A3 fridge open v3 (nodes 0,1, all 125
# episodes) and A7 cabinet close v3 (nodes 2,3, all 100 episodes), each 2 nodes x 8 GPUs x 32 = 512, no accumulation,
# 4000 steps; G1 default recipe (CLAUDE.md v2: no proprio delay / frame drop). Measured hands as every G1 task.
# Waits up to 90 min for the data views; skips (exit 3) unless both were verified.
G=$(dirname "$(dirname "$(realpath "$0")")")
L=$G/a37v3-20261115
for i in $(seq 540); do { [ -f "$L/psi0.ok" ] || [ -f "$L/prep.done" ]; } && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $L/psi0.ok"; sleep 10; done
[ -f "$L/psi0.ok" ] || { echo "SKIP psi0 a3v3/a7v3: data views not verified"; exit 3; }
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; M0=${MASTER_ADDR:?DLC MASTER_ADDR missing}; T0=$(date +%s)
MF=$G/psi0-runs/.a37pair-master-addr
case "$RK" in
  0|1) exec bash "$G/hb_b512/psi0_multi.sh" a3v3 2 "$RK" "$M0" 30880 32 ;;
  2) IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" a7v3 2 0 "$IP" 30890 32 ;;
  3) for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" a7v3 2 1 "$(cut -d' ' -f1 "$MF")" 30890 32 ;;
esac
