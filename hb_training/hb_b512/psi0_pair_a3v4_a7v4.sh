#!/bin/bash
# Relay cmd item (2026-11-15 18:4x user: A3 / A7 v3 replaced by v4): Psi-Zero A3 fridge open v4 (ranks 0,1, all 125
# episodes) and A7 cabinet close v4 (ranks 2,3, all 100 episodes), each 2 nodes x 8 GPUs x 32 = 512, no accumulation,
# 4000 steps, G1 default recipe (CLAUDE.md v2: no proprio delay / frame drop); measured hands (the v4 layout).
# Each pair waits (up to 2 h) until its data view is verified; skips (exit 3) when the preparation failed.
G=$(dirname "$(dirname "$(realpath "$0")")")
L=$G/a37v4-20261115
wait_ok() {
  for i in $(seq 720); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done
  return 1
}
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; M0=${MASTER_ADDR:?DLC MASTER_ADDR missing}; T0=$(date +%s)
MF=$G/psi0-runs/.a37v4pair-master-addr
case "$RK" in
  0|1) wait_ok "$L/psi0-a3v4.ok" "$L/psi0-a3v4.failed" || { echo "SKIP psi0 a3v4: data view not verified"; exit 3; }
       exec bash "$G/hb_b512/psi0_multi.sh" a3v4 2 "$RK" "$M0" 30880 32 ;;
  2) wait_ok "$L/psi0-a7v4.ok" "$L/psi0-a7v4.failed" || { echo "SKIP psi0 a7v4: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" a7v4 2 0 "$IP" 30890 32 ;;
  3) wait_ok "$L/psi0-a7v4.ok" "$L/psi0-a7v4.failed" || { echo "SKIP psi0 a7v4: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" a7v4 2 1 "$(cut -d' ' -f1 "$MF")" 30890 32 ;;
esac
