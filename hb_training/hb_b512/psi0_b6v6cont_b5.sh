#!/bin/bash
# Relay cmd item (2026-11-15 17:xx user: B6 v6 continuations first, then B5, then A3 / A7, per model):
#   ranks 0,1 -> Psi-Zero B6 v6 continuation (psi0_cont_b6v6.sh, 2 x 8 x 32 = 512, 4000 steps, 0.3 x peak lr)
#   ranks 2,3 -> Psi-Zero B5 chair push v1 (HF b5-chair-push-expert-v1 @b85b7e9, all 100 episodes): 2 x 8 x 32 = 512,
#                4000 steps, G1 default recipe (CLAUDE.md v2: no proprio delay / frame drop)
# Each pair waits (up to 2 h) until its data view is verified; skips (exit 3) when the preparation failed.
G=$(dirname "$(dirname "$(realpath "$0")")")
wait_ok() {
  for i in $(seq 720); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done
  return 1
}
unset HB_DELAY_PROB
RK=${RANK:?DLC RANK missing}; T0=$(date +%s)
MF=$G/psi0-runs/.b5pair-master-addr
case "$RK" in
  0|1) wait_ok "$G/b6v6-20261115/psi0-b6v6.ok" "$G/b6v6-20261115/psi0-b6v6.failed" || { echo "SKIP psi0 b6v6: data view not verified"; exit 3; }
       exec bash "$G/hb_b512/psi0_cont_b6v6.sh" ;;
  2) wait_ok "$G/b6v6-20261115/psi0-b5.ok" "$G/b6v6-20261115/psi0-b5.failed" || { echo "SKIP psi0 b5: data view not verified"; exit 3; }
     IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" b5 2 0 "$IP" 30910 32 ;;
  3) wait_ok "$G/b6v6-20261115/psi0-b5.ok" "$G/b6v6-20261115/psi0-b5.failed" || { echo "SKIP psi0 b5: data view not verified"; exit 3; }
     for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" b5 2 1 "$(cut -d' ' -f1 "$MF")" 30910 32 ;;
esac
