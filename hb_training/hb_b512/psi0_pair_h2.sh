#!/bin/bash
# Relay cmd item (2026-11-14 H2 training guides, a18 4.2 and a14 4.2): Psi0 on the a17 epoch-020 configuration.
#   nodes 0,1: a18c2 (head + chest, action75)   nodes 2,3: a14v2 (head + chest, action82 = hb75 + planner 7)
# each 2 nodes x 8 GPUs x 32 = 512, no accumulation, 4000 steps, warmup 100, checkpoints 2000 / 4000, from the Psi0
# initialization postpre.sonic1.1.unifolm.2609181726.40k; hb_delay_prob 0.15 as in a17 epoch-020 (H2-guide exception
# to CLAUDE.md section 3, which removed the delay for new runs). Skips (exit 3) unless both data views were verified.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/h2psi-20261114/psi0.ok" ] || { echo "SKIP psi0 h2 pair: data views not verified"; exit 3; }
export HB_DELAY_PROB=0.15
RK=${RANK:?DLC RANK missing}; M0=${MASTER_ADDR:?DLC MASTER_ADDR missing}; T0=$(date +%s)
MF=$G/psi0-runs/.h2pair-master-addr
case "$RK" in
  0|1) exec bash "$G/hb_b512/psi0_multi.sh" a18c2 2 "$RK" "$M0" 30850 32 ;;
  2) IP=$(hostname -I | awk '{print $1}'); echo "$IP $(date +%s)" > "$MF"
     exec bash "$G/hb_b512/psi0_multi.sh" a14v2 2 0 "$IP" 30860 32 ;;
  3) for i in $(seq 180); do [ -s "$MF" ] && [ "$(cut -d' ' -f2 "$MF")" -ge $((T0 - 300)) ] && break; sleep 5; done
     exec bash "$G/hb_b512/psi0_multi.sh" a14v2 2 1 "$(cut -d' ' -f1 "$MF")" 30860 32 ;;
esac
