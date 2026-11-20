#!/bin/bash
# Relay cmd item (2026-11-20 user: A14 screw-bulb cropped-v3 = newest A14; GR00T, then Psi-Zero, then FastWAM): Psi-Zero A14 cr3,
# head + chest, action82 (h2p = hb75 + planner 7), 4 nodes x 8 x 16 = 512, no accumulation, 4000 steps; hb_delay_prob 0.15 as the
# A14 v2 run (H2-guide exception to CLAUDE.md section 3). Skips (exit 3) unless the data view was CPU-verified.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/a14cr3-20261120/psi0.ok" ] || { echo "SKIP psi0 a14cr3: data view not verified"; exit 3; }
export HB_DELAY_PROB=0.15
exec bash "$G/hb_b512/psi0_multi.sh" a14cr3 4 "$RANK" "$MASTER_ADDR" 31720 16
