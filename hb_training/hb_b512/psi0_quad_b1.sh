#!/bin/bash
# Relay cmd item (2026-11-14 user): Psi-Zero b1 (all 115 episodes) on all 4 nodes, 4 x 8 x 16 = 512, 4000 steps.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi.sh" b1 4 "$RANK" "$MASTER_ADDR" 30840 16
