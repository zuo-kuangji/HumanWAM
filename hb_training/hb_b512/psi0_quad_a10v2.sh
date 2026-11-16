#!/bin/bash
# Relay cmd item (2026-11-13): Psi-Zero A10 v2 (150 episodes) on all 4 nodes, 4 x 8 x 16 = 512, 4000 steps.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi.sh" a10v2 4 "$RANK" "$MASTER_ADDR" 30770 16
