#!/bin/bash
# Relay cmd item (2026-11-13): Psi-Zero A1 on all 4 nodes, 4 x 8 x 16 = 512, 4000 steps, GR00T-A20-aligned augmentation.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi.sh" a1 4 "$RANK" "$MASTER_ADDR" 30750 16
