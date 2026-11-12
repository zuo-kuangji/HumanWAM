#!/bin/bash
# Relay cmd item (2026-11-12): Psi-Zero a6 on all 4 nodes, 4 x 8 x 16 = 512, 20 epochs, GR00T-A20-aligned augmentation.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi.sh" a6 4 "$RANK" "$MASTER_ADDR" 30730 16
