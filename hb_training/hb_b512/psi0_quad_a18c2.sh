#!/bin/bash
# Relay cmd item (2026-11-14 18:2x user): Psi-Zero a18 H2 2 cameras (head + chest), 4 x 8 x 16 = 512, 4000 steps, aligned augmentation.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi.sh" a18c2 4 "$RANK" "$MASTER_ADDR" 30820 16
