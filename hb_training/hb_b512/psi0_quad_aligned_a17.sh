#!/bin/bash
# Relay cmd item (2026-11-12, aligned re-train, always at the queue tail): Psi-Zero a17 on all 4 nodes, 4 x 8 x 16 = 512,
# 20 epochs, GR00T-A20-aligned augmentation; output psi0-runs/<task>/aligned, HF psi0/<task>/b512/.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi_aligned.sh" a17 4 "$RANK" "$MASTER_ADDR" 30736 16
