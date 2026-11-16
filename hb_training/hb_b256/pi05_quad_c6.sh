#!/bin/bash
# 2026-11-13 user: long tasks (C-class cross-scene C6, and A14) train 2x the basket step budget of this model.
export HB_FIXED_STEPS=8000
# Relay cmd item (2026-11-12): pi0.5 c6 on all 4 nodes, 4 x 8 x 8 = 256, 20 epochs, GR00T-A20-aligned augmentation.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b256/pi05_multi.sh" c6 4 "$RANK" "$MASTER_ADDR" 30720 8
