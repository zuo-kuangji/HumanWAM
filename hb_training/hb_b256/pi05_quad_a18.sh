#!/bin/bash
# Relay cmd item (2026-11-12, aligned re-train, always at the queue tail): pi0.5 a18 on all 4 nodes, 4 x 8 x 8 = 256, 20 epochs,
# GR00T-A20-aligned augmentation (no proprio delay); HF pi05/<task>/b256/.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b256/pi05_multi.sh" a18 4 "$RANK" "$MASTER_ADDR" 30722 8
