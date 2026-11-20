#!/bin/bash
# Relay cmd item (2026-11-20 user: every task on its latest data; FastWAM > Delta Flow > pi0.5 > DiT4DiT): pi0.5 a7v4 on all 4 nodes, 4 x 8 x 8 = 256, 4000 steps (HB_FIXED_STEPS default), GR00T-A20-aligned augmentation v2, no proprio delay.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/pi05-data/a7v4/READY" ] || { echo "SKIP pi05 a7v4: data not prepared"; exit 3; }
exec bash "$G/hb_b256/pi05_multi.sh" a7v4 4 "$RANK" "$MASTER_ADDR" 31510 8
