#!/bin/bash
# Relay cmd item (2026-11-20 user: every task on its latest data; FastWAM > Delta Flow > pi0.5 > DiT4DiT): pi0.5 a10v3 on all 4 nodes, 4 x 8 x 8 = 256, 4000 steps (HB_FIXED_STEPS default), GR00T-A20-aligned augmentation v2, no proprio delay.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/pi05-data/a10v3/READY" ] || { echo "SKIP pi05 a10v3: data not prepared"; exit 3; }
exec bash "$G/hb_b256/pi05_multi.sh" a10v3 4 "$RANK" "$MASTER_ADDR" 31520 8
