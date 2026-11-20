#!/bin/bash
# Relay cmd item (2026-11-20 user: every task on its latest data; FastWAM > Delta Flow > pi0.5 > DiT4DiT): pi0.5 b6v6 on all 4 nodes, 4 x 8 x 8 = 256, 4000 steps (HB_FIXED_STEPS default), GR00T-A20-aligned augmentation v2, no proprio delay.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/pi05-data/b6v6/READY" ] || { echo "SKIP pi05 b6v6: data not prepared"; exit 3; }
exec bash "$G/hb_b256/pi05_multi.sh" b6v6 4 "$RANK" "$MASTER_ADDR" 31570 8
