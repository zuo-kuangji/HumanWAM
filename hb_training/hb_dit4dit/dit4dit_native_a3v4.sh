#!/bin/bash
# hb_relay cmd item (2026-11-20 user: every task on its latest data; FastWAM > Delta Flow > pi0.5 > DiT4DiT): DiT4DiT a3v4, native state encoding (G1 sin/cos 73), global batch 64 = 32 x 2, no accumulation, 8000 steps (hb_d4d_config default).
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
[ -f "$G/dit4dit-data/a3v4/READY" ] || { echo "SKIP dit4dit a3v4: data not prepared"; exit 3; }
exec bash "$H/dit4dit_native_relay.sh" a3v4
