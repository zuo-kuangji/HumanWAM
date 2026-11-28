#!/bin/bash
# Relay cmd item (2026-11-28 user: steps scaled by episode length vs A20 basket v2, ceil to 1000; ckpt ~80 % + last).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/newtrim-20261127
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; sleep 10; done; return 1; }
wait_ok $L/cache-a14cr5.ok $L/cache-a14cr5.failed && wait_ok $L/psi0-a14cr5.ok $L/psi0-a14cr5.failed || { echo "SKIP psi0 a14cr5"; exit 3; }
# Psi-Zero A14 cropped-v5 (h2p), 4 x 8 x 16 = 512 (registered A14 layout exception), no proprio delay: 11000 steps, ckpt 8800 + 11000
unset HB_DELAY_PROB; export HB_FIXED_STEPS=11000 HB_CKPT_SAVE=8800
exec bash $G/hb_b512/psi0_multi.sh a14cr5 4 "${RANK:?}" "${MASTER_ADDR:?}" 32132 16
