#!/bin/bash
# Relay cmd item (2026-11-15 user): Psi-Zero A10 trolley push v3 (all 148 episodes), 4 x 8 x 16 = 512, 4000 steps.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/a10v3-20261115/psi0.ok" ] || { echo "SKIP psi0 a10v3: data view not verified"; exit 3; }
exec bash "$G/hb_b512/psi0_multi.sh" a10v3 4 "$RANK" "$MASTER_ADDR" 30870 16
