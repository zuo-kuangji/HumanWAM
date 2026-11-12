#!/bin/bash
# Relay cmd item (2026-11-12): Psi-Zero <task> on all 4 nodes, global batch 512 = 4 x 8 x 16, 20 epochs.
# Usage (relay): cmd .../psi0_quad.sh  with the task in $G/b256-20261112/psi0_quad_task (default a6)
G=$(dirname "$(dirname "$(realpath "$0")")")
TASK=$(cat "$G/b256-20261112/psi0_quad_task" 2>/dev/null || echo a6)
exec bash "$G/hb_b512/psi0_multi.sh" "$TASK" 4 "${RANK:?}" "${MASTER_ADDR:?}" 30610 16
