#!/bin/bash
# Relay cmd item (2026-11-22 user: H2 aligned with every other task): Psi-Zero A14 cropped-v3 (HF @79fe08a) again on the default
# recipe, i.e. WITHOUT the hb_delay_prob 0.15 of the 2026-11-20 run (task key a14cr3a = same data); head + chest, action82,
# 4 nodes x 8 x 16 = 512, no accumulation, 4000 steps.
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/gp-20261122
for i in $(seq 2160); do [ -f $L/psi0-a14cr3a.ok ] && break; [ -f $L/psi0-a14cr3a.failed ] && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for psi0-a14cr3a.ok"; sleep 10; done
[ -f $L/psi0-a14cr3a.ok ] || { echo "SKIP psi0 a14cr3a: data view not verified"; exit 3; }
unset HB_DELAY_PROB
exec bash "$G/hb_b512/psi0_multi.sh" a14cr3a 4 "$RANK" "$MASTER_ADDR" 31995 16
