#!/bin/bash
# Relay cmd item (2026-11-15 16:2x user: front of the waiting queue): GR00T A3 fridge open v3 (nodes 0,1, all 125
# episodes) and A7 cabinet close v3 (nodes 2,3, all 100 episodes) in parallel, each 2 x 8 x 32 = 512, no accumulation,
# 4000 steps, full fine-tune (groot_pair2x32.sh). Hand action label = measured hands (user 2026-11-15), as every G1 task.
# Waits up to 90 min for the data views; skips (exit 3) unless both were verified.
G=$(dirname "$(dirname "$(realpath "$0")")")
L=$G/a37v3-20261115
for i in $(seq 540); do { [ -f "$L/groot.ok" ] || [ -f "$L/prep.done" ]; } && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $L/groot.ok"; sleep 10; done
[ -f "$L/groot.ok" ] || { echo "SKIP groot a3v3/a7v3: data views not verified"; exit 3; }
exec bash "$G/hb_groot/groot_pair2x32.sh" a3v3 a7v3 29740 29742
