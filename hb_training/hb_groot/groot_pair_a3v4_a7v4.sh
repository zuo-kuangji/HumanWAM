#!/bin/bash
# Relay cmd item (2026-11-15 18:4x user: A3 / A7 v3 replaced by v4): GR00T A3 fridge open v4 (ranks 0,1, HF
# a3-fridge-open-expert-v4 @6393d4a, all 125 episodes) and A7 cabinet close v4 (ranks 2,3, @90cc2b5, all 100 episodes)
# in parallel, each 2 x 8 x 32 = 512, no accumulation, 4000 steps, full fine-tune from the GR00T N1.7 base
# (groot_pair2x32.sh). Action 76 = SONIC token 64 + measured hands 6 + 6 (the v4 layout = every G1 task).
# Each pair waits (up to 2 h) until its data view is verified; skips (exit 3) when the preparation failed.
G=$(dirname "$(dirname "$(realpath "$0")")")
L=$G/a37v4-20261115
wait_ok() {
  for i in $(seq 720); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done
  return 1
}
case "${RANK:?DLC RANK missing}" in
  0|1) wait_ok "$L/groot-a3v4.ok" "$L/groot-a3v4.failed" || { echo "SKIP groot a3v4: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" a3v4 none 29740 29742 ;;
  2|3) wait_ok "$L/groot-a7v4.ok" "$L/groot-a7v4.failed" || { echo "SKIP groot a7v4: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" none a7v4 29740 29742 ;;
esac
