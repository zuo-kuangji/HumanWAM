#!/bin/bash
# Relay cmd item (2026-11-12): pi0.5 batch-256 runs (A6 door-pull, A7/A3 re-trains, plus c6 GR00T/pi0.5 when its 100 episodes
# are ready). The node layout is decided when the item becomes due (c6 may join), so every node waits here for
# $G/b256-20261112/STAGE_READY and then execs hb_b256/stage256.sh (written then). STAGE_SKIP skips the item.
G=$(dirname "$(dirname "$(realpath "$0")")")
F=$G/b256-20261112
for i in $(seq 2880); do
  [ -f "$F/STAGE_SKIP" ] && exit 0
  [ -f "$F/STAGE_READY" ] && exec bash "$G/hb_b256/stage256.sh"
  sleep 15
done
exit 3
