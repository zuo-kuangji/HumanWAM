#!/bin/bash
# hb_relay cmd item (2026-11-11, user: strict model priority): B7 stage 1 = pi0.5 B7 on nodes 0-2 (3 x 8 x 29 = 696) in
# parallel with GR00T B7 on node 3 (8 x 87 = 696). This wrapper only waits (on every node) until the stage script is
# ready ($G/b7-20261111/S1_READY), then execs it. S1_SKIP (or 4 h without READY) skips the item.
G=$(dirname "$(dirname "$(realpath "$0")")")
F=$G/b7-20261111
for i in $(seq 960); do
  [ -f "$F/S1_SKIP" ] && { echo "S1 skipped"; exit 0; }
  [ -f "$F/S1_READY" ] && exec bash "$G/hb_b7/b7_s1.sh"
  sleep 15
done
echo "S1 not ready after 4 h: skipped"; exit 3
