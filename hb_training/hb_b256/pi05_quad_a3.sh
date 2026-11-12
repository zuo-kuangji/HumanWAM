#!/bin/bash
# Relay cmd item (2026-11-12): pi0.5 a3 on all 4 nodes, 4 x 8 x 8 = 256, 20 epochs, GR00T-A20-aligned augmentation.
# 07:30: nodes 2+3 may already run it on 2 nodes (fill2_a3.sh, 2 x 8 x 16 = 256); then every node only waits for that run.
G=$(dirname "$(dirname "$(realpath "$0")")")
F=$G/b256-20261112
if [ -f "$F/fill2.running" ] || [ -f "$F/fill2.rc" ] || [ -s "$F/fill2_master" ]; then
  until [ -f "$F/fill2.rc" ]; do sleep 60; done
  [ "$(cat "$F/fill2.rc")" = 0 ] && { echo "pi0.5 A3 finished on nodes 2+3 (fill2)"; exit 0; }
  echo "fill2 A3 failed (rc $(cat "$F/fill2.rc")), re-running on 4 nodes"
  rm -rf "$G/pi05-runs/b256/a3.failed-fill2"; mv "$G/pi05-runs/b256/a3" "$G/pi05-runs/b256/a3.failed-fill2-node$RANK" 2>/dev/null
fi
exec bash "$G/hb_b256/pi05_multi.sh" a3 4 "$RANK" "$MASTER_ADDR" 30720 8
