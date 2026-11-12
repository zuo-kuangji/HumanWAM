#!/bin/bash
# Relay cmd item (exec'd by stage256_wait.sh). Node layout chosen for the shortest idle time (user 2026-11-12):
#   nodes 0+1: pi0.5 A6 door-pull, 2 x 8 x 16 = 256        (~4.3 h)
#   node 2   : GR00T C6, 8 x 32 = 256                       (~3.9 h)
#   node 3   : pi0.5 A7 re-train (b256), 8 x 32 = 256       (~3.3 h)
# Afterwards (separate 4-node relay items): pi0.5 A3 b256, pi0.5 C6, Psi-Zero A6, Psi-Zero C6.
G=$(dirname "$(dirname "$(realpath "$0")")")
NODE=${RANK:?DLC RANK missing}
MA=${MASTER_ADDR:?DLC MASTER_ADDR missing}
# 2026-11-12 05:30: nodes 2/3 may have started their job early (fill_early.sh); then only wait for it here.
F=$G/b256-20261112
if [ "$NODE" = 2 ] || [ "$NODE" = 3 ]; then
  if [ -f "$F/fill-node-$NODE.running" ] || [ -f "$F/fill-node-$NODE.rc" ]; then
    until [ -f "$F/fill-node-$NODE.rc" ]; do sleep 60; done
    [ "$(cat "$F/fill-node-$NODE.rc")" = 0 ] && { echo "early fill on node $NODE finished OK"; exit 0; }
    echo "early fill on node $NODE failed (rc $(cat "$F/fill-node-$NODE.rc")), running it now"
  fi
fi
case $NODE in
  0|1) bash "$G/hb_b256/pi05_multi.sh" a6 2 "$NODE" "$MA" 30710 16 ;;
  2) for i in $(seq 720); do [ -s "$G/groot-data/c6/hb-dataset-receipt.json" ] && break; sleep 15; done
     bash "$G/hb_groot/groot_single.sh" c6 32 "$G/groot-runs-b256" ;;
  3) bash "$G/hb_b256/pi05_multi.sh" a7 1 0 localhost 0 32 ;;
esac
exit 0
