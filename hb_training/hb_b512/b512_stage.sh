#!/bin/bash
# Relay cmd item (2026-11-11 user: batch 512 for every not-yet-started GR00T / pi0.5 / Psi-Zero run; Psi-Zero two tasks in
# parallel on 2 nodes each). Pair A = nodes 0+1 (master-0 address): pi0.5 B7 -> Psi-Zero basket -> Psi-Zero B7.
# Pair B = nodes 2+3 (node 2 address): Psi-Zero a17. Node 3 enters this item only after ACT B7 (item 30) ends, so node 2
# waits for it before starting (rendezvous would time out after 15 min otherwise).
G=$(dirname "$(dirname "$(realpath "$0")")")
F=$G/b512-20261111
NODE=${RANK:?DLC RANK missing}
MA=${MASTER_ADDR:?DLC MASTER_ADDR missing}
B=$G/hb_b512
case $NODE in
  0|1)
    bash $B/pi05_pair.sh b7 $NODE "$MA" 30510; echo "pi05 b7 rc=$?"
    bash $B/psi0_pair.sh basket $NODE "$MA" 30520; echo "psi0 basket rc=$?"
    bash $B/psi0_pair.sh b7 $NODE "$MA" 30530; echo "psi0 b7 rc=$?" ;;
  2)
    hostname -i | awk '{print $1}' > $F/pairB_master
    until [ -f $F/pairB_node3_ready ]; do sleep 15; done
    bash $B/psi0_pair.sh a17 0 "$(cat $F/pairB_master)" 30540; echo "psi0 a17 rc=$?" ;;
  3)
    touch $F/pairB_node3_ready
    until [ -s $F/pairB_master ]; do sleep 5; done
    bash $B/psi0_pair.sh a17 1 "$(cat $F/pairB_master)" 30540; echo "psi0 a17 rc=$?" ;;
esac
exit 0
