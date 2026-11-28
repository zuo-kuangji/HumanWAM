#!/bin/bash
# 2026-11-28 04:4xZ (user: new tasks after the older items): ranks 0,1 run the older Psi-Zero a17cr4 (4000 steps = scaled value,
# ckpt 3200 + 4000, seed 292285, 2 x 8 x 32) OUTSIDE the relay; queue item gate_side01b.sh holds the next 4-node item until it ends.
# Start on master-0 and worker-0 (DLC env RANK / MASTER_ADDR): setsid nohup bash <this> > /dev/null 2>&1 < /dev/null &
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/newtask-20261128
RK=${RANK:?}; [ "$RK" = 0 ] || [ "$RK" = 1 ] || exit 2
exec >> $L/side01b-node-$RK.log 2>&1; exec 9> /tmp/hb_side01b.lock; flock -n 9 || exit 0
echo "$(date -u) side01b rank $RK start"
for i in $(seq 360); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done
touch $L/side01b-start-node-$RK
for i in $(seq 720); do [ -f $L/side01b-start-node-0 ] && [ -f $L/side01b-start-node-1 ] && break; sleep 5; done
echo "$(date -u) side01b rank $RK START psi0 a17cr4"
bash $G/hb_groot/gp_scaled_mix.sh psi0:a17cr4:4000:3200 none 32128 0; rc=$?
echo $rc > $L/side01b-a17cr4.rc-node-$RK; echo "$(date -u) side01b rank $RK END rc=$rc"
