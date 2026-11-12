#!/bin/bash
# 2026-11-12 07:30: after their early stage256 jobs (node 2 GR00T C6, node 3 pi0.5 A7) nodes 2+3 run the next queue item,
# pi0.5 A3 b256, on 2 nodes (2 x 8 x 16 = 256) instead of idling until nodes 0+1 finish pi0.5 A6 (~12:05).
# pi05_quad_a3.sh (relay item 33) sees the markers and only waits for this run.
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
NODE=$1; F=$G/b256-20261112; LOG=$F/fill2-node-$NODE.log
exec >> "$LOG" 2>&1
echo "$(date) node $NODE waiting for fill-node-$NODE.rc"
until [ -f "$F/fill-node-$NODE.rc" ]; do sleep 30; done
[ "$(cat "$F/fill-node-$NODE.rc")" = 0 ] || { echo "first fill failed, not starting A3"; exit 1; }
touch "$G/control/hold-node-$NODE"
if [ "$NODE" = 2 ]; then
  until [ -f "$F/fill-node-3.rc" ]; do sleep 30; done
  hostname -i | awk '{print $1}' > "$F/fill2_master.tmp"; mv "$F/fill2_master.tmp" "$F/fill2_master"
  touch "$F/fill2.running"; R=0
else
  until [ -s "$F/fill2_master" ]; do sleep 10; done; R=1
fi
MA=$(cat "$F/fill2_master")
echo "$(date) start pi0.5 A3 rank $R master $MA"
bash "$G/hb_b256/pi05_multi.sh" a3 2 "$R" "$MA" 30715 16
rc=$?
echo "$rc" > "$F/fill2-node-$NODE.rc"
[ "$NODE" = 2 ] && { echo "$rc" > "$F/fill2.rc"; rm -f "$F/fill2.running"; }
rm -f "$G/control/hold-node-$NODE"
echo "$(date) end rc=$rc"
