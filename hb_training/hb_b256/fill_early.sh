#!/bin/bash
# 2026-11-12 05:30: start this node's stage256 job as soon as its GR00T b256 fill run is finished, instead of idling at
# the relay barrier until nodes 0+1 finish Psi-Zero B7 (~07:40). stage256.sh sees the marker files and does not re-run it.
#   node 2: GR00T C6 (8 x 32 = 256)   after GR00T A3 b256      node 3: pi0.5 A7 b256 (8 x 32)   after GR00T A6
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
NODE=$1; F=$G/b256-20261112; LOG=$F/fill-node-$NODE.log
exec >> "$LOG" 2>&1
case $NODE in 2) DONE=$G/groot-runs-b256/a3/production-complete.json ;; 3) DONE=$G/groot-runs-b256/a6/production-complete.json ;; *) exit 64 ;; esac
echo "$(date) waiting for $DONE"
until [ -f "$DONE" ]; do sleep 30; done
for i in $(seq 180); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done
if grep -q "item 32 start" "$G/relay/node-$NODE.log"; then echo "$(date) relay already in stage256, nothing to do"; exit 0; fi
touch "$G/control/hold-node-$NODE" "$F/fill-node-$NODE.running"
echo "$(date) start"
case $NODE in
  2) bash "$G/hb_groot/groot_single.sh" c6 32 "$G/groot-runs-b256" ;;
  3) bash "$G/hb_b256/pi05_multi.sh" a7 1 0 localhost 0 32 ;;
esac
rc=$?
echo "$rc" > "$F/fill-node-$NODE.rc"; rm -f "$F/fill-node-$NODE.running" "$G/control/hold-node-$NODE"
echo "$(date) end rc=$rc"
