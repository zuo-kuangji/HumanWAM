#!/bin/bash
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109; E=$G/eval-action-mse
export LD_LIBRARY_PATH=$G/envs/groot-py312/lib/hbstub:$LD_LIBRARY_PATH
cd $E
for t in b7gt a14cr3; do
  [ -f cache/$t/train10/meta.json ] && { echo "$(date) $t cached"; continue; }
  A=""; [ $t = a14cr3 ] && A="--cams head,chest"
  nice -n 15 $G/envs/groot-py312/bin/python build_cache.py $t train10 --workers 6 $A > cache-build-$t-20261122.log 2>&1 && echo "$(date) CACHE_OK $t" || echo "$(date) CACHE_FAILED $t"
done
echo "$(date) ALL_DONE"
