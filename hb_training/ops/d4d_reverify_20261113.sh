#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z; G=$P/policies-20261109; L=$G/h2-3cam-20261113
cd $G/hb_dit4dit && source ./hb_env.sh
for t in a18 a17; do
  if nice -n 19 ionice -c3 python verify_cpu.py --task $t > $L/d4d-verify2-$t.log 2>&1; then
    echo "D4D_3CAM_OK $t (re-verified, 3-view frame check)" >> $L/prep.log
    date -u +%FT%TZ > $G/control/h2-3cam-ready-dit4dit-$t; echo "$(date) RELEASED dit4dit $t (per-GPU 1 x accumulation 2)" >> $L/release.log
  else echo "D4D_3CAM_FAILED2 $t" >> $L/prep.log; fi
done
