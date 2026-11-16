#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z; G=$P/policies-20261109; L=$G/h2-3cam-20261113
cd $G/hb_pi05 && source ./hb_env.sh
for t in a18 a17; do
  nice -n 19 ionice -c3 python verify_cpu.py --task $t > $L/pi05-verify2-$t.log 2>&1 && echo "PI05_3CAM_OK $t (re-verified with the aligned proprio checks)" >> $L/prep.log || echo "PI05_3CAM_FAILED2 $t" >> $L/prep.log
done
