#!/bin/bash
# 2026-11-13: rebuild the pi0.5 and DiT4DiT H2 views with head + chest + wrist (a18, a17) and build A14; CPU only,
# niced, after the Delta Flow 3-camera preparation. 2-camera views are kept as <task>.2cam-20261113.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/h2-3cam-20261113; mkdir -p $L; exec >> $L/prep.log 2>&1
echo "$(date) start (waiting for the Delta Flow 3-camera preparation)"
while pgrep -f prepare_deltaflow_h2w3.py > /dev/null; do sleep 60; done
( cd $G/hb_pi05 && source ./hb_env.sh
  for t in a18 a17; do [ -d "$HB_PI05_DATA/$t" ] && [ ! -e "$HB_PI05_DATA/$t.2cam-20261113" ] && mv "$HB_PI05_DATA/$t" "$HB_PI05_DATA/$t.2cam-20261113"; done
  for t in a18 a17 a14; do
    nice -n 19 python hb_prepare.py $t --workers 8 > $L/pi05-prepare-$t.log 2>&1 && nice -n 19 python verify_cpu.py --task $t > $L/pi05-verify-$t.log 2>&1 \
      && echo "PI05_3CAM_OK $t" || echo "PI05_3CAM_FAILED $t"
  done )
( cd $G/hb_dit4dit && source ./hb_env.sh
  R=$(python -c 'import hb_d4d_data as D; print(D.data_root())')
  for t in a18 a17; do [ -d "$R/$t" ] && [ ! -e "$R/$t.2cam-20261113" ] && mv "$R/$t" "$R/$t.2cam-20261113"; done
  for t in a18 a17 a14; do
    nice -n 19 python dit4dit_prepare.py $t --workers 6 > $L/d4d-prepare-$t.log 2>&1 && nice -n 19 python verify_cpu.py --task $t > $L/d4d-verify-$t.log 2>&1 \
      && echo "D4D_3CAM_OK $t" || echo "D4D_3CAM_FAILED $t"
  done )
echo "$(date) ALL_DONE"
