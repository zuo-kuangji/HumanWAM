#!/bin/bash
# 2026-11-13 14:0x user: DiT4DiT H2 3-camera tasks run per-GPU 1 x accumulation 2 (global 64) -> release them once
# their 3-camera views passed dit4dit_prepare.py + verify_cpu.py.
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
L=$G/h2-3cam-20261113
until grep -q ALL_DONE $L/prep.log 2>/dev/null; do sleep 60; done
sleep 120
for t in a18 a17 a14; do
  if grep -q "D4D_3CAM_OK $t" $L/prep.log; then date -u +%FT%TZ > $G/control/h2-3cam-ready-dit4dit-$t; echo "$(date) RELEASED dit4dit $t (per-GPU 1 x accumulation 2)" >> $L/release.log
  else echo "$(date) NOT_RELEASED dit4dit $t" >> $L/release.log; fi
done
