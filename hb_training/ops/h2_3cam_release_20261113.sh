#!/bin/bash
# 2026-11-13: release the guarded H2 queue items of Delta Flow / pi0.5 once their 3-camera preparation passed.
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
L=$G/h2-3cam-20261113; mkdir -p $L
while pgrep -f prepare_deltaflow_h2w3.py > /dev/null; do sleep 60; done
for t in a18 a17 a14; do
  if grep -q "H2W3_PREPARED $t" $G/deltaflow-runs/logs/prepare-h2w3-20261113.log && grep -q head $G/deltaflow-runs/$t/data-verified.json && grep -q wrist $G/deltaflow-runs/$t/data-verified.json; then
    date -u +%FT%TZ > $G/control/h2-3cam-ready-deltaflow-$t; echo "$(date) RELEASED deltaflow $t" >> $L/release.log
  else echo "$(date) NOT_RELEASED deltaflow $t" >> $L/release.log; fi
done
until grep -q ALL_DONE $L/prep.log 2>/dev/null; do sleep 60; done
for t in a18 a17 a14; do
  if grep -q "PI05_3CAM_OK $t" $L/prep.log; then date -u +%FT%TZ > $G/control/h2-3cam-ready-b256-$t; echo "$(date) RELEASED pi05 $t" >> $L/release.log
  else echo "$(date) NOT_RELEASED pi05 $t" >> $L/release.log; fi
done
echo "$(date) DiT4DiT stays guarded until its 3-camera memory plan (per-GPU 1 x accumulation 2) is in place" >> $L/release.log
