#!/bin/bash
# 2026-11-13: when the Psi-Zero 3-camera H2 views pass verify_cpu.py (3 images per sample to Qwen3-VL, raw state 78 /
# action 75, no third_person), release the guarded queue items by creating control/h2-3cam-ready-b512-<task>.
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
L=$G/psi0-3cam-20261113
until grep -q ALL_DONE $L/prep.log; do sleep 60; done
for t in a18 a17 a14; do
  v=$L/verify-$t.log
  if grep -q "PSI0_3CAM_OK $t" $L/prep.log && grep -q "3 images per sample to Qwen3-VL" $v && ! grep -q third_person $G/psi0-data/$t/hb-view-receipt.json; then
    date -u +%FT%TZ > $G/control/h2-3cam-ready-b512-$t; echo "$(date) RELEASED $t" >> $L/prep.log
  else
    echo "$(date) NOT_RELEASED $t (verify failed or not 3 cameras)" >> $L/prep.log
  fi
done
