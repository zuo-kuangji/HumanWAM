#!/bin/bash
# 2026-11-13: rebuild the Psi-Zero views of the H2 tasks with head + chest + wrist and CPU-verify them (niced).
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
L=$G/psi0-3cam-20261113; mkdir -p $L; exec >> $L/prep.log 2>&1
echo "$(date) start"
for t in a18 a17 a14; do
  (env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force $t > $L/prepare-$t.log 2>&1    && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/verify-$t.log 2>&1)    && echo "PSI0_3CAM_OK $t" || echo "PSI0_3CAM_FAILED $t"
done
echo "$(date) ALL_DONE"
