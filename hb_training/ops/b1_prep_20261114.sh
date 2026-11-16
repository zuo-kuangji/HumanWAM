#!/bin/bash
# 2026-11-14: B1 (all 115 episodes) hash check, then GR00T + Psi-Zero data preparation and CPU verification.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/b1-20261114
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
python3 $G/b1_check.py $P > $L/check.log 2>&1 && echo CHECK_OK || { echo CHECK_FAILED; exit 2; }
(E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python prepare_groot_data.py b1 > $L/groot-prepare-b1.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python verify_groot_cpu.py $G/groot-data/b1 > $L/groot-verify-b1.log 2>&1) && echo "GROOT_OK b1" || echo "GROOT_FAILED b1"
(env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force b1 > $L/psi0-prepare-b1.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task b1 > $L/psi0-verify-b1.log 2>&1) && echo "PSI0_OK b1" || echo "PSI0_FAILED b1"
echo "$(date) ALL_DONE"
