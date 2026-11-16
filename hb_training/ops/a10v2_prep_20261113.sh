#!/bin/bash
# 2026-11-13: download + prepare + CPU-verify A10 v2 (150 episodes) for GR00T and Psi-Zero (niced, idle IO).
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/a10v2-20261113
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
nice -n 19 ionice -c3 $G/envs/psi0/bin/python $G/dl_a10v2.py $P > $P/dl-a10v2-20261113.log 2>&1
grep -q A10V2_DONE $P/dl-a10v2-20261113.log || { echo DOWNLOAD_FAILED; exit 1; }
echo "$(date) download done"
(E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 ionice -c3 python prepare_groot_data.py a10v2 > $L/groot-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 ionice -c3 python verify_groot_cpu.py $G/groot-data/a10v2 > $L/groot-verify.log 2>&1) && echo "GROOT_OK" || echo "GROOT_FAILED"
(env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 ionice -c3 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force a10v2 > $L/psi0-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 ionice -c3 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task a10v2 > $L/psi0-verify.log 2>&1) && echo "PSI0_OK" || echo "PSI0_FAILED"
echo "$(date) ALL_DONE"
