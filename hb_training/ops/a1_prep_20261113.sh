#!/bin/bash
# 2026-11-13: download + prepare + CPU-verify A1 for GR00T, Psi-Zero and Delta Flow (niced; master also trains).
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/a1-20261113
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
$G/envs/psi0/bin/python $G/dl_a1.py $P > $P/dl-a1-20261113.log 2>&1
grep -q A1_DONE $P/dl-a1-20261113.log || { echo DOWNLOAD_FAILED; exit 1; }
echo "$(date) download done"
(E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 python prepare_groot_data.py a1 > $L/groot-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 python verify_groot_cpu.py $G/groot-data/a1 > $L/groot-verify.log 2>&1) && echo "GROOT_OK" || echo "GROOT_FAILED"
(env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force a1 > $L/psi0-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task a1 > $L/psi0-verify.log 2>&1) && echo "PSI0_OK" || echo "PSI0_FAILED"
(cd $G/hb_deltaflow && nice -n 19 $P/venv/bin/python prepare_deltaflow.py a1 > $L/deltaflow-prepare.log 2>&1) && echo "DF_OK" || echo "DF_FAILED"
echo "$(date) ALL_DONE"
