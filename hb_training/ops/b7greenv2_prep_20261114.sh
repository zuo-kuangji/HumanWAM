#!/bin/bash
# 2026-11-13: download + prepare + CPU-verify B7 green clothes v2 (120 ep) for GR00T (niced, idle IO).
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/b7greenv2-20261114
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
until grep -q -e B7GREENV2_DONE -e GAVE_UP $P/dl-b7greenv2-20261114.log; do sleep 20; done
grep -q B7GREENV2_DONE $P/dl-b7greenv2-20261114.log || { echo DOWNLOAD_FAILED; exit 1; }
echo "$(date) download done"
(E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 ionice -c3 python prepare_groot_data.py b7greenv2 > $L/groot-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 ionice -c3 python verify_groot_cpu.py $G/groot-data/b7greenv2 > $L/groot-verify.log 2>&1) && echo "GROOT_OK" || echo "GROOT_FAILED"
echo "$(date) ALL_DONE"
