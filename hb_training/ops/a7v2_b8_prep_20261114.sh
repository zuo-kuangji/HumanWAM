#!/bin/bash
# 2026-11-14: wait for the download, SHA-check, then prepare + CPU-verify A7 v2 and B8 for GR00T, Psi-Zero, Delta Flow.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/a7v2-b8-20261114
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
until grep -q -e ALL_DONE -e GAVE_UP $P/dl-a7v2-b8-20261114.log; do sleep 20; done
grep -q ALL_DONE $P/dl-a7v2-b8-20261114.log || { echo DOWNLOAD_FAILED; exit 1; }
echo "$(date) download done"
python3 $G/a7v2_b8_check.py $P > $L/check.log 2>&1 && echo CHECK_OK || { echo CHECK_FAILED; exit 2; }
for t in a7v2 b8; do
  (E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python prepare_groot_data.py $t > $L/groot-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python verify_groot_cpu.py $G/groot-data/$t > $L/groot-verify-$t.log 2>&1) && echo "GROOT_OK $t" || echo "GROOT_FAILED $t"
done
for t in a7v2 b8; do
  (env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force $t > $L/psi0-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify-$t.log 2>&1) && echo "PSI0_OK $t" || echo "PSI0_FAILED $t"
done
(cd $G/hb_deltaflow && nice -n 15 $P/venv/bin/python prepare_deltaflow.py a7v2 b8 > $L/deltaflow-prepare.log 2>&1) && echo "DF_OK" || echo "DF_FAILED"
echo "$(date) ALL_DONE"
