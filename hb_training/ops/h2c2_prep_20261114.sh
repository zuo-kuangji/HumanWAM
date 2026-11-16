#!/bin/bash
# 2026-11-14: A18 / A17 2-camera (head + chest) views for GR00T, Psi-Zero, Delta Flow + CPU checks (niced).
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/h2c2-20261114
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
for t in a18c2 a17c2; do
  (E=$G/envs/groot-py312; GR=$G/HumanWAM/third_party/Isaac-GR00T; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$GR:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROOT_SRC=$GR HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm,head; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python prepare_groot_data.py $t > $L/groot-prepare-$t.log 2>&1 && cd $GR && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python $G/hb_groot/verify_groot_cpu.py $G/groot-data/$t > $L/groot-verify-$t.log 2>&1) && echo "GROOT_OK $t" || echo "GROOT_FAILED $t"
done
for t in a18c2 a17c2; do
  (env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --cams head,chest --force $t > $L/psi0-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify-$t.log 2>&1) && echo "PSI0_OK $t" || echo "PSI0_FAILED $t"
done
(cd $G/hb_deltaflow && nice -n 15 $P/venv/bin/python prepare_deltaflow_h2.py a18c2 a17c2 > $L/deltaflow-prepare.log 2>&1) && echo "DF_OK" || echo "DF_FAILED"
echo "$(date) ALL_DONE"
