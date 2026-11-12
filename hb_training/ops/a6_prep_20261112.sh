#!/bin/bash
# 2026-11-12: prepare + CPU-verify A6 door-pull (100 ep) and the c6 8-episode adaptation test (c6t) for pi0.5 / Psi-Zero / GR00T.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/a6-20261112
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
until grep -q C6TEST_DONE $P/dl-a6-c6test-20261112.log; do grep -qE 'Traceback' $P/dl-a6-c6test-20261112.log && { echo DOWNLOAD_FAILED; exit 1; }; sleep 30; done
echo "$(date) downloads done"
for t in a6 c6t; do
  FAIL=0
  (cd $G/hb_pi05 && source ./hb_env.sh && python hb_prepare.py $t --workers 24 > $L/pi05-prepare-$t.log 2>&1 && python verify_cpu.py --task $t > $L/pi05-verify-$t.log 2>&1) || { echo "PI05 $t FAILED"; FAIL=1; }
  (env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force $t > $L/psi0-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify-$t.log 2>&1) || { echo "PSI0 $t FAILED"; FAIL=1; }
  (E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR python prepare_groot_data.py $t > $L/groot-prepare-$t.log 2>&1) || { echo "GROOT $t FAILED"; FAIL=1; }
  echo "$(date) $t done FAIL=$FAIL"
done
echo "$(date) ALL_DONE"
