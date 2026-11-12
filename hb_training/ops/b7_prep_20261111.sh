#!/bin/bash
# 2026-11-11: prepare + CPU-verify B7 (black clothes -> hamper, first 100 of 107) for pi0.5 / Psi-Zero / GR00T / ACT.
# No queue changes here; insertion is done separately after the 3+1-node scheduling item exists.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/b7-20261111
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
FAIL=0
t=b7
(cd $G/hb_pi05 && source ./hb_env.sh && python hb_prepare.py $t --workers 24 > $L/pi05-prepare.log 2>&1 && python verify_cpu.py --task $t > $L/pi05-verify.log 2>&1) || { echo "PI05 FAILED"; FAIL=1; }
echo "$(date) pi05 done FAIL=$FAIL"
(env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force $t > $L/psi0-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify.log 2>&1) || { echo "PSI0 FAILED"; FAIL=1; }
echo "$(date) psi0 done FAIL=$FAIL"
(E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR python prepare_groot_data.py $t > $L/groot-prepare.log 2>&1) || { echo "GROOT FAILED"; FAIL=1; }
echo "$(date) groot done FAIL=$FAIL"
(PSI=$G/HumanWAM/third_party/Psi0; cd $PSI; export PYTHONPATH=$PSI/src:$G/hb_act:$G/hb_psi0 PSI_SRC=$PSI HB_AUG_DIR=$G/hb_groot TORCH_HOME=$G/torch-home HF_HOME=$G/hf-home HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/tmp/hb_act_hfds; env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_act/verify_cpu.py --task $t > $L/act-verify.log 2>&1) || { echo "ACT FAILED"; FAIL=1; }
echo "$(date) act done FAIL=$FAIL"
[ "$FAIL" = 0 ] && echo "$(date) B7_PREP_OK" || echo "$(date) B7_PREP_FAILED"
