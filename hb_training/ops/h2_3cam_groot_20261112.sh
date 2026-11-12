#!/bin/bash
# 2026-11-12 22:0x: GR00T 3-camera views for A18 / A17 (+ CPU checks); A17 waits for its wrist-camera download.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z; G=$P/policies-20261109
E=$G/envs/groot-py312; GR=$G/HumanWAM/third_party/Isaac-GR00T
export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$GR:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 GROOT_SRC=$GR
export HB_BODY_STATE_KEYS=left_leg,right_leg,waist,left_arm,right_arm,head
while pgrep -f "prepare_groot_data.py a18" > /dev/null; do sleep 20; done
cd $GR; echo "$(date) verify a18"; nice -n 10 python $G/hb_groot/verify_groot_cpu.py $G/groot-data/a18 2>&1 | grep CPU_VERIFY
until grep -q "A17_WRIST_DONE" $P/dl-a17-wrist-20261112.log; do grep -q GAVE_UP $P/dl-a17-wrist-20261112.log && { echo "A17 wrist download gave up"; exit 1; }; sleep 30; done
echo "$(date) prepare a17"; mv $G/groot-data/a17 $G/groot-data/a17.2cam-20261112
cd $G/hb_groot && nice -n 10 env -u RANK -u WORLD_SIZE -u MASTER_ADDR python prepare_groot_data.py a17 2>&1 | tail -2
cd $GR; echo "$(date) verify a17"; nice -n 10 python $G/hb_groot/verify_groot_cpu.py $G/groot-data/a17 2>&1 | grep CPU_VERIFY
echo "$(date) H2_3CAM_DONE"
