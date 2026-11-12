#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z; G=$P/policies-20261109
pkill -f 'psi0_h2_3cam_prep_20261113\.sh|psi0_3cam_release_20261113\.sh'; sleep 1
pkill -f 'verify_cpu\.py --task a1[478]|prepare_data\.py --force a1[478]'; sleep 3
pgrep -af 'psi0_h2_3cam|psi0_3cam_release|task a1[478]' | grep -v restart || echo NONE_LEFT
mv $G/psi0-3cam-20261113/prep.log $G/psi0-3cam-20261113/prep-run1-oldverify.log
$G/envs/psi0/bin/python -m py_compile $G/hb_psi0/verify_cpu.py && echo COMPILED
cd $P && setsid nohup bash $G/psi0_h2_3cam_prep_20261113.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash $G/psi0_3cam_release_20261113.sh > /dev/null 2>&1 < /dev/null &
setsid nohup bash -c "env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 19 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task a1 > $G/a1-20261113/psi0-verify2.log 2>&1; echo PSI0_VERIFY2_RC \$? >> $G/a1-20261113/prep.log" > /dev/null 2>&1 < /dev/null &
sleep 2; pgrep -af 'psi0_h2_3cam|psi0_3cam_release|verify_cpu' | cut -c1-110
