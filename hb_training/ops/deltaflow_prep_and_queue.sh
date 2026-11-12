#!/bin/bash
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
cd $G
nice -n 10 $P/venv/bin/python $G/hb_deltaflow/prepare_deltaflow.py > $G/deltaflow-prepare.log 2>&1 || { echo PREPARE_FAIL; exit 1; }
nice -n 10 $P/venv/bin/python $G/hb_deltaflow/verify_cpu.py > $G/deltaflow-verify.log 2>&1 || { echo VERIFY_FAIL; exit 1; }
grep -q deltaflow_b6.sh $G/relay/queue.txt || printf "cmd %s\n" $G/hb_deltaflow/deltaflow_b6.sh $G/hb_deltaflow/deltaflow_a3.sh $G/hb_deltaflow/deltaflow_a7.sh >> $G/relay/queue.txt
echo QUEUED; tail -4 $G/relay/queue.txt
