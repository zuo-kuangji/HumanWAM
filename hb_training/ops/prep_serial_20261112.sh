#!/bin/bash
# 2026-11-12: remaining CPU preparations, one at a time to keep master-0's training dataloaders fed.
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
P=$(dirname $G)
while pgrep -f "prepare_deltaflow.py b6" >/dev/null || pgrep -f "c6_prep_20261112|a6_prep_20261112" >/dev/null; do sleep 60; done
echo "$(date) DF c6"; cd $G/hb_deltaflow && nice -n 10 $P/venv/bin/python prepare_deltaflow.py c6 >> $G/deltaflow-runs/logs/prepare-aligned-c6-20261112.log 2>&1; echo "DF_C6_RC $?"
cd $G/hb_dit4dit && source hb_env.sh
for t in trolley basket b7 a6 a17 c6; do echo "$(date) D4D $t"; nice -n 19 python dit4dit_prepare.py $t --workers 6 >> $G/dit4dit-data/prepare-$t-20261112.log 2>&1; echo "D4D_${t}_RC $?"; done
echo "$(date) ALL_DONE"
