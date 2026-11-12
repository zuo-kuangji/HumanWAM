#!/bin/bash
# 2026-11-12: Delta Flow + DiT4DiT data preparation for C6 once the other preparations have finished (CPU, master-0).
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
P=$(dirname $G)
while pgrep -f "prepare_deltaflow.py b6" >/dev/null || pgrep -f "dit4dit_prepare.py trolley" >/dev/null || pgrep -f c6_prep_20261112 >/dev/null; do sleep 60; done
echo "$(date) start DF c6"
cd $G/hb_deltaflow && nice -n 10 $P/venv/bin/python prepare_deltaflow.py c6 >> $G/deltaflow-runs/logs/prepare-aligned-c6-20261112.log 2>&1; echo "DF_C6_RC $?"
echo "$(date) start D4D c6"
cd $G/hb_dit4dit && source hb_env.sh && nice -n 19 python dit4dit_prepare.py c6 --workers 8 >> $G/dit4dit-data/prepare-c6-20261112.log 2>&1; echo "D4D_C6_RC $?"
echo "$(date) done"
