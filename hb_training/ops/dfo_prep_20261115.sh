#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/dfalign-20261115
exec >> $L/prep.log 2>&1
echo "$(date) start a3v4"
cd $G/hb_deltaflow
nice -n 15 $P/venv/bin/python prepare_deltaflow_b6b5.py a3v4 > $L/prepare-a3v4.log 2>&1 && grep -q B6B5_DF_OK $L/prepare-a3v4.log && echo "$(date) PREP_OK a3v4" || echo "$(date) PREP_FAILED a3v4"
C=$G/deltaflow-runs/HumanWAM-allep
(cd $C && PYTHONPATH=$C/src $P/venv/bin/python scripts/wholebody/train_deltaflow.py --config $G/deltaflow-runs/a3v4/run/production.yaml --check-only > $L/check-a3v4.log 2>&1) \
  && grep -q CONFIG_OK $L/check-a3v4.log && { touch $L/deltaflow-a3v4.ok; echo "$(date) DF_OK a3v4"; } || echo "$(date) DF_CHECK_FAILED a3v4"
