#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/dfalign-20261115
exec >> $L/prep.log 2>&1
echo "$(date) start"
cd $G/hb_deltaflow
nice -n 15 $P/venv/bin/python prepare_deltaflow_b6b5.py a7v4 > $L/prepare-a7v4.log 2>&1 && grep -q B6B5_DF_OK $L/prepare-a7v4.log && echo "$(date) PREP_OK a7v4" || echo "$(date) PREP_FAILED a7v4"
nice -n 15 $P/venv/bin/python prepare_deltaflow_a17cr2lg.py > $L/prepare-a17cr2lg.log 2>&1 && grep -q A17CR2LG_DF_OK $L/prepare-a17cr2lg.log && echo "$(date) PREP_OK a17cr2lg" || echo "$(date) PREP_FAILED a17cr2lg"
for t in a7v4 a17cr2lg; do
  C=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["code"])' $G/deltaflow-runs/$t/data-verified.json 2>/dev/null) || { echo "NO_RECEIPT $t"; continue; }
  (cd $C && PYTHONPATH=$C/src $P/venv/bin/python scripts/wholebody/train_deltaflow.py --config $G/deltaflow-runs/$t/run/production.yaml --check-only > $L/check-$t.log 2>&1) \
    && grep -q CONFIG_OK $L/check-$t.log && { touch $L/deltaflow-$t.ok; echo "$(date) DF_OK $t"; } || echo "$(date) DF_CHECK_FAILED $t"
done
echo "$(date) ALL_DONE"
