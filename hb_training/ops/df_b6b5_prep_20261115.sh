#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/b6v6-20261115
exec >> $L/df-prep.log 2>&1
echo "$(date) start"
until [ -f $L/prep-b6v6-redo.done ] && [ -f $L/prep.done ]; do sleep 30; done
for t in b5 b6v6; do
  if [ ! -f $L/groot-$t.ok ] && [ ! -f $L/psi0-$t.ok ]; then echo "SKIP $t: data check failed"; continue; fi
  if (cd $G/hb_deltaflow && nice -n 15 $P/venv/bin/python prepare_deltaflow_b6b5.py $t > $L/deltaflow-prepare-$t.log 2>&1) && grep -q B6B5_DF_OK $L/deltaflow-prepare-$t.log; then
    C=$G/deltaflow-runs/HumanWAM-allep
    (cd $C && PYTHONPATH=$C/src $P/venv/bin/python scripts/wholebody/train_deltaflow.py --config $G/deltaflow-runs/$t/run/production.yaml --check-only > $L/deltaflow-check-$t.log 2>&1) \
      && grep -q CONFIG_OK $L/deltaflow-check-$t.log && { touch $L/deltaflow-$t.ok; echo "$(date) DF_OK $t"; } || echo "$(date) DF_CHECK_FAILED $t"
  else
    echo "$(date) DF_PREP_FAILED $t"
  fi
done
echo "$(date) DF_ALL_DONE"
