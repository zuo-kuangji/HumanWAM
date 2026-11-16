#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/a10v3-20261115
exec >> $L/prep.log 2>&1
echo "$(date) start"
until grep -q -e A10V3_DONE -e GAVE_UP $P/dl-a10v3-20261115.log 2>/dev/null; do sleep 30; done
grep -q A10V3_DONE $P/dl-a10v3-20261115.log || { echo DOWNLOAD_FAILED; exit 1; }
python3 $G/a10v3_check.py $P > $L/check.log 2>&1 && echo CHECK_OK || { echo CHECK_FAILED; exit 2; }
(E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python prepare_groot_data.py a10v3 > $L/groot-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python verify_groot_cpu.py $G/groot-data/a10v3 > $L/groot-verify.log 2>&1) && python3 - $L/groot-verify.log <<'PY' && { touch $L/groot.ok; echo GROOT_OK; } || echo GROOT_FAILED
import json, sys
d = [json.loads(l.split(' ', 1)[1]) for l in open(sys.argv[1]) if l.startswith('CPU_VERIFY')][0]
ok = d['episodes'] == 148 and d['state_total'] == 44 and d['action_total'] == 76 and d['patched'] and d['proprio_max_abs_deg'] <= 2.75
print('GROOT_CHECK', ok, d['episodes'], d['state_total'], d['action_total'], sorted(d['video']), d['proprio_max_abs_deg']); sys.exit(0 if ok else 1)
PY
(env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force a10v3 > $L/psi0-prepare.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task a10v3 > $L/psi0-verify.log 2>&1 && grep -q VERIFY_OK $L/psi0-verify.log) && { touch $L/psi0.ok; echo PSI0_OK; } || echo PSI0_FAILED
(cd $G/hb_deltaflow && nice -n 15 $P/venv/bin/python prepare_deltaflow_a10v3.py > $L/deltaflow-prepare.log 2>&1 && grep -q A10V3_DF_OK $L/deltaflow-prepare.log) && echo DF_OK || echo DF_FAILED
echo "$(date) ALL_DONE"
