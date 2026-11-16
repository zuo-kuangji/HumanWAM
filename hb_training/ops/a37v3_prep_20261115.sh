#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/a37v3-20261115
exec >> $L/prep.log 2>&1
echo "$(date) start"
until grep -q -e A37V3_DONE -e GAVE_UP $P/dl-a37v3-20261115.log 2>/dev/null; do sleep 30; done
grep -q A37V3_DONE $P/dl-a37v3-20261115.log || { echo DOWNLOAD_FAILED; touch $L/prep.done; exit 1; }
for t in a3v3 a7v3; do
  python3 $G/a37v3_check.py $P $t > $L/check-$t.log 2>&1 && echo "CHECK_OK $t" || { echo "CHECK_FAILED $t"; touch $L/prep.done; exit 2; }
done
g=0
for t in a3v3 a7v3; do
  n=125; [ $t = a7v3 ] && n=100
  if (E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python prepare_groot_data.py $t > $L/groot-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python verify_groot_cpu.py $G/groot-data/$t > $L/groot-verify-$t.log 2>&1) \
     && python3 $G/a37v3_groot_check.py $L/groot-verify-$t.log $n; then
    g=$((g + 1)); echo "GROOT_OK $t"
  else
    echo "GROOT_FAILED $t"
  fi
done
[ $g = 2 ] && touch $L/groot.ok
p=0
for t in a3v3 a7v3; do
  if env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force $t > $L/psi0-prepare-$t.log 2>&1 \
     && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify-$t.log 2>&1 \
     && grep -q VERIFY_OK $L/psi0-verify-$t.log; then
    p=$((p + 1)); echo "PSI0_OK $t"
  else
    echo "PSI0_FAILED $t"
  fi
done
[ $p = 2 ] && touch $L/psi0.ok
touch $L/prep.done
echo "$(date) ALL_DONE groot=$g psi0=$p"
