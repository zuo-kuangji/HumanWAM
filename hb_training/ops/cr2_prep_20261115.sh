#!/bin/bash
# 2026-11-15 22:xx: GR00T data views for A18 / A17 cropped v2 (a18cr2, a17cr2; H2, head + chest, state 78, action 75):
# hash check, GR00T view + CPU verify. Markers in $L: groot-<t>.ok / .failed.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/h2cr2-20261115
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
one() {
  local t=$1
  until grep -q -e "CR2_FULL $t" -e "GAVE_UP $t" $P/dl-cr2-20261115.log 2>/dev/null; do sleep 20; done
  if ! grep -q "CR2_FULL $t" $P/dl-cr2-20261115.log; then echo "DOWNLOAD_FAILED $t"; touch $L/groot-$t.failed; return; fi
  if ! python3 $G/cr2_check.py $P $t > $L/check-$t.log 2>&1; then echo "CHECK_FAILED $t"; touch $L/groot-$t.failed; return; fi
  echo "$(date) CHECK_OK $t"
  if (E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python prepare_groot_data.py $t > $L/groot-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python verify_groot_cpu.py $G/groot-data/$t > $L/groot-verify-$t.log 2>&1) \
     && python3 $G/cr2_groot_check.py $L/groot-verify-$t.log; then
    touch $L/groot-$t.ok; echo "$(date) GROOT_OK $t"
  else
    touch $L/groot-$t.failed; echo "$(date) GROOT_FAILED $t"
  fi
}
one a18cr2 &
one a17cr2 &
wait
touch $L/prep.done
echo "$(date) ALL_DONE"
