#!/bin/bash
# 2026-11-15 17:xx: data views for the B6 v6 continuations (task b6v6, 542 episodes) and B5 (task b5, 100 episodes):
# hash check, GR00T view + CPU verify (state 44, action 76), Psi-Zero view + CPU verify. Per-task markers in $L:
# groot-<t>.ok / .failed, psi0-<t>.ok / .failed. The two tasks run in parallel.
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/b6v6-20261115
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
one() {  # $1 task  $2 download log  $3 done tag  $4 episodes
  local t=$1 n=$4
  until grep -q -e "$3" -e GAVE_UP "$2" 2>/dev/null; do sleep 20; done
  if ! grep -q "$3" "$2"; then echo "DOWNLOAD_FAILED $t"; touch $L/groot-$t.failed $L/psi0-$t.failed; return; fi
  if ! python3 $G/b6b5_check.py $P $t > $L/check-$t.log 2>&1; then
    echo "CHECK_FAILED $t"; touch $L/groot-$t.failed $L/psi0-$t.failed; return
  fi
  echo "$(date) CHECK_OK $t"
  if (E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python prepare_groot_data.py $t > $L/groot-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 10 python verify_groot_cpu.py $G/groot-data/$t > $L/groot-verify-$t.log 2>&1) \
     && python3 $G/a37v3_groot_check.py $L/groot-verify-$t.log $n; then
    touch $L/groot-$t.ok; echo "$(date) GROOT_OK $t"
  else
    touch $L/groot-$t.failed; echo "$(date) GROOT_FAILED $t"
  fi
  if env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force $t > $L/psi0-prepare-$t.log 2>&1 \
     && env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify-$t.log 2>&1 \
     && grep -q VERIFY_OK $L/psi0-verify-$t.log; then
    touch $L/psi0-$t.ok; echo "$(date) PSI0_OK $t"
  else
    touch $L/psi0-$t.failed; echo "$(date) PSI0_FAILED $t"
  fi
}
one b6v6 $P/dl-b6v6-20261115.log B6V6_DONE 542 &
one b5 $P/dl-b5-20261115.log B5_DONE 100 &
wait
touch $L/prep.done
echo "$(date) ALL_DONE"
