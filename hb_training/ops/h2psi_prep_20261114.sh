#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/h2psi-20261114
exec >> $L/prep.log 2>&1
echo "$(date) start"
until grep -q -e CHECK_OK -e CHECK_FAILED -e DOWNLOAD_FAILED $G/a14v2-20261114/prep.log 2>/dev/null; do sleep 30; done
grep -q CHECK_OK $G/a14v2-20261114/prep.log || { echo "A14V2 DATA CHECK NOT OK"; exit 1; }
ok=1
for t in a18c2 a14v2; do
  if [ "$t" = a14v2 ]; then
    env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --cams head,chest --force a14v2 > $L/psi0-prepare-a14v2.log 2>&1 || { echo "PREPARE_FAILED a14v2"; ok=0; continue; }
  fi
  env -u RANK -u WORLD_SIZE -u MASTER_ADDR nice -n 15 $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify-$t.log 2>&1 && grep -q VERIFY_OK $L/psi0-verify-$t.log && echo "PSI0_OK $t" || { echo "PSI0_FAILED $t"; ok=0; }
  HB_DELAY_PROB=0.15 $G/envs/psi0/bin/python $G/hb_psi0/hb_args.py args $t --batch 32 --steps 4000 --save 2000 --warmup 100 --out /tmp/x --timestamp b512 > $L/args-$t.txt 2>&1 || ok=0
done
[ $ok = 1 ] && touch $L/psi0.ok && echo READY
echo "$(date) ALL_DONE"
