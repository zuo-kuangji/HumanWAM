#!/bin/bash
# One-shot CPU pipeline on one node: wait for build_env.sh, build data views for all 5 tasks, CPU verify per task.
# Usage: setsid nohup bash prep_all.sh > $G/psi0-prep-all.log 2>&1 &
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
until grep -q BUILD_ENV_OK "$G/hb_psi0_build_env.log" 2>/dev/null; do
  pgrep -f "hb_psi0/build_env.sh" > /dev/null || { echo "BUILD_ENV_FAILED (see $G/hb_psi0_build_env.log)"; exit 1; }
  sleep 30
done
echo "$(date -u) env ready"
PY=$G/envs/psi0/bin/python
# basket is postponed (user 2026-11-10); its raw state layout also differs (body joints not at state[0:29])
for t in b6 a3 a7 a18; do
  env -u RANK -u WORLD_SIZE -u MASTER_ADDR "$PY" "$H/prepare_data.py" --force "$t" >> "$G/psi0-prepare.log" 2>&1 \
    || { echo "PREPARE_FAIL $t (see $G/psi0-prepare.log)"; exit 1; }
done
echo "$(date -u) data ready"
for t in b6 a3 a7 a18; do
  env -u RANK -u WORLD_SIZE -u MASTER_ADDR "$PY" "$H/verify_cpu.py" --task "$t" > "$G/psi0-verify-$t.log" 2>&1
  echo "VERIFY $t exit $?"
done
echo "$(date -u) PREP_ALL_DONE"
