#!/bin/bash
# One-shot CPU pipeline on one node: wait for build_env.sh, 76-D init from lerobot/pi05_base, data caches for all
# 5 tasks, CPU verify per task. Usage: setsid nohup bash prep_all.sh > $G/pi05-prep-all.log 2>&1 &
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
until grep -q BUILD_ENV_OK "$G/envs/build_openpi.log" 2>/dev/null; do
  pgrep -f "hb_pi05/build_env.sh" > /dev/null || { echo "BUILD_ENV_FAILED (see $G/envs/build_openpi.log)"; exit 1; }
  sleep 30
done
echo "$(date -u) env ready"
source "$H/hb_env.sh"
mkdir -p "$HB_PI05_DATA"
cd "$H"
if [ ! -f "$HB_PI05_INIT/model.safetensors" ]; then
  bash "$H/convert_base.sh" > "$G/pi05-convert.log" 2>&1 || { echo "CONVERT_FAIL (see $G/pi05-convert.log)"; exit 1; }
fi
echo "$(date -u) init ready"
python hb_prepare.py b6 basket a3 a7 a18 --workers 32 > "$HB_PI05_DATA/prepare.log" 2>&1 \
  || { echo "PREPARE_FAIL (see $HB_PI05_DATA/prepare.log)"; exit 1; }
echo "$(date -u) data ready"
for t in b6 basket a3 a7 a18; do
  python verify_cpu.py --task "$t" > "$G/pi05-verify-$t.log" 2>&1
  echo "VERIFY $t exit $?"
done
echo "$(date -u) PREP_ALL_DONE"
