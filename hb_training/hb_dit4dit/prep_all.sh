#!/bin/bash
# One-shot CPU pipeline on ONE node: wait for build_env.sh, then (in parallel) build the Cosmos pipeline dir
# (fetch_models.sh) and the data caches of b6 a3 a7 a18, then CPU verify per task.
# Usage: setsid nohup bash $G/hb_dit4dit/prep_all.sh > $G/dit4dit-prep-all.log 2>&1 &
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
BUILD_LOG=${HB_BUILD_LOG:-$G/envs/build_dit4dit.log}
until grep -q BUILD_ENV_OK "$BUILD_LOG" 2>/dev/null; do
  pgrep -f "hb_dit4dit/build_env.sh" > /dev/null || { echo "BUILD_ENV_FAILED (see $BUILD_LOG)"; exit 1; }
  sleep 30
done
echo "$(date -u) env ready"
source "$H/hb_env.sh"
unset RANK WORLD_SIZE MASTER_ADDR
mkdir -p "$HB_D4D_DATA" "$HB_G/models"
bash "$H/fetch_models.sh" > "$HB_G/models/fetch_dit4dit.log" 2>&1 &
FETCH=$!
TASKS=(b6 a3 a7 a18)   # basket postponed (user 2026-11-10)
"$HB_ENV/bin/python" "$H/dit4dit_prepare.py" "${TASKS[@]}" --workers "${HB_PREP_WORKERS:-32}" > "$HB_D4D_DATA/prepare.log" 2>&1 \
  || { echo "PREPARE_FAIL (see $HB_D4D_DATA/prepare.log)"; wait $FETCH; exit 1; }
echo "$(date -u) data ready: $(grep -c PREPARE_OK "$HB_D4D_DATA/prepare.log") PREPARE_OK"
wait $FETCH
if grep -q MODELS_OK "$HB_G/models/fetch_dit4dit.log"; then
  echo "$(date -u) models ready"
elif grep -q NEED_TOKEN "$HB_G/models/fetch_dit4dit.log"; then
  echo "$(date -u) WAITING_FOR_TOKEN: public parts $(grep -c PUBLIC_PARTS_OK "$HB_G/models/fetch_dit4dit.log") ok; official gated transformer needs HF_TOKEN (rerun fetch_models.sh)"
else
  echo "FETCH_FAIL (see $HB_G/models/fetch_dit4dit.log); continuing with CPU verify of data/config"
fi
for t in "${TASKS[@]}"; do
  "$HB_ENV/bin/python" "$H/verify_cpu.py" --task "$t" > "$HB_G/dit4dit-verify-$t.log" 2>&1
  echo "VERIFY $t exit $?"
done
echo "$(date -u) PREP_ALL_DONE"
