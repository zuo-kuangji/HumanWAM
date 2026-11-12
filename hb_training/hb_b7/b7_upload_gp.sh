#!/bin/bash
# Uploads GR00T B7 and pi0.5 B7 (same pipeline as the batch-696 round) once both are complete.
G=$(dirname "$(dirname "$(realpath "$0")")")
P=$(dirname "$G")
T=$P/hf-sidecar-tools-20261110
PY=$G/envs/psi0/bin/python
export HB_TOKEN_FILE=/root/.hb-secrets/hf_token HF_HUB_DISABLE_PROGRESS_BARS=1
exec >> "$G/b7-20261111/upload_gp.log" 2>&1
until [ -f $G/groot-runs-b696/b7/production-complete.json ] && ls $G/pi05-runs/b512/b7/run/ckpt/*/run/HB_DONE >/dev/null 2>&1; do sleep 120; done
until grep -qE 'ALL_DONE|_FAILED|not finished' $P/hf-models-20261111-b696.log; do sleep 120; done   # a17 round 2 first
echo "$(date) both done"
(cd $G/hb_pi05 && source ./hb_env.sh && python "$T/repro_supplement.py" "$P" pi05cfg) || echo PI05CFG_FAILED
"$PY" "$T/build_b696_repos.py" "$P" groot pi05 || { echo BUILD_FAILED; exit 1; }
"$PY" "$T/upload_model_repos2.py" "$P/hf-models-20261111-b696" groot pi05 || { echo UPLOAD_FAILED; exit 2; }
for r in groot pi05; do "$PY" "$T/prune_remote2.py" "$P/hf-models-20261111-b696/$r" $r "Add b7-black-clothes-hamper" --apply || { echo PRUNE_FAILED; exit 3; }; done
echo "$(date) B7_GP_UPLOADED"
