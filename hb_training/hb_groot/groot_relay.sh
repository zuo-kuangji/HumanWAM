#!/bin/bash
# hb_relay "cmd" item helper: GR00T 20-epoch production (save at epoch 10 and 20) for ONE task on all 4 nodes.
# The relay passes no arguments, so the queue uses the per-task wrappers groot_<task>.sh, which call this.
# Batch: probed once (64 48 40 32 24 16 12 8) on the first task run, stored in $G/groot-runs/batch-decision.json.
H=$(dirname "$(realpath "$0")")
G=$(dirname "$H")
T=${1:?usage: groot_relay.sh <task>}
case $T in b6) i=0 ;; a3) i=1 ;; a7) i=2 ;; a18) i=3 ;; trolley) i=4 ;; basket) i=5 ;; a17) i=6 ;; *) echo "unknown task $T"; exit 64 ;; esac
ls "$G/models/groot-n17-base"/*.safetensors > /dev/null 2>&1 || { echo "GR00T base weights missing at $G/models/groot-n17-base"; exit 4; }
R2=$G/hf-home/hub/models--nvidia--Cosmos-Reason2-2B
[ "$(cat "$R2/refs/main" 2>/dev/null)" = 9ce19a195e423419c349abfc86fd07178b230561 ] \
  && ls "$R2/snapshots/9ce19a195e423419c349abfc86fd07178b230561"/*.safetensors > /dev/null 2>&1 \
  || { echo "NEED_TOKEN: gated nvidia/Cosmos-Reason2-2B not in $G/hf-home (cache_cosmos_reason2.py)"; exit 4; }
export HB_GROOT_PORT_BASE=$((32500 + 100 * i))   # 31500+ was used by the per-GPU-64 runs in groot-runs/
exec python3 "$H/groot_chain.py" "$T"
