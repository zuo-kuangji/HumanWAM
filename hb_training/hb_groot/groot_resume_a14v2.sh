#!/bin/bash
# Relay cmd item (2026-11-15): resume GR00T A14 v2 (H2 guide a14 4.1; head + chest, action82, full FT, 2 x 8 x 32 = 512,
# no accumulation, 4000 steps) on nodes 0,1 from its latest complete checkpoint. The run leaks host RAM (~0.5 GB per
# step, as the 3-camera H2 runs), so it is restarted after every 1000-step checkpoint (--resume_from_checkpoint: weights,
# optimizer, scheduler, RNG; GR00T ignore_data_skip). Recipe and output dir unchanged (groot-runs-2x32/a14v2).
# Nodes 2,3: open-loop eval queue on their own GPUs until nodes 0,1 have finished this item.
G=$(dirname "$(dirname "$(realpath "$0")")")
T=a14v2; ROOT=$G/groot-runs-2x32/$T; OUT=$ROOT/production; K=$(grep -v '^\s*#' "$G/relay/queue.txt" | grep -v '^\s*$' | grep -n "groot_resume_a14v2.sh" | head -1 | cut -d: -f1); K=$((K - 1))
case "${RANK:?DLC RANK missing}" in
  0|1) ;;
  *) E=$G/eval-action-mse; H=$(hostname); T0=$(date +%s)
     echo "$(date -u) groot_resume_a14v2: node $RANK eval queue until nodes 0,1 finish item $K"
     cd "$E" || exit 0
     HB_EVAL_IGNORE_RELAY=1 setsid python3 eval_queue.py --free-min 1 >> "logs/queue-$H-item$K.log" 2>&1 < /dev/null &
     QP=$!
     while ! { [ -f "$G/relay/item-$K-done-node-0" ] && [ -f "$G/relay/item-$K-done-node-1" ]; } \
           && [ $(( $(date +%s) - T0 )) -lt 18000 ]; do sleep 30; done
     kill -TERM "$QP" 2>/dev/null
     pkill -TERM -f 'eval-action-mse/(eval_run|run_eval)' 2>/dev/null; sleep 20
     pkill -KILL -f 'eval-action-mse/(eval_run|run_eval)' 2>/dev/null; kill -KILL "$QP" 2>/dev/null
     for d in "$E"/results/claims/*/; do grep -rqs -- "$H" "$d" && rm -rf "$d"; done
     rm -f "$E/results/RUNNING-$H.json"
     echo "$(date -u) evals stopped on $H; results: $(ls "$E"/results/*.json 2>/dev/null | wc -l)"
     exit 0 ;;
esac
mkdir -p "$ROOT/logs"; GL=$ROOT/logs/guard-node-$RANK.log
log() { echo "$(date -u) $*" | tee -a "$GL"; }
PAT="hb_launch\.py.*groot-data/$T "
gpu_free() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 5; done; return 1; }
last_ck() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | while read s; do
  [ -f "$OUT/checkpoint-$s/trainer_state.json" ] && [ -f "$OUT/checkpoint-$s/scheduler.pt" ] && echo "$s"; done | tail -1; }
seg=0 fails=0
while [ ! -f "$ROOT/production-complete.json" ] && [ $fails -lt 3 ] && [ $seg -lt 6 ]; do
  seg=$((seg + 1)); start=$(last_ck); extra=""
  [ -n "$start" ] && extra="--resume_from_checkpoint"
  gpu_free
  log "node $RANK segment $seg: start from ${start:-scratch} port $((29700 + 2 * seg))"
  HB_GROOT_EXTRA=$extra bash "$G/hb_groot/groot_pair2x32_h2p_r.sh" "$T" none $((29700 + 2 * seg)) 29699 &
  pid=$! stopped=0
  while kill -0 $pid 2>/dev/null; do
    now=$(last_ck)
    if [ $stopped = 0 ] && [ -n "$now" ] && [ "$now" != "${start:-0}" ] && [ $((now % 1000)) = 0 ] && [ "$now" -lt 4000 ]; then
      sleep 60; log "checkpoint-$now complete: stopping to reset host memory"
      pkill -TERM -f "$PAT"; sleep 30; pkill -KILL -f "$PAT"; stopped=1
    fi
    sleep 20
  done
  wait $pid; rc=$?
  log "segment $seg rc=$rc stopped=$stopped last_ck=$(last_ck)"
  [ $stopped = 1 ] || [ $rc = 0 ] || fails=$((fails + 1))
  sleep 30
done
[ -f "$ROOT/production-complete.json" ] || { log "not complete after $seg segments ($fails failures)"; exit 1; }
[ "$RANK" = 0 ] && python3 - "$ROOT" <<'EOF'
import json, sys, pathlib
r = pathlib.Path(sys.argv[1]); p = r / "production-complete.json"; d = json.load(open(p))
d["resume_note"] = ("host-RAM growth (~0.5 GB per step on master-0) in the H2 video data pipeline: the run was stopped "
                    "after checkpoint-2000 and checkpoint-3000 and resumed from them (--resume_from_checkpoint: weights, "
                    "optimizer, scheduler, RNG; GR00T ignore_data_skip); recipe unchanged")
json.dump(d, open(p, "w"), indent=1); print("RESUME_NOTE_ADDED")
EOF
log "complete"
exit 0
