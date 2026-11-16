#!/bin/bash
# Relay cmd item (2026-11-14 H2 training guide a14 4.1): GR00T A14 v2, head + chest, action82 (+ planner), full fine-tune,
# 2 x 8 x 32 = 512, no accumulation, 4000 steps, on nodes 0,1.
# Nodes 2,3 have no GR00T partner: instead of idling they run the open-loop action eval queue (eval-action-mse) on
# their own GPUs until both training nodes have finished this item, then stop their evals, release their claims, exit 0.
# Guard: skips (exit 3, nothing started) unless the a14v2 data view was CPU-verified.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/a14v2-20261114/groot.ok" ] || { echo "SKIP groot a14v2: data view not verified"; exit 3; }
case "${RANK:?DLC RANK missing}" in
  0|1) exec bash "$G/hb_groot/groot_pair2x32_h2p.sh" a14v2 none 29696 29698 ;;
esac
E=$G/eval-action-mse; H=$(hostname); T0=$(date +%s)
echo "$(date -u) groot_solo_a14v2: node $RANK has no partner task; eval-action-mse on this node until nodes 0,1 finish"
cd "$E" || exit 0
setsid python3 eval_queue.py --free-min 1 >> "logs/queue-$H-item59.log" 2>&1 < /dev/null &
QP=$!
while ! { [ -f "$G/relay/item-59-done-node-0" ] && [ -f "$G/relay/item-59-done-node-1" ]; } \
      && [ $(( $(date +%s) - T0 )) -lt 18000 ]; do sleep 30; done
echo "$(date -u) training nodes done (or 5 h cap): stopping evals on $H"
kill -TERM "$QP" 2>/dev/null
pkill -TERM -f 'eval-action-mse/(eval_run|run_eval)' 2>/dev/null; sleep 20
pkill -KILL -f 'eval-action-mse/(eval_run|run_eval)' 2>/dev/null; kill -KILL "$QP" 2>/dev/null
for d in "$E"/results/claims/*/; do grep -rqs -- "$H" "$d" && rm -rf "$d"; done
rm -f "$E/results/RUNNING-$H.json"
echo "$(date -u) evals stopped on $H; results so far: $(ls "$E"/results/*.json 2>/dev/null | wc -l)"
exit 0
