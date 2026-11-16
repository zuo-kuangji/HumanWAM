#!/bin/bash
# Relay cmd item (2026-11-15 user): GR00T A10 trolley push v3 (all 148 episodes) on nodes 0,1: 2 x 8 x 32 = 512, no
# accumulation, 4000 steps (groot_pair2x32.sh). Nodes 2,3 have no GR00T partner: they run the open-loop eval queue
# (eval-action-mse, idle-slot mode) until nodes 0,1 finish this item, then stop it and exit 0.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/a10v3-20261115/groot.ok" ] || { echo "SKIP groot a10v3: data view not verified"; exit 3; }
K=$(grep -v '^\s*#' "$G/relay/queue.txt" | grep -v '^\s*$' | grep -n "groot_solo_a10v3.sh" | head -1 | cut -d: -f1); K=$((K - 1))
case "${RANK:?DLC RANK missing}" in
  0|1) exec bash "$G/hb_groot/groot_pair2x32.sh" a10v3 none 29720 29722 ;;
esac
E=$G/eval-action-mse; H=$(hostname); T0=$(date +%s)
echo "$(date -u) groot_solo_a10v3: node $RANK eval queue until nodes 0,1 finish item $K"
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
exit 0
