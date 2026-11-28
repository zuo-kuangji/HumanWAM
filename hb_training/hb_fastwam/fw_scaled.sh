#!/bin/bash
# Relay cmd helper (2026-11-28 user): FastWAM in ONE run (no 4000 + 2000 continuation), steps scaled by the mean episode
# length vs A20 basket v2 (544.33 frames; FastWAM base 6000), ceil to 1000; native peak lr (1e-4 cosine), ckpt ~80 % + last.
# 4 nodes x 8 x 16 = 512, seed 42, same recipe as fastwam_task_r80.sh otherwise.
#   fw_scaled.sh <task> <profile g1|h2|h2p> <steps> <save> <port>
# G1 tasks must train on commanded hands (11-27 user): they wait for $G/fastwam/data/<task>/meta/hb_cmd_hands.ok.
G=$(dirname "$(dirname "$(realpath "$0")")")
TASK=$1 PROF=$2 STEPS=$3 SAVE=$4 PORT=$5
L=$G/newtask-20261128; grep -q "^$TASK " $L/tasks.txt 2>/dev/null || L=$G/newtrim-20261127
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
wait_ok $L/cache-$TASK.ok $L/cache-$TASK.failed && wait_ok $L/fastwam-$TASK.ok $L/fastwam-$TASK.failed || { echo "SKIP fastwam $TASK"; exit 3; }
if [ "$PROF" = g1 ]; then
  wait_ok $G/fastwam/data/$TASK/meta/hb_cmd_hands.ok $G/fastwam/data/$TASK/meta/hb_cmd_hands.failed || { echo "SKIP fastwam $TASK: G1 data not on commanded hands"; exit 3; }
fi
export HB_STEPS=$STEPS HB_SAVE=$SAVE
exec bash $G/hb_fastwam/fastwam_task_scaled.sh "$TASK" "$PROF" "$PORT"
