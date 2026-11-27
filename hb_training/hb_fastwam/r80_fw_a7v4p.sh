#!/bin/bash
# Relay cmd item (2026-11-27 user: newest pause-trim data, big three at the end of the queue; checkpoint rule r80).
G=$(dirname "$(dirname "$(realpath "$0")")")
# FastWAM a7v4p on all 4 nodes, 4 x 8 x 16 = 512: 4000 steps (ckpt 3200 / 4000) then the 2000-step weights-only
# continuation (peak lr 3e-5, ckpt 5600 / 6000), seed 42, in one queue item.
L=$G/newtrim-20261127
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
H=$G/hb_fastwam; R=$G/fastwam/runs/a7v4p
wait_ok $L/cache-a7v4p.ok $L/cache-a7v4p.failed && wait_ok $L/fastwam-a7v4p.ok $L/fastwam-a7v4p.failed || { echo "SKIP fastwam a7v4p"; exit 3; }
bash $H/fastwam_task_r80.sh a7v4p g1 32144; rc=$?
[ "$rc" = 0 ] || { echo "fastwam a7v4p main run failed rc=$rc"; exit $rc; }
for i in $(seq 60); do [ -f $R/complete.json ] && break; sleep 10; done
[ -f $R/complete.json ] || { echo "fastwam a7v4p: complete.json missing"; exit 70; }
exec bash $H/fastwam_cont_r80.sh a7v4p g1 32146
