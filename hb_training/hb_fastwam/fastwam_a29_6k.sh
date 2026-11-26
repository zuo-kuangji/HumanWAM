#!/bin/bash
# Relay cmd item (2026-11-25 user): FastWAM A29 drawer close black-marble 30 Hz (100 ep) on all 4 nodes, 4 x 8 x 16 = 512, the FastWAM recipe of every task:
# 4000 steps (lr 1e-4 cosine, ckpt 2000/4000 -> HF fastwam/a29-drawer-close/b512/), then right away the 2000-step weights-only
# continuation (peak lr 3e-5 = 0.3 x 1e-4, ckpt 5000/6000 -> b512-cont2k/), in one queue item.
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H"); L=$G/a29-20261125; R=$G/fastwam/runs/a29
for i in $(seq 2160); do [ -f $L/fastwam-a29.ok ] && [ -f $L/cache-a29.ok ] && break; [ -f $L/fastwam-a29.failed ] || [ -f $L/cache-a29.failed ] && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for fastwam-a29.ok"; sleep 10; done
[ -f $L/fastwam-a29.ok ] && [ -f $L/cache-a29.ok ] || { echo "SKIP fastwam a29: data or MSE cache not prepared"; exit 3; }
bash $H/fastwam_task.sh a29 g1 31970; rc=$?
[ "$rc" = 0 ] || { echo "fastwam a29 main run failed rc=$rc"; exit $rc; }
for i in $(seq 60); do [ -f $R/complete.json ] && break; sleep 10; done
[ -f $R/complete.json ] || { echo "fastwam a29: complete.json missing"; exit 70; }
exec bash $H/fastwam_cont.sh a29 g1 31980
