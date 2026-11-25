#!/bin/bash
# Relay cmd item (2026-11-25 user): FastWAM B1 v2 (200 ep) on all 4 nodes, 4 x 8 x 16 = 512, the FastWAM recipe of every task:
# 4000 steps (lr 1e-4 cosine, ckpt 2000/4000 -> HF fastwam/b1-riceball-microwave/b512-v2/), then right away the 2000-step
# weights-only continuation (peak lr 3e-5 = 0.3 x 1e-4, ckpt 5000/6000 -> b512-v2-cont2k/), in one queue item so the
# continuation can never be claimed before the main run has finished.
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H"); L=$G/b1v2-20261125; R=$G/fastwam/runs/b1v2
for i in $(seq 2160); do [ -f $L/fastwam-b1v2.ok ] && [ -f $L/cache-b1v2.ok ] && break; [ -f $L/fastwam-b1v2.failed ] || [ -f $L/cache-b1v2.failed ] && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for fastwam-b1v2.ok"; sleep 10; done
[ -f $L/fastwam-b1v2.ok ] && [ -f $L/cache-b1v2.ok ] || { echo "SKIP fastwam b1v2: data or MSE cache not prepared"; exit 3; }
bash $H/fastwam_task.sh b1v2 g1 31930; rc=$?
[ "$rc" = 0 ] || { echo "fastwam b1v2 main run failed rc=$rc"; exit $rc; }
for i in $(seq 60); do [ -f $R/complete.json ] && break; sleep 10; done
[ -f $R/complete.json ] || { echo "fastwam b1v2: complete.json missing"; exit 70; }
exec bash $H/fastwam_cont.sh b1v2 g1 31940
