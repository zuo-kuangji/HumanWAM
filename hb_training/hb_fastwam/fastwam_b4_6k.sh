#!/bin/bash
# Relay cmd item (2026-11-25 user): FastWAM B4 cushion (302 ep) on all 4 nodes, 4 x 8 x 16 = 512, the FastWAM recipe of every task:
# 4000 steps (lr 1e-4 cosine, ckpt 2000/4000 -> HF fastwam/b4-cushion/b512/), then right away the 2000-step weights-only
# continuation (peak lr 3e-5 = 0.3 x 1e-4, ckpt 5000/6000 -> b512-cont2k/), in one queue item.
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H"); L=$G/b4-20261125; R=$G/fastwam/runs/b4
for i in $(seq 2160); do [ -f $L/fastwam-b4.ok ] && [ -f $L/cache-b4.ok ] && break; [ -f $L/fastwam-b4.failed ] || [ -f $L/cache-b4.failed ] && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for fastwam-b4.ok"; sleep 10; done
[ -f $L/fastwam-b4.ok ] && [ -f $L/cache-b4.ok ] || { echo "SKIP fastwam b4: data or MSE cache not prepared"; exit 3; }
bash $H/fastwam_task.sh b4 g1 31950; rc=$?
[ "$rc" = 0 ] || { echo "fastwam b4 main run failed rc=$rc"; exit $rc; }
for i in $(seq 60); do [ -f $R/complete.json ] && break; sleep 10; done
[ -f $R/complete.json ] || { echo "fastwam b4: complete.json missing"; exit 70; }
exec bash $H/fastwam_cont.sh b4 g1 31960
