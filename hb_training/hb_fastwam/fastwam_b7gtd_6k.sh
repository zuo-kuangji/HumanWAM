#!/bin/bash
# Relay cmd item (2026-11-24 user): FastWAM b7gtd (B7 lead-in-dropped data), 4 x 8 x 16 = 512: 4000 steps (ckpt 2000/4000 -> HF
# fastwam/<B7 dir>/b512-leaddrop/), then the 2000-step weights-only continuation (peak lr 3e-5, ckpt 5000/6000 -> b512-leaddrop-cont2k/).
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H"); L=$G/b7d-20261124; T=b7gtd; R=$G/fastwam/runs/$T
for i in $(seq 2160); do [ -f $L/fastwam-$T.ok ] && [ -f $L/cache-$T.ok ] && break; [ -f $L/fastwam-$T.failed ] || [ -f $L/cache-$T.failed ] && break; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for fastwam-$T.ok"; sleep 10; done
[ -f $L/fastwam-$T.ok ] && [ -f $L/cache-$T.ok ] || { echo "SKIP fastwam $T: data or MSE cache not prepared"; exit 3; }
bash $H/fastwam_task.sh $T g1 31890; rc=$?
[ "$rc" = 0 ] || { echo "fastwam $T main run failed rc=$rc"; exit $rc; }
for i in $(seq 60); do [ -f $R/complete.json ] && break; sleep 10; done
[ -f $R/complete.json ] || { echo "fastwam $T: complete.json missing"; exit 70; }
exec bash $H/fastwam_cont.sh $T g1 31900
