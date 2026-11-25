#!/bin/bash
# Relay cmd item (2026-11-25 user): FastWAM B6 v6 on ALL 542 episodes (B6 v6 rule since 11-24; the published fastwam b6v6
# used episodes 0-499), 4 x 8 x 16 = 512, seed 42: 4000 steps (ckpt 2000/4000 -> HF fastwam/b6-pedal-bin/b512-v6-pure/),
# then the 2000-step weights-only continuation (peak lr 3e-5, ckpt 5000/6000 -> b512-v6-pure-cont2k/).
# Data fastwam/data/b6v6p -> b6v6.all542-20261118 (542 episodes, 331,577 frames); in-training MSE on eval task b6v6 (all 542).
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H"); T=b6v6p; R=$G/fastwam/runs/$T
bash $H/fastwam_task.sh $T g1 31910; rc=$?
[ "$rc" = 0 ] || { echo "fastwam $T main run failed rc=$rc"; exit $rc; }
for i in $(seq 60); do [ -f $R/complete.json ] && break; sleep 10; done
[ -f $R/complete.json ] || { echo "fastwam $T: complete.json missing"; exit 70; }
exec bash $H/fastwam_cont.sh $T g1 31920
