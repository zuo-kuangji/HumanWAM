#!/bin/bash
# Relay cmd item (2026-11-26 user: "B5 standing start can be trained now, all models, at the END of the queue"): FastWAM
# B5 standing start (HF A22-chair-push-standing-start-expert-v1-formerly-b5, data @913f5a1 = head dd339aa8 (card-only rename), 100 ep;
# FastWAM data view + in-loop MSE cache prepared 11-16), 4 x 8 x 16 = 512, 4000 steps (ckpt 2000/4000 ->
# HF fastwam/b5-chair-push-standing-start/b512/), then the 2000-step weights-only continuation (peak lr 3e-5 = 0.3 x 1e-4,
# ckpt 5000/6000 -> b512-cont2k/), in one queue item.
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H"); R=$G/fastwam/runs/b5ss
[ -f $G/fastwam/data/b5ss/meta/hb_fastwam_interface.json ] && [ -f $G/eval-action-mse/cache/b5ss/train10/meta.json ] || { echo "SKIP fastwam b5ss: data or MSE cache missing"; exit 3; }
bash $H/fastwam_task.sh b5ss g1 32010; rc=$?
[ "$rc" = 0 ] || { echo "fastwam b5ss main run failed rc=$rc"; exit $rc; }
for i in $(seq 60); do [ -f $R/complete.json ] && break; sleep 10; done
[ -f $R/complete.json ] || { echo "fastwam b5ss: complete.json missing"; exit 70; }
exec bash $H/fastwam_cont.sh b5ss g1 32020
