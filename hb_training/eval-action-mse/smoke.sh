#!/bin/bash
# CPU smoke test: every adapter, 1 episode x 2 queries of the B6 held-out cache.
E=$G/eval-action-mse; mkdir -p $E/smoke
run() { OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= bash $E/run_eval.sh "$1" b6 heldout "$2" $E/smoke/$1.json --device cpu --max-episodes 1 --max-queries 2 > $E/smoke/$1.log 2>&1; echo "rc=$?" >> $E/smoke/$1.log; }
run groot $G/groot-runs-ft4k/b6/production/checkpoint-4000 &
run pi05 $G/pi05-runs/b696/b6/run/ckpt/pi05_hb_g1_b6/run/1872 &
run psi0 $G/psi0-runs/b6/aligned/train/finetune/hb-b6.hb.flow1000.cosine.lr1.0e-04.b512.gpus32.aligned/checkpoints/ckpt_4000 &
run act $G/act-runs-b696/b6/train/act-g1/hb-act-b6.g1.cosine.lr1.0e-04.b696.gpus8.b696/checkpoints/ckpt_1871 &
run deltaflow $G/deltaflow-runs/b6/run/production/checkpoints/weights/step_010200.pt &
wait
