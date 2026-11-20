#!/bin/bash
# 2026-11-20: pi0.5 views of every G1 task on its latest data (CPU only, niced; training runs on the GPUs)
H=$(dirname "$(realpath "$0")"); source $H/hb_env.sh; cd $H
for t in a7v4 b5ss b5 a3v4 b8 basketv2 b7t a10v3 b1 b7gt b6v6; do [ -f $HB_PI05_DATA/$t/READY ] && continue; CUDA_VISIBLE_DEVICES= nice -n 19 $HB_ENV/bin/python hb_prepare.py $t --workers 6 || echo "PREP_FAILED $t"; done; echo ALL_DONE
