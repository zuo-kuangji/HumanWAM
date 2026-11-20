#!/bin/bash
# 2026-11-20: DiT4DiT views of every G1 task on its latest data (CPU only, niced); starts after the pi0.5 prep finished
H=$(dirname "$(realpath "$0")"); G=$(dirname "$H")
until grep -q ALL_DONE $G/pi05-data/prepare-latest-20261120.log; do sleep 60; done
source $H/hb_env.sh; cd $H
for t in a7v4 b5ss b5 a3v4 b8 basketv2 b7t a10v3 b1 b7gt b6v6; do [ -f $HB_D4D_DATA/$t/READY ] && continue; CUDA_VISIBLE_DEVICES= nice -n 19 $HB_ENV/bin/python dit4dit_prepare.py $t --workers 6 || echo "PREP_FAILED $t"; done; echo ALL_DONE
