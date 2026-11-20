#!/bin/bash
# hb_relay "cmd" item helper (2026-11-10 evening): DiT4DiT 20-epoch production with the NATIVE state encoding
# (sin/cos on joint angles: G1 73, H2 153; benchmark input stays raw 44 / 78 without velocity). New chain id hb1110n,
# so the abandoned 44-D chain hb1110 stays untouched. Per-task wrappers: dit4dit_native_<task>.sh.
# Batch probe: G1 per-GPU 4 already OOMs at 224x448 two-camera frames (chain hb1110), so G1 tries 3 then 2;
# H2 is probed on its own (4 3 2). Peak limit 0.92 as for every model.
H=$(dirname "$(realpath "$0")")
T=${1:?usage: dit4dit_native_relay.sh <task>}
case $T in b6) i=0 ;; a3) i=1 ;; a7) i=2 ;; a18) i=3 ;; trolley) i=4 ;; basket) i=5 ;; a17) i=6 ;; b7) i=7 ;; a6) i=8 ;; c6) i=9 ;; a14) i=10 ;;
  a3v4) i=11 ;; a7v4) i=12 ;; a10v3) i=13 ;; basketv2) i=14 ;; b1) i=15 ;; b5) i=16 ;; b5ss) i=17 ;; b6v6) i=18 ;; b7t) i=19 ;; b7gt) i=20 ;; b8) i=21 ;;  # 2026-11-20 user: latest data of every G1 task
  *) echo "unknown task $T"; exit 64 ;; esac
# 2026-11-12 user: DiT4DiT global batch 64 = 32 GPUs x 2 for every task
C="2"
# 2026-11-13 user: H2 tasks have 3 cameras (224x672 frames) -> per-GPU 1 x gradient accumulation 2 (global 64)
case $T in a18|a17|a14) export HB_BATCH_H2=1 HB_D4D_ACCUM=2 ;; esac
export HB_PROBE_CANDIDATES=${HB_PROBE_CANDIDATES:-$C}
exec bash "$H/dit4dit_chain.sh" hb1110n $((33100 + 100 * i)) "$T"
