#!/bin/bash
# hb_relay "cmd" item helper (2026-11-10 evening): DiT4DiT 20-epoch production with the NATIVE state encoding
# (sin/cos on joint angles: G1 73, H2 153; benchmark input stays raw 44 / 78 without velocity). New chain id hb1110n,
# so the abandoned 44-D chain hb1110 stays untouched. Per-task wrappers: dit4dit_native_<task>.sh.
# Batch probe: G1 per-GPU 4 already OOMs at 224x448 two-camera frames (chain hb1110), so G1 tries 3 then 2;
# H2 is probed on its own (4 3 2). Peak limit 0.92 as for every model.
H=$(dirname "$(realpath "$0")")
T=${1:?usage: dit4dit_native_relay.sh <task>}
case $T in b6) i=0 ;; a3) i=1 ;; a7) i=2 ;; a18) i=3 ;; *) echo "unknown task $T"; exit 64 ;; esac
case $T in a18) C="4 3 2" ;; *) C="3 2" ;; esac
export HB_PROBE_CANDIDATES=${HB_PROBE_CANDIDATES:-$C}
exec bash "$H/dit4dit_chain.sh" hb1110n $((33100 + 100 * i)) "$T"
