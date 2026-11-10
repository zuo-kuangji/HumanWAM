#!/bin/bash
# hb_relay "cmd" item helper: DiT4DiT 20-epoch production (save at epoch 10 and 20) for ONE task on all 4 nodes.
# The relay passes no arguments, so the queue uses the per-task wrappers dit4dit_<task>.sh, which call this.
# Same chain id for all tasks (hb1110): the per-layout batch probe (g1 / h2) is reused across tasks.
H=$(dirname "$(realpath "$0")")
T=${1:?usage: dit4dit_relay.sh <task>}
case $T in b6) i=0 ;; a3) i=1 ;; a7) i=2 ;; a18) i=3 ;; *) echo "unknown task $T"; exit 64 ;; esac
# user rule 2026-11-10: per-GPU batch as large as memory allows (peak <= HB_PROBE_MAX_FRAC 0.92); the chain's default
# list started at 24, so start higher.
export HB_PROBE_CANDIDATES=${HB_PROBE_CANDIDATES:-"64 48 40 32 24 16 12 8 6 4 2"}
exec bash "$H/dit4dit_chain.sh" hb1110 $((31100 + 100 * i)) "$T"
