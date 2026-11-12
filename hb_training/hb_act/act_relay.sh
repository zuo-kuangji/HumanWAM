#!/bin/bash
# hb_relay cmd item helper: ACT 20-epoch production (save at epoch 10 and 20) for ONE task on all 4 nodes.
H=$(dirname "$(realpath "$0")")
T=${1:?usage: act_relay.sh <task>}
case $T in b6) i=0 ;; a3) i=1 ;; a7) i=2 ;; a18) i=3 ;; trolley) i=4 ;; basket) i=5 ;; a17) i=6 ;; *) echo "unknown task $T"; exit 64 ;; esac
exec bash "$H/act_node.sh" "$T" hb1110 $((34100 + 10 * i))
