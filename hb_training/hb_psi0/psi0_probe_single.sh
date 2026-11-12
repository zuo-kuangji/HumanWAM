#!/bin/bash
# 2026-11-11: pool job "groot psi0probe:<task>" (dispatched by groot_single.sh): run the Psi-Zero batch probe for <task>
# on this node while the pool would otherwise leave it idle. Writes psi0-runs/<task>/hb1110/decision.json; the later
# 4-node relay item "psi0 <task>" then reuses it ("reusing decision.json") instead of probing with 3 nodes waiting.
T=${1:?task}
H=$(dirname "$(dirname "$(realpath "$0")")")
export RANK=0 WORLD_SIZE=4 MASTER_ADDR=127.0.0.1 HB_PROBE_ONLY=1
exec bash "$H/hb_psi0/psi0_node.sh" "$T" hb1110 39990
