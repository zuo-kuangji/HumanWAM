#!/bin/bash
# Probe -> production for the G1 tasks, sequentially. Start on ALL 4 nodes with identical args:
#   bash $G/hb_psi0/psi0_chain.sh <chain_id> [port_base=29610] [task ...]   (default: b6 basket a3 a7 a18)
# Re-running with the same chain_id reuses decision.json files and resumes from the latest checkpoint.
# The batch probe runs per task (hence per robot layout: G1 A76/S44 vs H2 A75/S78, same 2 cameras).
set -euo pipefail
H=$(dirname "$(realpath "$0")")
CHAIN=${1:?chain id}
shift
PB=${1:-29610}
if [ $# -gt 0 ]; then shift; fi
TASKS=("$@")
[ "${#TASKS[@]}" -gt 0 ] || TASKS=(b6 basket a3 a7 a18)
i=0
for t in "${TASKS[@]}"; do
  echo "=== $(date -u) task $t chain $CHAIN port $((PB + i))"
  bash "$H/psi0_node.sh" "$t" "$CHAIN" "$((PB + i))"
  i=$((i + 1))
done
echo "CHAIN_DONE $CHAIN ${TASKS[*]}"
