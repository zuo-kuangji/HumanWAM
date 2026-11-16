#!/bin/bash
# Relay cmd item (2026-11-14 user): GR00T B1 rice ball microwave (all 115 episodes) on nodes 0,1: 2 x 8 x 32 = 512,
# no accumulation, 4000 steps. No GR00T partner task for nodes 2,3 (A14 on hold): they exit at once and wait at the
# relay barrier; the eval-action-mse queue takes idle GPUs after 10 min.
case "${RANK:?DLC RANK missing}" in
  0|1) exec bash "$(dirname "$(realpath "$0")")/groot_pair2x32.sh" b1 none 29692 29694 ;;
  *) echo "$(date -u) groot_solo_b1: node $RANK has no partner task, exiting"; exit 0 ;;
esac
