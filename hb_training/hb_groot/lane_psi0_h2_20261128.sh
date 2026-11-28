#!/bin/bash
# 2026-11-28 02:1xZ (4-hour GPU rule, ranks 2,3 idle since GR00T a7v4p ended 01:12Z): Psi-Zero H2 halves of r80_p2 (which
# this replaces in gq/queue.txt): ranks 2,3 psi0 a18cr4 at once; ranks 0,1 psi0 a17cr4 after side01's psi0 b1v2p ends.
# Same helper, recipe, seed 292285, layout 2 x 8 x 32, checkpoints 3200 + 4000 as r80_p2. H2 action = commanded joints.
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/newtrim-20261127/side01
RK=${RANK:?}; log() { echo "$(date -u) lane_psi0_h2 rank $RK $*"; }
case $RK in 0|1) for i in $(seq 4320); do [ -f $L/psi0-b1v2p.rc-node-$RK ] && break
  [ $((i % 60)) = 1 ] && log "waiting for side01 psi0 b1v2p"; sleep 10; done ;; esac
for i in $(seq 360); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break
  [ $((i % 30)) = 1 ] && log "waiting for free GPUs"; sleep 10; done
log "START"
exec bash $G/hb_groot/gp_r80_mix.sh psi0:a17cr4 psi0:a18cr4 32128 32130
