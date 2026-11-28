#!/bin/bash
# Relay cmd item, NEW tasks A15 / A19 (2026-11-28 12:xx user: new tasks go AFTER the older waiting items; steps scaled by the
# mean episode length vs A20 basket v2 = 544.33 frames, ceil to 1000; ckpt last + ~80 %).
#   ranks 0,1: Psi-Zero a19 11000 steps (ckpt 8800 + 11000)                                        ~11.5 h
#   ranks 2,3: GR00T a15 3000 (2400 + 3000) -> Psi-Zero a15 3000 (2400 + 3000) -> GR00T a19 11000 (8800 + 11000)  ~8.7 h
G=$(dirname "$(dirname "$(realpath "$0")")")
RK=${RANK:?}; log() { echo "$(date -u) lane-new rank $RK $*"; }
free() { for i in $(seq 360); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0
  [ $((i % 30)) = 1 ] && log "waiting for free GPUs"; sleep 10; done; }
case $RK in
  0|1) free; log "START psi0 a19 11000"
       bash $G/hb_groot/gp_scaled_mix.sh psi0:a19:11000:8800 none 32144 0; log "END psi0 a19 rc=$?" ;;
  2|3) free; log "START grooth2 a15 3000"
       bash $G/hb_groot/gp_scaled_mix.sh none grooth2:a15:3000:2400 0 32140; log "END grooth2 a15 rc=$?"
       free; log "START psi0 a15 3000"
       bash $G/hb_groot/gp_scaled_mix.sh none psi0:a15:3000:2400 0 32142; log "END psi0 a15 rc=$?"
       free; log "START grooth2 a19 11000"
       bash $G/hb_groot/gp_scaled_mix.sh none grooth2:a19:11000:8800 0 32146; log "END grooth2 a19 rc=$?" ;;
esac
exit 0
