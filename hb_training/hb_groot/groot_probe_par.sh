#!/bin/bash
# Relay cmd item (2026-11-14 user): four single-node GR00T batch probes on basket in parallel, one per node (500 steps).
H=$(dirname "$(realpath "$0")")
case "${RANK:?}" in
  0) exec bash "$H/groot_probe.sh" p1n-32x2 1 32 2 ;;
  1) exec bash "$H/groot_probe.sh" p1n-8x8 1 8 8 ;;
  2) exec bash "$H/groot_probe.sh" p1n-16x4 1 16 4 ;;
  3) exec bash "$H/groot_probe.sh" p1n-8x4 1 8 4 ;;
esac
