#!/bin/bash
# Relay cmd item (2026-11-22 user: GR00T on the newest data missing on HF):
#   ranks 0,1 -> GR00T B7 black trimmed (HF b7-black-clothes-hamper @949da46, first 100 of 107)
#   ranks 2,3 -> GR00T B7 green trimmed v2 (HF b7-green-clothes-hamper @22e3b83, all 421)
# each 2 x 8 x 32 = 512, no accumulation, full fine-tune from the GR00T N1.7 base, 4000 steps (groot_pair2x32.sh).
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/gp-20261122
wait_ok() { for i in $(seq 2160); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done; return 1; }
case "${RANK:?DLC RANK missing}" in
  0|1) wait_ok $L/groot-b7t.ok $L/groot-b7t.failed || { echo "SKIP groot b7t: data view not verified"; exit 3; } ;;
  2|3) wait_ok $L/groot-b7gt.ok $L/groot-b7gt.failed || { echo "SKIP groot b7gt: data view not verified"; exit 3; } ;;
esac
exec bash "$G/hb_groot/groot_pair2x32.sh" b7t b7gt 29760 29762
