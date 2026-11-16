#!/bin/bash
# Relay cmd item (2026-11-15 17:xx user: B6 v6 continuations first, then B5, then A3 / A7, per model):
#   ranks 0,1 -> GR00T B6 v6 continuation (groot_cont_b6v6.sh, 512, 4000 steps, peak lr 3e-5)
#   ranks 2,3 -> GR00T B5 chair push v1 (HF b5-chair-push-expert-v1 @b85b7e9, all 100 episodes) from the GR00T N1.7
#                base: 2 x 8 x 32 = 512, no accumulation, 4000 steps, lr 1e-4 (groot_pair2x32.sh)
# Each pair waits (up to 2 h) until its data view is verified; skips (exit 3) when the preparation failed.
G=$(dirname "$(dirname "$(realpath "$0")")")
wait_ok() {  # $1 = verified marker, $2 = failure marker
  for i in $(seq 720); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done
  return 1
}
case "${RANK:?DLC RANK missing}" in
  0|1) wait_ok "$G/b6v6-20261115/groot-b6v6.ok" "$G/b6v6-20261115/groot-b6v6.failed" || { echo "SKIP groot b6v6: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_cont_b6v6.sh" ;;
  2|3) wait_ok "$G/b6v6-20261115/groot-b5.ok" "$G/b6v6-20261115/groot-b5.failed" || { echo "SKIP groot b5: data view not verified"; exit 3; }
       exec bash "$G/hb_groot/groot_pair2x32.sh" none b5 29752 29754 ;;
esac
