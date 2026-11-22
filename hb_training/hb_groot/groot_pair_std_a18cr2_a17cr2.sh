#!/bin/bash
# Relay cmd item (2026-11-22 user: H2 aligned with every other task; the h32 variant is dropped): GR00T standard recipe
# (same as every G1 task: full fine-tune from the N1.7 base, horizon 16, default dropout, lr 1e-4, HB aug v2 + A20 proprio)
#   ranks 0,1 -> A18 cropped v2 (HF a18-pour-water-expert-cropped-v2 @44cdd57)   ranks 2,3 -> A17 cropped v2 (@3520b70)
# head + chest, 2 x 8 x 32 = 512, no accumulation, 4000 steps. Data views are the verified 2026-11-15 ones.
G=$(dirname "$(dirname "$(realpath "$0")")"); L=$G/h2cr2-20261115
case "${RANK:?DLC RANK missing}" in
  0|1) [ -f $L/groot-a18cr2.ok ] || { echo "SKIP groot a18cr2: data view not verified"; exit 3; } ;;
  2|3) [ -f $L/groot-a17cr2.ok ] || { echo "SKIP groot a17cr2: data view not verified"; exit 3; } ;;
esac
unset HB_GROOT_MODEL_OVERRIDES HB_GROOT_LR_GROUPS
exec bash "$G/hb_groot/groot_pair2x32_h2.sh" a18cr2 a17cr2 29768 29770
