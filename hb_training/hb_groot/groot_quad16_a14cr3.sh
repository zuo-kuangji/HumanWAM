#!/bin/bash
# Relay cmd item (2026-11-20 user: A14 screw-bulb cropped-v3 = newest A14; GR00T, then Psi-Zero, then FastWAM): GR00T A14 cr3,
# head + chest, action82 (+ planner), full fine-tune, 4 nodes x 8 x 16 = 512, no accumulation, 4000 steps.
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/a14cr3-20261120/groot.ok" ] || { echo "SKIP groot a14cr3: data view not verified"; exit 3; }
exec bash "$G/hb_groot/groot_quad16_h2p.sh" a14cr3 31700
