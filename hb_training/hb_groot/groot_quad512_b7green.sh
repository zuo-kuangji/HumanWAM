#!/bin/bash
# Relay cmd item (2026-11-13 user): GR00T B7 green clothes -> hamper, global batch 512 full fine-tune, 4000 steps, ckpt 2000..4000 every 500.
[ -f "$(dirname "$(dirname "$(realpath "$0")")")/groot-data/b7green/hb-dataset-receipt.json" ] || { echo "SKIP: groot-data/b7green not prepared"; exit 3; }
exec bash "$(dirname "$(realpath "$0")")/groot_quad_ft512.sh" b7green 29662
