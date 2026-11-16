#!/bin/bash
# Relay cmd item (2026-11-13 user): GR00T C6 (cross-scene, long episodes), global batch 512 full fine-tune, 8000 steps (2x normal), ckpt 4000..8000 every 500.
export HB_GROOT_STEPS=8000
exec bash "$(dirname "$(realpath "$0")")/groot_quad_ft512.sh" c6 29663
