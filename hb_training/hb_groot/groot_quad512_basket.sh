#!/bin/bash
# Relay cmd item (2026-11-13 user): GR00T A20 basket transfer, global batch 512 full fine-tune, 4000 steps, ckpt 2000..4000 every 500.
exec bash "$(dirname "$(realpath "$0")")/groot_quad_ft512.sh" basket 29660
