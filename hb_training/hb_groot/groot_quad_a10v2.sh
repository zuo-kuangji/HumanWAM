#!/bin/bash
# Relay cmd item (2026-11-13): GR00T A10 v2 (150 episodes) full fine-tune on all 4 nodes (4 x 8 x 8 = 256, 4000 steps).
exec bash "$(dirname "$(realpath "$0")")/groot_quad_ft.sh" a10v2 8 29650
