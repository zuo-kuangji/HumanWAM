#!/bin/bash
# Relay cmd item (2026-11-14 18:2x user): GR00T A18 (nodes 0,1) + A17 (nodes 2,3), H2 2 cameras head + chest,
# 512 = 2 nodes x 8 x 32 each, no accumulation. Fallback if per-GPU 32 does not fit: groot_quad16_h2.sh (4 x 8 x 16).
exec bash "$(dirname "$(realpath "$0")")/groot_pair2x32_h2.sh" a18c2 a17c2 29688 29690
