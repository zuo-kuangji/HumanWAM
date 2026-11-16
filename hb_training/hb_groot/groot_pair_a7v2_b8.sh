#!/bin/bash
# Relay cmd item (2026-11-14 user): GR00T A7 v2 (nodes 0,1) + B8 (nodes 2,3), 2 x (2 x 8 x 32) = 512 each.
exec bash "$(dirname "$(realpath "$0")")/groot_pair2x32.sh" a7v2 b8 29684 29686
