#!/bin/bash
# Relay cmd item (2026-11-14 user): GR00T 4-node probe without gradient accumulation, 4 x 8 x 16 = 512, basket, 500 steps.
exec bash "$(dirname "$(realpath "$0")")/groot_probe.sh" p4n-16x1 4 16 1 29671
