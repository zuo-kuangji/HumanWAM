#!/bin/bash
# Relay cmd item (2026-11-28 user: steps scaled by episode length vs A20 basket v2, ceil to 1000; ckpt ~80 % + last).
G=$(dirname "$(dirname "$(realpath "$0")")"); exec bash $G/hb_fastwam/fw_scaled.sh a19 h2 16000 12800 32152
