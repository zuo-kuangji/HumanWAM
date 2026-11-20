#!/bin/bash
# hb_relay cmd item (2026-11-20 user: A14 screw-bulb cropped-v3 = newest A14; GR00T, then Psi-Zero, then FastWAM): FastWAM a14cr3 (h2p = head + chest, action82), 4 nodes, largest benchmarked batch, 4000 steps (A14 v2 rule: no 2x).
G=$(dirname "$(dirname "$(realpath "$0")")")
[ -f "$G/a14cr3-20261120/fastwam.ok" ] || { echo "SKIP fastwam a14cr3: data not verified"; exit 3; }
exec bash "$(dirname "$(realpath "$0")")/fastwam_task.sh" a14cr3 h2p 31740
