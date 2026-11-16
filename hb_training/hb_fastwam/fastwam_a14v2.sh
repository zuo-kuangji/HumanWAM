#!/bin/bash
# hb_relay cmd item (2026-11-16 user): FastWAM a14v2 (h2p), 4 nodes, largest benchmarked batch, 4000 steps.
exec bash "$(dirname "$(realpath "$0")")/fastwam_task.sh" a14v2 h2p 31310
