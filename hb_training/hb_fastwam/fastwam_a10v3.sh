#!/bin/bash
# hb_relay cmd item (2026-11-16 user): FastWAM a10v3 (g1), 4 nodes, largest benchmarked batch, 4000 steps.
exec bash "$(dirname "$(realpath "$0")")/fastwam_task.sh" a10v3 g1 31210
