#!/bin/bash
# hb_relay cmd item (2026-11-15 23:4x user): Delta Flow A3 v4 (all 125 episodes, tree HumanWAM-allep), 8000
# steps, 256 = 32 x 8, no accumulation.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a3v4
