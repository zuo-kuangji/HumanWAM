#!/bin/bash
# hb_relay cmd item (2026-11-15 user): Delta Flow b5, 8000 steps, 256 = 32 x 8, no accumulation (all episodes, tree
# HumanWAM-allep).
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" b5
