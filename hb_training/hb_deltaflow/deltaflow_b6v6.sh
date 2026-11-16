#!/bin/bash
# hb_relay cmd item (2026-11-15 user): Delta Flow b6v6, 8000 steps, 256 = 32 x 8, no accumulation (all episodes, tree
# HumanWAM-allep). Weights-only continuation of deltaflow-runs/b6 step_010200, peak lr 3e-5.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" b6v6
