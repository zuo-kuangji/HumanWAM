#!/bin/bash
# hb_relay cmd item (2026-11-14 user): Delta-flow b1 rice ball microwave, ALL 115 episodes (code tree HumanWAM-allep),
# 8000 steps, 256 = 32 x 8, no accumulation.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" b1
