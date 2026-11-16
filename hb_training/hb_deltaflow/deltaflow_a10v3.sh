#!/bin/bash
# hb_relay cmd item (2026-11-15 user): Delta-flow A10 trolley push v3, ALL 148 episodes (code tree HumanWAM-allep),
# 8000 steps, 256 = 32 x 8, no accumulation.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a10v3
