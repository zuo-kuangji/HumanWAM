#!/bin/bash
# hb_relay cmd item (2026-11-15, H2 training guide a14 4.3 + user): Delta-flow A14 v2 (H2, head + chest, action82 =
# body31 + hands44 + planner7; code tree HumanWAM-h2p) on the legacy a18 epoch-040 configuration, 256 = 32 x 8,
# no accumulation, 8000 steps, checkpoints 4000 / 8000.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a14lg
