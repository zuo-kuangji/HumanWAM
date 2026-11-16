#!/bin/bash
# hb_relay cmd item (2026-11-15, H2 training guide a17 4.3 + user): Delta-flow A17 (H2, head + chest, action75) on the
# configuration of the legacy a18 epoch-040 run (H2PourDataset, b6-visual-aug-v2, no proprio aug), 256 = 32 x 8,
# no accumulation, 8000 steps, checkpoints 4000 / 8000.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a17lg
