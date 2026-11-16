#!/bin/bash
# hb_relay cmd item (2026-11-14 user): Delta-flow a17 H2 with 2 cameras (head + chest), 8000 steps, 256 = 32 x 8, no accumulation.
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a17c2
