#!/bin/bash
# hb_relay cmd item (2026-11-14 user: every H2 task 2 cameras): Delta-flow a14 H2 with head + chest, 16000 steps (C6/A14 2x),
# 256 = 32 x 8, no accumulation. Replaces the 3-camera deltaflow_a14.sh (per-GPU 4 x accumulation 2).
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a14c2
