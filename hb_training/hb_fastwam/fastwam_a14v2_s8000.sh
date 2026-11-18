#!/bin/bash
# hb_relay cmd item (2026-11-18 11:xx user): FastWAM A14 v2 (H2, head + chest) with the long-task rule (C6 / A14 = 2x
# steps), 8000 steps, lr warmup / cosine over 8000. Replaces fastwam_a14v2.sh (4000 steps).
exec bash "$(dirname "$(realpath "$0")")/fastwam_task_s.sh" a14v2 h2p 31311 8000
