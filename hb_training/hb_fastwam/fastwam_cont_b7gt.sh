#!/bin/bash
# hb_relay cmd item (2026-11-22 user): FastWAM b7gt (B7 green trimmed v2, 421 episodes) continuation, 2000 more steps from step_004000 (peak lr 3e-5), aligned with the other tasks at 6000 total
exec bash "$(dirname "$(realpath "$0")")/fastwam_cont.sh" b7gt g1 31900
