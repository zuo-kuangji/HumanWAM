#!/bin/bash
# hb_relay cmd item (2026-11-20 22:3x user): FastWAM a18cr2 continuation, 2000 more steps from step_004000 (peak lr 3e-5); added after the const-dim nMSE re-eval
exec bash "$(dirname "$(realpath "$0")")/fastwam_cont.sh" a18cr2 h2 31880
