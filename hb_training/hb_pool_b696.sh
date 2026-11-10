#!/bin/bash
# hb_relay cmd item: GR00T + pi0.5 retrain, global batch 696 (87 x 8), one node per job (task pool b696)
exec bash "$(dirname "$(realpath "$0")")/hb_pool.sh" b696
