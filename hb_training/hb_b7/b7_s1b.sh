#!/bin/bash
# B7 stage 1b (2026-11-11, user, after the 17:34 stage-1 failure): nodes 0-2 = pi0.5 B7 on 3 x 8 x 29 = 696 (b7_s1.sh);
# node 3 = ACT B7 single node 8 x 87 = 696 (act_single.sh, GPU trial first). Replaces the later 3-node ACT stage (b7_s2).
G=$(dirname "$(dirname "$(realpath "$0")")")
NODE=${RANK:?DLC RANK missing}
if [ "$NODE" = 3 ]; then
  for i in $(seq 36); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
  exec bash "$G/hb_act/act_single.sh" b7 87 "$G/act-runs-b696"
fi
exec bash "$G/hb_b7/b7_s1.sh"
