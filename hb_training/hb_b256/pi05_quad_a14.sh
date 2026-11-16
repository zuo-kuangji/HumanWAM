#!/bin/bash
# 2026-11-13 user: long tasks (C-class cross-scene C6, and A14) train 2x the basket step budget of this model.
export HB_FIXED_STEPS=8000
# 2026-11-13: H2 tasks train with head + chest + wrist (user rule); skipped until the 3-camera adapter of this model
# is verified and $G/control/h2-3cam-ready-b256-a14 exists.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-b256-a14" ] || { echo "SKIP b256 a14: 3-camera adapter not ready"; exit 3; }
# Relay cmd item (2026-11-13): pi0.5 A14 (H2, head + chest + wrist) on all 4 nodes, 4 x 8 x 8 = 256, 4000 steps.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b256/pi05_multi.sh" a14 4 "$RANK" "$MASTER_ADDR" 30727 8
