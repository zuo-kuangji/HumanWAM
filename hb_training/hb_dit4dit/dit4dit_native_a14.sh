#!/bin/bash
# 2026-11-13 user: long tasks (C-class cross-scene C6, and A14) train 2x the basket step budget of this model.
export HB_FIXED_STEPS=16000
# 2026-11-13: H2 tasks train with head + chest + wrist (user rule); skipped until the 3-camera adapter of this model
# is verified and $G/control/h2-3cam-ready-dit4dit-a14 exists.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-dit4dit-a14" ] || { echo "SKIP dit4dit a14: 3-camera adapter not ready"; exit 3; }
# hb_relay cmd item: DiT4DiT A14 (H2, head + chest + wrist), native state encoding, 4000 steps, global batch 64
exec bash "$(dirname "$(realpath "$0")")/dit4dit_native_relay.sh" a14
