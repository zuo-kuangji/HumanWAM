#!/bin/bash
# 2026-11-13: H2 tasks train with head + chest + wrist (user rule); skipped until the 3-camera adapter of this model
# is verified and $G/control/h2-3cam-ready-dit4dit-a17 exists.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-dit4dit-a17" ] || { echo "SKIP dit4dit a17: 3-camera adapter not ready (control/h2-3cam-ready-dit4dit-a17 missing)"; exit 3; }
# hb_relay cmd item: DiT4DiT a17, native state encoding (20 epochs, save at 10/20), chain hb1110n, global batch 64
exec bash "$(dirname "$(realpath "$0")")/dit4dit_native_relay.sh" a17
