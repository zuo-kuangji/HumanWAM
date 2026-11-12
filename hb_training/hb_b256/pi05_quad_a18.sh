#!/bin/bash
# 2026-11-13: H2 tasks train with head + chest + wrist (user rule); skipped until the 3-camera adapter of this model
# is verified and $G/control/h2-3cam-ready-b256-a18 exists.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-b256-a18" ] || { echo "SKIP b256 a18: 3-camera adapter not ready (control/h2-3cam-ready-b256-a18 missing)"; exit 3; }
# Relay cmd item (2026-11-12, aligned re-train, always at the queue tail): pi0.5 a18 on all 4 nodes, 4 x 8 x 8 = 256, 20 epochs,
# GR00T-A20-aligned augmentation (no proprio delay); HF pi05/<task>/b256/.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b256/pi05_multi.sh" a18 4 "$RANK" "$MASTER_ADDR" 30722 8
