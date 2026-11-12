#!/bin/bash
# 2026-11-13: H2 tasks train with head + chest + wrist (user rule); skipped until the 3-camera adapter of this model
# is verified and $G/control/h2-3cam-ready-b512-a17 exists.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-b512-a17" ] || { echo "SKIP b512 a17: 3-camera adapter not ready (control/h2-3cam-ready-b512-a17 missing)"; exit 3; }
# Relay cmd item (2026-11-12, aligned re-train, always at the queue tail): Psi-Zero a17 on all 4 nodes, 4 x 8 x 16 = 512,
# 20 epochs, GR00T-A20-aligned augmentation; output psi0-runs/<task>/aligned, HF psi0/<task>/b512/.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi_aligned.sh" a17 4 "$RANK" "$MASTER_ADDR" 30736 16
