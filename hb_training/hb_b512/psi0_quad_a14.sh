#!/bin/bash
# 2026-11-13: skipped until the Psi-Zero 3-camera A14 view is verified ($G/control/h2-3cam-ready-b512-a14).
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-b512-a14" ] || { echo "SKIP b512 a14: 3-camera view not verified"; exit 3; }
# Relay cmd item (2026-11-13): Psi-Zero A14 (H2, head + chest + wrist) on all 4 nodes, 4 x 8 x 16 = 512, 4000 steps,
# GR00T-A20-aligned augmentation; output psi0-runs/a14/b512, HF psi0/a14-unscrew-bulb/.
G=$(dirname "$(dirname "$(realpath "$0")")")
exec bash "$G/hb_b512/psi0_multi.sh" a14 4 "$RANK" "$MASTER_ADDR" 30760 16
