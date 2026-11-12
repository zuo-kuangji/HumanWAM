#!/bin/bash
# 2026-11-13: H2 tasks train with head + chest + wrist (user rule); skipped until the 3-camera adapter of this model
# is verified and $G/control/h2-3cam-ready-deltaflow-a18 exists.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-deltaflow-a18" ] || { echo "SKIP deltaflow a18: 3-camera adapter not ready (control/h2-3cam-ready-deltaflow-a18 missing)"; exit 3; }
# hb_relay `cmd` item for the Delta-flow a18 40-epoch run (cmd items cannot pass arguments).
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a18
