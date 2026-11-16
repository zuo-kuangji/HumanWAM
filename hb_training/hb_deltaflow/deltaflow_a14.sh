#!/bin/bash
# hb_relay cmd item: Delta Flow A14 (H2, head + chest + wrist), 8000 steps.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/h2-3cam-ready-deltaflow-a14" ] || { echo "SKIP deltaflow a14: 3-camera adapter not ready"; exit 3; }
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a14
