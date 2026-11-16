#!/bin/bash
# hb_relay cmd item: Delta Flow A10 v2 (all 150 episodes), 8000 steps. Skipped until the 150-episode loader is in place.
GG=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
[ -f "$GG/control/deltaflow-a10v2-ready" ] || { echo "SKIP deltaflow a10v2: 150-episode Delta Flow loader not ready"; exit 3; }
exec bash "$(dirname "$(realpath "$0")")/deltaflow_task.sh" a10v2
