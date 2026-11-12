#!/bin/bash
# 2026-11-13: re-run one GR00T full-FT task on THIS node outside the pool, with fewer DataLoader workers (host RAM:
# 3-camera H2 video decoding with 8 workers x 8 ranks exhausted 1.5 TB and the kernel OOM killer stopped the run).
# Usage (in the node's PAI terminal): setsid nohup bash $G/hb_groot/groot_retry_node.sh <task> <node> <workers> > /dev/null 2>&1 &
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
T=$1 N=$2 W=${3:-4}
C=$G/control
touch "$C/hold-node-$N" "$C/fill-$T.running"
echo "$(date -u) $(hostname) start task=$T workers=$W" > "$C/fill-$T.log"
R=$G/groot-runs-ft4k
for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 5; done
[ -d "$R/$T" ] && [ ! -e "$R/$T.fail-hostoom-20261113" ] && mv "$R/$T" "$R/$T.fail-hostoom-20261113"
HB_GROOT_WORKERS=$W bash "$G/hb_groot/groot_single_ft_w.sh" "$T" 32 "$R" > "$G/pools/groot256/status/retry-$T.out" 2>&1
rc=$?
echo "$rc" > "$C/fill-$T.rc"
echo "$(date -u) end rc=$rc" >> "$C/fill-$T.log"
rm -f "$C/fill-$T.running" "$C/hold-node-$N"
