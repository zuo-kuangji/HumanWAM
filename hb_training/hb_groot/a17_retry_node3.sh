#!/bin/bash
# 2026-11-13 01:3x: GR00T a17 (pool job g02) died at step 1924 on worker-0: a DataLoader worker was killed by the
# kernel OOM killer (host RAM, 3-camera H2 video decoding with 8 workers x 8 ranks). Re-run on node 3 (worker-2,
# idle at the relay barrier) with 4 dataloader workers per rank; recipe otherwise identical (groot_single_ft.sh).
# Started by hand in the worker-2 PAI terminal: setsid nohup bash $G/hb_groot/a17_retry_node3.sh > /dev/null 2>&1 &
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
C=$G/control
touch "$C/hold-node-3" "$C/fill-a17.running"
echo "$(date -u) $(hostname) start" > "$C/fill-a17.log"
R=$G/groot-runs-ft4k
[ -d "$R/a17" ] && [ ! -e "$R/a17.fail-hostoom-20261113" ] && mv "$R/a17" "$R/a17.fail-hostoom-20261113"
HB_GROOT_WORKERS=4 bash "$G/hb_groot/groot_single_ft_w.sh" a17 32 "$R" > "$G/pools/groot256/status/g02r.out" 2>&1
rc=$?
echo "$rc" > "$C/fill-a17.rc"
echo "$(date -u) end rc=$rc" >> "$C/fill-a17.log"
rm -f "$C/fill-a17.running" "$C/hold-node-3"
