#!/bin/bash
# 2026-11-13 01:4x: GR00T full-FT on 3-camera H2 tasks leaks host RAM in the DataLoader workers (~0.6 GB/min per
# worker, linear; 8 workers x 8 ranks reached 1.5 TB and the kernel OOM killer stopped a17 at step 1924, a14 was
# stopped pre-emptively). Guarded run on THIS node, same recipe (groot_single_ft.sh: full FT, 4000 steps, global 256,
# checkpoints 2000 / 4000) with:
#   - <workers> DataLoader workers per rank (default 2),
#   - phase 1: train from scratch; as soon as checkpoint-2000 is complete the run is stopped (memory reset) and
#   - phase 2: resumed from checkpoint-2000 (--resume_from_checkpoint: weights, optimizer, scheduler, RNG) to 4000.
#   A phase-1 crash before step 2000 restarts from scratch with 1 worker; a phase-2 crash resumes once more.
# Usage (node's PAI terminal): setsid nohup bash $G/hb_groot/groot_ft_guard.sh <task> <node> [workers] > /dev/null 2>&1 &
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
T=$1 N=$2 W=${3:-2}
C=$G/control R=$G/groot-runs-ft4k
OUT=$R/$T/production
L=$C/fill-$T.log
touch "$C/hold-node-$N" "$C/fill-$T.running"
log() { echo "$(date -u) $*" >> "$L"; }
log "$(hostname) guard start task=$T node=$N workers=$W"
gpu_free() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 5; done; return 1; }
PAT="hb_launch\.py.*groot-data/$T "
stop_run() { pkill -TERM -f "$PAT"; sleep 30; pkill -KILL -f "$PAT"; sleep 10; gpu_free; }
run_phase() {  # $1 workers, $2 extra args; watchdog: $3 = 1 -> stop once checkpoint-2000 is complete
  HB_GROOT_WORKERS=$1 HB_GROOT_EXTRA=$2 bash "$G/hb_groot/groot_single_ft_r.sh" "$T" 32 "$R" >> "$G/pools/groot256/status/guard-$T.out" 2>&1 &
  local pid=$! stopped=0
  while kill -0 $pid 2>/dev/null; do
    if [ "$3" = 1 ] && [ -f "$OUT/checkpoint-2000/trainer_state.json" ] && [ -f "$OUT/checkpoint-2000/scheduler.pt" ]; then
      sleep 90; log "checkpoint-2000 complete: stopping phase 1 to reset host memory"; stop_run; stopped=1
    fi
    sleep 20
  done
  wait $pid; local rc=$?
  [ $stopped = 1 ] && return 100
  return $rc
}
gpu_free
sleep 20; touch "$C/hold-node-$N" "$C/fill-$T.running"  # an older launcher of this task may have removed them on exit
for d in "$R/$T"; do [ -d "$d" ] && mv "$d" "$d.fail-hostoom-$(date -u +%Y%m%dT%H%M)"; done
run_phase "$W" "" 1; rc=$?
log "phase 1 rc=$rc"
if [ $rc != 100 ] && [ $rc != 0 ]; then
  if [ ! -f "$OUT/checkpoint-2000/trainer_state.json" ]; then
    log "phase 1 failed before step 2000: restart from scratch with 1 worker"
    gpu_free; mv "$R/$T" "$R/$T.fail-hostoom-$(date -u +%Y%m%dT%H%M)"
    run_phase 1 "" 1; rc=$?; log "phase 1 (retry) rc=$rc"
  else
    rc=100
  fi
fi
if [ $rc = 100 ]; then
  gpu_free
  printf '{"task": "%s", "resumed_at_step": 2000, "reason": "host-RAM growth in the 3-camera video DataLoader workers; run stopped after checkpoint-2000 and resumed (weights, optimizer, scheduler, RNG restored)", "dataloader_workers_per_rank": %s, "utc": "%s"}\n' "$T" "$W" "$(date -u +%FT%TZ)" > "$R/$T/resume-note.json"
  run_phase "$W" "--resume_from_checkpoint" 0; rc=$?; log "phase 2 rc=$rc"
  if [ $rc != 0 ]; then gpu_free; run_phase "$W" "--resume_from_checkpoint" 0; rc=$?; log "phase 2 (retry) rc=$rc"; fi
fi
echo "$rc" > "$C/fill-$T.rc"
log "guard end rc=$rc"
rm -f "$C/fill-$T.running" "$C/hold-node-$N"
