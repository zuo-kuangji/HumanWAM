#!/bin/bash
# 2026-11-13 01:5x: GR00T full-FT on 3-camera H2 tasks leaks host RAM in proportion to the samples decoded (~0.7 GB per
# optimizer step, independent of the DataLoader worker count; 1.5 TB is exhausted after ~1650-1900 steps, which killed
# a17 at step 1924). Guarded run on THIS node, same recipe as groot_single_ft.sh (full FT, 4000 steps, global 256,
# lr / warmup / schedule unchanged) except:
#   - a checkpoint every 1000 steps; after each one the run is stopped (host memory reset) and resumed from it
#     (--resume_from_checkpoint: weights, optimizer, scheduler, RNG; ignore_data_skip=True in GR00T),
#   - at the end only checkpoint-2000 (weights) and checkpoint-4000 (weights + resume state) are kept.
# Usage (node's PAI terminal): setsid nohup bash $G/hb_groot/groot_ft_guard2.sh <task> <node> [workers] > /dev/null 2>&1 &
G=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109
T=$1 N=$2 W=${3:-2}
C=$G/control R=$G/groot-runs-ft4k
OUT=$R/$T/production
L=$C/fill-$T.log
touch "$C/hold-node-$N" "$C/fill-$T.running"
log() { echo "$(date -u) $*" >> "$L"; }
log "$(hostname) guard2 start task=$T node=$N workers=$W save_every=1000"
gpu_free() { for i in $(seq 60); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && return 0; sleep 5; done; return 1; }
PAT="hb_launch\.py.*groot-data/$T "
stop_run() { pkill -TERM -f "$PAT"; sleep 30; pkill -KILL -f "$PAT"; sleep 10; gpu_free; }
last_ck() { ls -d "$OUT"/checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | while read s; do
  [ -f "$OUT/checkpoint-$s/trainer_state.json" ] && [ -f "$OUT/checkpoint-$s/scheduler.pt" ] && echo "$s"; done | tail -1; }
gpu_free
sleep 20; touch "$C/hold-node-$N" "$C/fill-$T.running"
[ -d "$R/$T" ] && mv "$R/$T" "$R/$T.fail-hostoom-$(date -u +%Y%m%dT%H%M)"
fails=0 seg=0
while [ ! -f "$R/$T/production-complete.json" ] && [ $fails -lt 4 ] && [ $seg -lt 12 ]; do
  seg=$((seg + 1)); start=$(last_ck); extra=""
  [ -n "$start" ] && extra="--resume_from_checkpoint"
  log "segment $seg: start from ${start:-scratch}"
  HB_GROOT_WORKERS=$W HB_GROOT_EXTRA=$extra HB_GROOT_SAVE_STEPS=1000 HB_GROOT_SAVE_LIMIT=6 \
    bash "$G/hb_groot/groot_single_ft_r2.sh" "$T" 32 "$R" >> "$G/pools/groot256/status/guard-$T.out" 2>&1 &
  pid=$! stopped=0
  while kill -0 $pid 2>/dev/null; do
    now=$(last_ck)
    if [ -n "$now" ] && [ "$now" != "${start:-0}" ] && [ "$now" -lt 4000 ]; then
      sleep 60; log "checkpoint-$now complete: stopping to reset host memory"; stop_run; stopped=1
    fi
    sleep 20
  done
  wait $pid; rc=$?
  log "segment $seg rc=$rc stopped=$stopped last_ck=$(last_ck)"
  [ $stopped = 1 ] || [ $rc = 0 ] || { fails=$((fails + 1)); gpu_free; }
done
if [ -f "$R/$T/production-complete.json" ]; then
  for s in 1000 3000; do rm -rf "$OUT/checkpoint-$s"; done
  printf '{"task": "%s", "segments": %s, "checkpoint_every": 1000, "kept": [2000, 4000], "reason": "host-RAM growth (~0.7 GB / step) in the 3-camera video data pipeline; the run was stopped after every 1000-step checkpoint and resumed from it (weights, optimizer, scheduler, RNG restored; same lr schedule and total 4000 steps)", "dataloader_workers_per_rank": %s, "utc": "%s"}\n' "$T" "$seg" "$W" "$(date -u +%FT%TZ)" > "$R/$T/resume-note.json"
  rc=0
else
  rc=1
fi
echo "$rc" > "$C/fill-$T.rc"
log "guard2 end rc=$rc segments=$seg fails=$fails"
rm -f "$C/fill-$T.running" "$C/hold-node-$N"
