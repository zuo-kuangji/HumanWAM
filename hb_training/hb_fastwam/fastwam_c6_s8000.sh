#!/bin/bash
# hb_relay cmd item (2026-11-18 11:xx user): FastWAM C6 aligned with the long-task rule (C6 / A14 = 2x steps; GR00T C6
# 8000), 8000 steps, lr warmup / cosine over 8000. Replaces item 87 (fastwam_c6.sh, 4000-step schedule, stopped at
# about step 600; its run directory is kept as runs/c6.aborted-s4000-20261118).
G=$(dirname "$(dirname "$(realpath "$0")")")
R=$G/fastwam/runs/c6; A=$G/fastwam/runs/c6.aborted-s4000-20261118
if [ "${RANK:?}" = 0 ]; then
  if [ -d $R ] && [ ! -f $R/.s8000 ] && [ ! -f $R/complete.json ]; then mv $R $A; fi
  mkdir -p $R && echo "8000 steps (2026-11-18 user)" > $R/.s8000
else
  for i in $(seq 120); do [ -f $R/.s8000 ] && break; sleep 5; done
  [ -f $R/.s8000 ] || { echo "rank 0 did not prepare $R"; exit 1; }
fi
exec bash "$G/hb_fastwam/fastwam_task_s.sh" c6 g1 31291 8000
