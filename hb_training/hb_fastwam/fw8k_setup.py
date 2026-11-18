"""2026-11-18 11:xx user: long tasks (C6, A14) train 2x steps (GR00T C6 8000); FastWAM C6 and A14 v2 -> 8000 steps,
lr warmup / cosine over 8000. Writes fastwam_task_s.sh (fastwam_task.sh + STEPS arg; fastwam_task.sh is being executed,
so it is not touched), the two new relay items, and edits the relay queue (insert after the running item 87)."""
import os
import sys
from pathlib import Path
G = Path('/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109')
H = G / 'hb_fastwam'
mode = sys.argv[1]
if mode == 'scripts':
    s = (H / 'fastwam_task.sh').read_text()
    rep = [
        ('TASK=$1 PROF=$2 PORT=$3\n', 'TASK=$1 PROF=$2 PORT=$3 STEPS=${4:-4000}; SAVE=$((STEPS / 2))\n'),
        ('global=$((BS * 32)) steps=4000"', 'global=$((BS * 32)) steps=$STEPS save_every=$SAVE"'),
        ('max_steps=4000 log_every=20 save_every=2000', 'max_steps=$STEPS log_every=20 save_every=$SAVE'),
        ('"$F/data/$TASK/meta/hb_fastwam_interface.json" <<\'EOF\'', '"$F/data/$TASK/meta/hb_fastwam_interface.json" "$STEPS" "$SAVE" <<\'EOF\''),
        ('max_steps=4000, save_every=2000,', 'max_steps=int(sys.argv[6]), save_every=int(sys.argv[7]),'),
        ('#   fastwam_task.sh <task> <profile g1|h2|h2p> <port>\n',
         '#   fastwam_task_s.sh <task> <profile g1|h2|h2p> <port> [steps, default 4000; checkpoints at steps/2 and steps]\n'
         '# 2026-11-18 user: long tasks (C6, A14) train 2x steps (8000), lr warmup / cosine planned over the full run.\n'),
    ]
    for a, b in rep:
        assert s.count(a) == 1, a
        s = s.replace(a, b)
    (H / 'fastwam_task_s.sh').write_text(s)
    (H / 'fastwam_c6_s8000.sh').write_text('''#!/bin/bash
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
''')
    (H / 'fastwam_a14v2_s8000.sh').write_text('''#!/bin/bash
# hb_relay cmd item (2026-11-18 11:xx user): FastWAM A14 v2 (H2, head + chest) with the long-task rule (C6 / A14 = 2x
# steps), 8000 steps, lr warmup / cosine over 8000. Replaces fastwam_a14v2.sh (4000 steps).
exec bash "$(dirname "$(realpath "$0")")/fastwam_task_s.sh" a14v2 h2p 31311 8000
''')
    for n in ('fastwam_task_s.sh', 'fastwam_c6_s8000.sh', 'fastwam_a14v2_s8000.sh'):
        os.chmod(H / n, 0o755)
    print('scripts ok')
elif mode == 'queue':
    q = G / 'relay/queue.txt'
    lines = q.read_text().split('\n')
    c6 = [i for i, l in enumerate(lines) if l.strip() == f'cmd {H}/fastwam_c6.sh']
    a14 = [i for i, l in enumerate(lines) if l.strip() == f'cmd {H}/fastwam_a14v2.sh']
    assert len(c6) == 1 and len(a14) == 1, (c6, a14)
    if not any('fastwam_c6_s8000.sh' in l for l in lines):
        (G / 'relay/queue.txt.bak-fw8k-20261118').write_text(q.read_text())
        a = a14[0]
        lines[a] = (f'# 2026-11-18 11:xx user: long task 2x steps -> 8000 (was: cmd {H}/fastwam_a14v2.sh)\n'
                    f'cmd {H}/fastwam_a14v2_s8000.sh')
        i = c6[0]
        lines.insert(i + 1, f'# 2026-11-18 11:xx user: C6 is a long task (2x steps, GR00T C6 8000): item above stopped, re-run with 8000 steps\n'
                            f'cmd {H}/fastwam_c6_s8000.sh')
        tmp = q.with_suffix('.txt.tmp')
        tmp.write_text('\n'.join(lines))
        tmp.rename(q)
    body = [l for l in q.read_text().split('\n') if l.strip() and not l.strip().startswith('#')]
    for k in range(86, 93):
        print(k, body[k].split('/')[-1])
