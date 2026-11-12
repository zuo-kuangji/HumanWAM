#!/bin/bash
# After B7 prep + CPU verification pass: release stage 1 (S1_READY), queue Psi-Zero B7 after Psi-Zero a17 and ACT B7
# (stage 2) right after it (user priority GR00T > pi0.5 > Psi-Zero > ACT > Delta Flow > DiT4DiT), add auto-upload jobs.
G=$(dirname "$(dirname "$(realpath "$0")")")
P=$(dirname "$G")
F=$G/b7-20261111
exec >> $F/finalize.log 2>&1
echo "$(date) waiting for prep"
until grep -qE 'B7_PREP_OK|B7_PREP_FAILED' $F/prep.log; do sleep 30; done
grep -q B7_PREP_OK $F/prep.log || { echo "$(date) prep failed: nothing released"; exit 1; }
touch $F/S1_READY; echo "$(date) S1_READY"
python3 - $G <<'PY'
import os, re, sys, json
G = sys.argv[1]; Q = G + '/relay/queue.txt'
q = open(Q).read().split(chr(10))
if q and q[-1] == '': q.pop()
act = [i for i, l in enumerate(q) if l.strip() and not re.match(r'\s*#', l)]
pos = min(int(open(G + '/relay/pos-node-%d' % n).read()) for n in range(4))
ia = next(i for i in act if q[i].strip() == 'psi0 a17'); assert act.index(ia) > pos
q[ia + 1:ia + 1] = ['psi0 b7', 'cmd %s/hb_b7/b7_s2.sh' % G]
open(Q + '.tmp', 'w').write(chr(10).join(q) + chr(10)); os.replace(Q + '.tmp', Q)
print('QUEUED')
J = os.path.dirname(G) + '/hf-autoupload/jobs.json'
jobs = json.load(open(J))
for m in ('psi0', 'act'):
    if {'model': m, 'task': 'b7'} not in jobs: jobs.append({'model': m, 'task': 'b7'})
open(J + '.tmp', 'w').write(json.dumps(jobs, indent=1)); os.replace(J + '.tmp', J); print('JOBS', len(jobs))
PY
echo "$(date) FINALIZED"
