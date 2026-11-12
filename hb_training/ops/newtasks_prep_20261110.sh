#!/bin/bash
# 2026-11-10: prepare + CPU-verify the new tasks (basket new version, a17 H2) for GR00T / pi0.5 / Psi-Zero / ACT, then
# insert them by model priority (pool b696: GR00T jobs before pi0.5 jobs; relay queue: Psi-Zero then ACT items).
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z
G=$P/policies-20261109
L=$G/newtasks-20261110
mkdir -p $L
exec >> $L/prep.log 2>&1
echo "$(date) start"
until grep -q A17_OK $P/dl-basket-a17-20261110.log; do grep -qE 'Error|Traceback' $P/dl-basket-a17-20261110.log && { echo DOWNLOAD_FAILED; exit 1; }; sleep 30; done
echo "$(date) downloads done"
FAIL=0
for t in basket a17; do
  (cd $G/hb_pi05 && source ./hb_env.sh && python hb_prepare.py $t --workers 24 > $L/pi05-prepare-$t.log 2>&1 && python verify_cpu.py --task $t > $L/pi05-verify-$t.log 2>&1) || { echo "PI05 $t FAILED"; FAIL=1; }
  (env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_psi0/prepare_data.py --force $t > $L/psi0-prepare-$t.log 2>&1 && env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_psi0/verify_cpu.py --task $t > $L/psi0-verify-$t.log 2>&1) || { echo "PSI0 $t FAILED"; FAIL=1; }
  (E=$G/envs/groot-py312; export LD_LIBRARY_PATH=$E/lib/hbstub:\$LD_LIBRARY_PATH PATH=$E/bin:$PATH PYTHONPATH=$G/HumanWAM/third_party/Isaac-GR00T:$G/hb_groot HF_HOME=$G/hf-home HF_HUB_OFFLINE=1; cd $G/hb_groot && env -u RANK -u WORLD_SIZE -u MASTER_ADDR python prepare_groot_data.py $t > $L/groot-prepare-$t.log 2>&1) || { echo "GROOT $t FAILED"; FAIL=1; }
  (PSI=$G/HumanWAM/third_party/Psi0; cd $PSI; export PYTHONPATH=$PSI/src:$G/hb_act:$G/hb_psi0 PSI_SRC=$PSI HB_AUG_DIR=$G/hb_groot TORCH_HOME=$G/torch-home HF_HOME=$G/hf-home HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/tmp/hb_act_hfds; env -u RANK -u WORLD_SIZE -u MASTER_ADDR $G/envs/psi0/bin/python $G/hb_act/verify_cpu.py --task $t > $L/act-verify-$t.log 2>&1) || { echo "ACT $t FAILED"; FAIL=1; }
  echo "$(date) $t done (FAIL=$FAIL)"
done
[ "$FAIL" = 0 ] || { echo "NOT INSERTED (some step failed)"; exit 1; }
python3 - $G <<'EOF'
import os, re, sys
G = sys.argv[1]
J = G + '/pools/b696/jobs.txt'
lines = [l for l in open(J).read().split('\n') if l.strip()]
ids = [l.split()[0] for l in lines if not l.startswith('#')]
assert 'j11' not in ids
out = []
for l in lines:
    out.append(l)
    if l.startswith('j10 '):  # after the last GR00T job
        out += ['j11 groot basket 87', 'j12 groot a17 87']
out += ['# 2026-11-10 new tasks basket (current-dynamics-100-v21) and a17 (H2 unscrew bottle): GR00T above, pi0.5 here', 'j13 pi05 basket 87', 'j14 pi05 a17 87']
assert sum(1 for l in out if l.startswith(('j11 ', 'j12 '))) == 2
open(J + '.tmp', 'w').write('\n'.join(out) + '\n'); os.replace(J + '.tmp', J)
Q = G + '/relay/queue.txt'
q = open(Q).read().split('\n')
if q and q[-1] == '': q.pop()
act = [i for i, l in enumerate(q) if l.strip() and not re.match(r'\s*#', l)]
pos = min(int(open(G + '/relay/pos-node-%d' % n).read()) for n in range(4))
ip = next(i for i in act if q[i] == 'psi0 trolley'); assert act.index(ip) > pos
q[ip + 1:ip + 1] = ['psi0 basket', 'psi0 a17']
ia = next(i for i, l in enumerate(q) if l.endswith('hb_act/act_trolley.sh'))
q[ia + 1:ia + 1] = ['cmd %s/hb_act/act_basket.sh' % G, 'cmd %s/hb_act/act_a17.sh' % G]
open(Q + '.tmp', 'w').write('\n'.join(q) + '\n'); os.replace(Q + '.tmp', Q)
print('INSERTED')
EOF
echo "$(date) ALL_DONE"
