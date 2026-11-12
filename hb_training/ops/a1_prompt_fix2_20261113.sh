#!/bin/bash
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z; G=$P/policies-20261109
pkill -f 'a1_prep_20261113\.sh'; sleep 1; pkill -f 'groot-data/a1$|prepare_groot_data\.py a1|prepare_data\.py --force a1|prepare_deltaflow\.py a1|verify_cpu\.py --task a1'; sleep 3
pgrep -af 'a1_prep_2026|prepare_groot_data.py a1|force a1|prepare_deltaflow.py a1' | grep -v a1_prompt || echo NO_A1_PREP_LEFT
D=$P/a1-ground-object-pickup-v1-f47f07e/data
$P/venv/bin/python - "$D" <<'EOF'
import sys
from pathlib import Path
M = Path(sys.argv[1]) / 'meta'
for name in ('tasks.jsonl', 'episodes.jsonl'):
    s = (M / (name + '.orig-hf-f47f07e')).read_text()
    (M / name).write_text(s.replace('a1-Ground Object Pickup: ', ''))
    print('PATCHED', name, 'left', (M / name).read_text().count('a1-Ground Object Pickup'))
EOF
sed -e 's#^\$G/envs/psi0/bin/python \$G/dl_a1.py .*#echo "download skipped (done 04:29; meta task text edited locally, see data/HB_LOCAL_EDITS.md)" > $P/dl-a1-prep2.log; echo A1_DONE >> $P/dl-a1-prep2.log#' -e 's#grep -q A1_DONE \$P/dl-a1-20261113.log#grep -q A1_DONE $P/dl-a1-prep2.log#' $G/a1_prep_20261113.sh > $G/a1_prep2_20261113.sh
grep -n 'dl_a1\|A1_DONE\|skipped' $G/a1_prep2_20261113.sh | cut -c1-140
bash -n $G/a1_prep2_20261113.sh && echo SYNTAX_OK
[ -d $G/groot-data/a1 ] && rm -rf $G/groot-data/a1.prefix-rerun && mv $G/groot-data/a1 $G/groot-data/a1.prefix-rerun
cd $P && setsid nohup bash $G/a1_prep2_20261113.sh > /dev/null 2>&1 < /dev/null &
sleep 2; echo started
