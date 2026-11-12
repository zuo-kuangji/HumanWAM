#!/bin/bash
# 2026-11-13 04:4x user: A1 task text without the "a1-Ground Object Pickup: " title (aligned with the other tasks).
P=/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z; G=$P/policies-20261109
pkill -f 'a1_prep_20261113\.sh'; pkill -f 'groot-data/a1$|prepare_data\.py --force a1|prepare_deltaflow\.py a1|verify_cpu\.py --task a1'; sleep 3
pgrep -af 'a1_prep|groot-data/a1|force a1|prepare_deltaflow.py a1' | grep -v a1_prompt || echo NO_A1_PREP_LEFT
D=$P/a1-ground-object-pickup-v1-f47f07e/data
$P/venv/bin/python - "$D" <<'EOF'
import json, shutil, sys
from pathlib import Path
D = Path(sys.argv[1]); M = D / 'meta'
old_prefix = 'a1-Ground Object Pickup: '
for name in ('tasks.jsonl', 'episodes.jsonl'):
    f = M / name; bak = M / (name + '.orig-hf-f47f07e')
    if not bak.exists(): shutil.copy2(f, bak)
    s = bak.read_text()
    n = s.count(old_prefix)
    f.write_text(s.replace(old_prefix, ''))
    print('PATCHED', name, 'occurrences', n)
others = [p.name for p in M.iterdir() if p.is_file() and not p.name.endswith('f47f07e') and old_prefix in p.read_text(errors='ignore')]
print('PREFIX_STILL_IN', others)
print('TASK', json.loads((M / 'tasks.jsonl').read_text().splitlines()[0])['task'][:90])
(D / 'HB_LOCAL_EDITS.md').write_text('2026-11-13 (user): the task text in meta/tasks.jsonl and meta/episodes.jsonl has the title prefix "a1-Ground Object Pickup: " removed so that it starts with the instruction like every other benchmark task. Originals: meta/*.orig-hf-f47f07e. Nothing else changed.\n')
EOF
[ -d $G/groot-data/a1 ] && mv $G/groot-data/a1 $G/groot-data/a1.titleprefix-20261113
cd $P && setsid nohup bash $G/a1_prep_20261113.sh > /dev/null 2>&1 < /dev/null &
sleep 2; pgrep -af a1_prep_20261113 | head -2 | cut -c1-80
