"""2026-11-14 21:xx user: B1 rice ball into microwave (G1) trains on ALL 115 episodes (no forced 100 any more).
Delta Flow on a copy of the shared G1 code tree (deltaflow-runs/HumanWAM-allep) whose loader (wholebody.py) and train
entry (train_deltaflow.py) accept training episodes 0..N-1 instead of exactly 0-99. Everything else is the unchanged
G1 recipe of prepare_deltaflow.py: aug v2, 44-D proprio (no velocity), proprio aligned with GR00T A20, fixed 8000 steps,
checkpoints 6000 / 8000, warmup 5 %, 32 GPUs x 8 = 256, no gradient accumulation. The shared tree stays untouched."""
import json
import os
import shutil
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow as base  # noqa: E402

N = 115
base.TASKS['b1'] = dict(data=base.P / 'b1-riceball-microwave-v1-36da298/data', old=None, control_hz=50,
                        total_episodes=N, source='b1-riceball-microwave-expert-v1 @36da298 (all 115 episodes)')
SRC = base.ROOT / 'HumanWAM'
CODE_ALL = base.ROOT / 'HumanWAM-allep'
PATCHES = {
    'src/imagewam/datasets/wholebody.py': [
        ('    if (train != list(range(100)) or', '    if (train != list(range(len(train))) or'),
        ('"Require training episodes exactly 0-99; later episodes may only be listed as shelved"',
         '"Require training episodes exactly 0..N-1 (HB 2026-11-14: all episodes of the collection)"'),
        ('list(meta["train"]) != list(range(100)):', 'list(meta["train"]) != list(range(len(meta["train"]))):'),
        ('"Require the fixed first-100 training partition (episodes 0-99)"',
         '"Require a 0..N-1 training partition (HB 2026-11-14: all episodes)"')],
    'scripts/wholebody/train_deltaflow.py': [
        ("receipt['train_episode_ids'] == list(range(100)),",
         "receipt['train_episode_ids'] == list(range(len(receipt['train_episode_ids']))),"),
        ("'Training must use exactly episodes 0-99')", "'Training must use exactly episodes 0..N-1')")],
}
_base_build = base.build_code


def build_all():
    base.CODE = SRC
    _base_build()                     # the shared G1 tree must be present and unchanged
    base.CODE = CODE_ALL
    record = CODE_ALL / 'deltaflow-code.json'
    if record.is_file():
        for rel, digest in json.loads(record.read_text())['sha256'].items():
            if base.sha(CODE_ALL / rel) != digest:
                raise SystemExit(f'{CODE_ALL / rel} changed after preparation; rename {CODE_ALL} aside and rerun')
        base.log('all-episodes code copy present and unchanged:', CODE_ALL)
        return
    if CODE_ALL.exists():
        raise SystemExit(f'{CODE_ALL} exists without deltaflow-code.json (incomplete); rename it aside and rerun')
    src_rec = json.loads((SRC / 'deltaflow-code.json').read_text())
    tmp = base.ROOT / f'.HumanWAM-allep.tmp{os.getpid()}'
    shutil.copytree(SRC, tmp, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for rel, pairs in PATCHES.items():
        f = tmp / rel
        text = f.read_text(encoding='utf-8')
        for old, new in pairs:
            if text.count(old) != 1:
                raise SystemExit(f'{rel}: patch anchor found {text.count(old)}x: {old!r}')
            text = text.replace(old, new)
        f.write_text(text, encoding='utf-8')
    files = sorted(set(src_rec['sha256']) | set(PATCHES))
    base.dump(tmp / 'deltaflow-code.json', dict(
        source=str(SRC), source_record_sha256=base.sha(SRC / 'deltaflow-code.json'),
        created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        changes=list(src_rec['changes']) + [
            'HB 2026-11-14 (user: no forced 100 episodes): wholebody.py and train_deltaflow.py accept training '
            'episodes 0..N-1 (B1: all 115) instead of exactly 0-99; nothing else changed'],
        sha256={rel: base.sha(tmp / rel) for rel in files}))
    tmp.rename(CODE_ALL)
    base.log('all-episodes code copy built:', CODE_ALL)


base.build_code = build_all
base.EPISODES = N

if __name__ == '__main__':
    sys.argv = [sys.argv[0], 'b1']
    base.main()
    R = base.ROOT / 'b1'
    d = json.loads((R / 'data-verified.json').read_text())
    if (d['train_episode_ids'] != list(range(N)) or d['shelved_episode_ids'] != [] or d['frames'] != 105102
            or d['max_steps'] != 8000 or d['code'] != str(CODE_ALL) or d['proprio_dim'] != 44 or d['joint_velocity']):
        raise SystemExit(f'b1: unexpected receipt {d}')
    y = (R / 'run/production.yaml').read_text()
    if y.count('train_episode_ids: 0-99\n') == 1:
        (R / 'run/production.yaml').write_text(y.replace('train_episode_ids: 0-99\n', f'train_episode_ids: 0-{N - 1}\n'))
    import yaml
    c = yaml.safe_load((R / 'run/production.yaml').read_text())
    if c['batch_size'] != 8 or c['gradient_accumulation_steps'] != 1 or c['max_steps'] != 8000:
        raise SystemExit(f"b1: config batch {c['batch_size']} accum {c['gradient_accumulation_steps']}")
    d.update(episode_rule='2026-11-14 user: all 115 episodes, no forced 100', per_gpu_micro_batch=base.MICRO,
             gradient_accumulation=1, global_batch=base.MICRO * base.WORLD)
    (R / 'data-verified.json').write_text(json.dumps(d, indent=2))
    print('B1_DF_OK', json.dumps({k: d[k] for k in ('frames', 'episodes', 'updates_per_epoch', 'max_steps', 'save_steps',
                                                    'warmup_steps', 'per_gpu_micro_batch', 'gradient_accumulation',
                                                    'global_batch', 'code')}), flush=True)
