"""2026-11-15 (H2 training guide a14 4.3, user list item 1): Delta Flow A14 v2 on the legacy a18 e40 configuration,
action82 = body_target 31 + hands 44 + planner 7. Builds the code tree deltaflow-runs/HumanWAM-h2p = HumanWAM-h2 with
exactly two files patched (the 75-D action hard-coded in the loader and the model wrapper becomes ACTION_DIM = 82;
nothing else), then prepares task a14lg through prepare_deltaflow_h2legacy.prepare (same recipe as a17lg: 256 / 8000
steps / checkpoints 4000 + 8000 / warmup 400, b6-visual-aug-v2, no proprio augmentation), with the model config's
action dims set to 82."""
import inspect
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow as base  # noqa: E402
import prepare_deltaflow_h2 as h2  # noqa: E402
import prepare_deltaflow_h2legacy as lg  # noqa: E402

SRC, H2P = h2.CODE, base.ROOT / 'HumanWAM-h2p'
SCHEMA = 'a14v2-h2-sharpa-body31-hand44-planner7-state78-chunk16-v1'
POUR = 'src/imagewam/datasets/h2_pour.py'
DELTA = 'src/imagewam/h2_pour_delta.py'
PATCHES = {
    POUR: [
        ("SCHEMA = 'a18-h2-sharpa-body31-hand44-state78-chunk16-v1'\nACTION_DIM = 75\n",
         f"SCHEMA = '{SCHEMA}'\nACTION_DIM = 82  # HB 2026-11-15: body_target 31 + hands 44 + planner 7 (a14 v2)\n", 1),
        ("    fields=('observation.state','observation.projected_gravity','action.body_target','action.hands','action.arms')\n",
         "    fields=('observation.state','observation.projected_gravity','action.body_target','action.hands','action.arms','action.planner')\n", 1),
        ("    q,g,b,h,arms=(np.asarray(tab[field].to_pylist(),dtype=np.float32) for field in fields)\n"
         "    if [x.shape for x in (q,g,b,h,arms)]!=[(n,75),(n,3),(n,31),(n,44),(n,14)]:\n",
         "    q,g,b,h,arms,pl=(np.asarray(tab[field].to_pylist(),dtype=np.float32) for field in fields)\n"
         "    if [x.shape for x in (q,g,b,h,arms,pl)]!=[(n,75),(n,3),(n,31),(n,44),(n,14),(n,7)]:\n", 1),
        ("    action=np.concatenate([b,h],axis=1)\n", "    action=np.concatenate([b,h,pl],axis=1)\n", 1),
    ],
    DELTA: [
        ('SCHEMA_ID = "a18-h2-sharpa-body31-hand44-state78-chunk16-v1"\n',
         f'SCHEMA_ID = "{SCHEMA}"\nACTION_DIM = 82  # HB 2026-11-15: body31 + hands44 + planner7 (a14 v2)\n', 1),
        ('CHECKPOINT_FORMAT = "h2_sharpa_body31_hand44_flux2_a18_chunk16_v1"',
         'CHECKPOINT_FORMAT = "h2_sharpa_body31_hand44_planner7_flux2_a14v2_chunk16_v1"', 1),
        ('action_dim == 75', 'action_dim == ACTION_DIM', None),
        ('(True,) * 75', '(True,) * ACTION_DIM', None),
        ('[True] * 75', '[True] * ACTION_DIM', None),
        ('"action_dim": 75', '"action_dim": ACTION_DIM', None),
        ('"state_only_action_dim": 75', '"state_only_action_dim": ACTION_DIM', None),
        ('(1, 16, 75)', '(1, 16, ACTION_DIM)', None),
        ('(16, 75)', '(16, ACTION_DIM)', None),
        ('expected_action_dim=75', 'expected_action_dim=ACTION_DIM', None),
        ('(78, 75, 16)', '(78, ACTION_DIM, 16)', None),
        ('nn.Linear(75, hidden)', 'nn.Linear(ACTION_DIM, hidden)', None),
        ('nn.Linear(hidden, 75, bias=False)', 'nn.Linear(hidden, ACTION_DIM, bias=False)', None),
        ('"target_action_dim": 75', '"target_action_dim": ACTION_DIM', None),
        ('int(action_config.get("action_dim", -1)) == 75', 'int(action_config.get("action_dim", -1)) == ACTION_DIM', None),
        ('action_dit_config.action_dim=75', 'action_dit_config.action_dim=82', None),
    ],
}


def build_h2p():
    record = H2P / 'deltaflow-code.json'
    if record.is_file():
        for rel, digest in json.loads(record.read_text())['sha256'].items():
            if base.sha(H2P / rel) != digest:
                raise SystemExit(f'{H2P / rel} changed after preparation; rename {H2P} aside and rerun')
        base.log('H2 action82 code copy present and unchanged:', H2P)
        return
    if H2P.exists():
        raise SystemExit(f'{H2P} exists without deltaflow-code.json (incomplete); rename it aside and rerun')
    src_rec = json.loads((SRC / 'deltaflow-code.json').read_text())
    for rel, digest in src_rec['sha256'].items():
        if base.sha(SRC / rel) != digest:
            raise SystemExit(f'H2 code copy changed: {SRC / rel}')
    tmp = base.ROOT / f'.HumanWAM-h2p.tmp{os.getpid()}'
    shutil.copytree(SRC, tmp, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    report = {}
    for rel, pairs in PATCHES.items():
        f = tmp / rel
        text = f.read_text(encoding='utf-8')
        for old, new, count in pairs:
            n = text.count(old)
            if (count is not None and n != count) or n == 0:
                raise SystemExit(f'{rel}: patch anchor found {n}x (want {count or ">=1"}): {old!r}')
            text = text.replace(old, new)
            report[f'{rel}: {old}'] = n
        f.write_text(text, encoding='utf-8')
    left = [f'{i + 1}: {line.strip()[:120]}' for i, line in enumerate((tmp / DELTA).read_text().splitlines())
            if re.search(r'(?<![\w.])75(?![\w.])', line)]
    base.log('remaining literal 75 in h2_pour_delta.py (messages only expected):', json.dumps(left, ensure_ascii=False))
    code_left = [x for x in left if 'require(' in x or '==' in x or 'Linear(' in x or '": 75' in x]
    if code_left:
        raise SystemExit(f'h2_pour_delta.py still hard-codes 75 in code: {code_left}')
    files = sorted(set(src_rec['sha256']) | set(PATCHES))
    base.dump(tmp / 'deltaflow-code.json', dict(
        source=str(SRC), source_record_sha256=base.sha(SRC / 'deltaflow-code.json'),
        created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        changes=list(src_rec.get('changes', [])) + [
            'HB 2026-11-15 (H2 training guide a14 4.3): action82 = body_target 31 + hands 44 + planner 7. h2_pour.py loads '
            'action.planner and concatenates it after the hands; h2_pour_delta.py uses ACTION_DIM = 82 wherever the 75-D '
            'action was hard-coded (checks, a2a / inference noise masks, action encoder / head sizes); schema ' + SCHEMA],
        patches=report, sha256={rel: base.sha(tmp / rel) for rel in files}))
    tmp.rename(H2P)
    base.log('H2 action82 code copy built:', H2P)


lg.TASKS['a14lg'] = dict(data=base.P / 'a14-unscrew-bulb-v2-8411a6d' / 'data', frames=185638, code=H2P, action_dim=82,
                         qwen_from=base.ROOT / 'a14c2' / 'qwen-text', policy_id='a14v2-h2-rgb-v1',
                         source='humanoid-benchmark/a14-unscrew-bulb-expert-v2 @ 8411a6dac45d0097d6068a60307de0c2e3b621a7')
_yaml_load = lg.yaml.safe_load


def _load_for_82(text):
    cfg = _yaml_load(text)
    if isinstance(cfg, dict) and 'model' in cfg and cfg.get('data', {}).get('train', {}).get('_target_', '').endswith('H2PourDataset'):
        m = cfg['model']
        m['expected_action_dim'] = 82
        m['action_dit_config']['action_dim'] = 82
        m['loss']['action_a2a']['source_noise_dim_mask'] = [True] * 82
        m['state_only_schema_id'] = SCHEMA
    return cfg


if __name__ == '__main__':
    build_h2p()
    sys.path.insert(0, str(H2P / 'src'))
    import imagewam.h2_pour_delta as md
    if not md.__file__.startswith(str(H2P / 'src')):
        raise SystemExit(f'imported the wrong imagewam: {md.__file__}')
    sig = inspect.signature(md.create_imagewam_flux2_wholebody_delta)
    base.log('model module check: ACTION_DIM', md.ACTION_DIM, 'factory defaults',
             {k: sig.parameters[k].default for k in ('action_state_dim', 'expected_action_dim') if k in sig.parameters})
    if md.ACTION_DIM != 82 or md.SCHEMA_ID != SCHEMA:
        raise SystemExit('h2_pour_delta.py is not the action82 variant')
    lg.yaml.safe_load = _load_for_82
    lg.prepare('a14lg')
    lg.yaml.safe_load = _yaml_load
    import yaml
    y = (base.ROOT / 'a14lg' / 'run/production.yaml').read_text()
    c = yaml.safe_load(y)
    m = c['model']
    print('A14LG_MODEL', m['expected_action_dim'], m['action_dit_config']['action_dim'],
          len(m['loss']['action_a2a']['source_noise_dim_mask']), m['state_only_schema_id'],
          'yaml lines with 75:', [l.strip()[:80] for l in y.splitlines() if re.search(r'(?<![\w.])75(?![\w.])', l)][:8],
          flush=True)
