"""2026-11-13: Delta Flow on H2 tasks with head + chest + wrist (user rule: every H2 task, every model, 3 cameras; third
person never an input). CPU only, node 0, idempotent.   $P/venv/bin/python prepare_deltaflow_h2w3.py <a18|a17|a14 ...>

Same recipe as prepare_deltaflow_h2.py (fixed 8000 steps, aug v2, proprio aligned with GR00T A20, 78-D state / 75-D
action, no velocity) with the 3-camera adaptation of the model's native multi-view path (views side by side):
  - code copy $G/deltaflow-runs/HumanWAM-h2w3 = verified HumanWAM-h2 with
      datasets/h2_pour.py   CAMERAS = head, chest, wrist  -> video [3, T=2, 224, 672]
      h2_pour_delta.py      current/future FLUX.2 token grid 14x42 = 588 tokens (was 14x28 = 392), input 224x672
  - global batch stays 256 (32 GPUs); per-GPU micro batch 4 x gradient accumulation 2 (was 8 x 1) because every image
    has 1.5x the tokens (conservative: no GPU was free to probe the memory)
  - the 2-camera preparations of a18 / a17 are kept as <task>.2cam-20261113
"""
import inspect
import json
import os
import shutil
import sys
import time
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow_h2 as h2  # noqa: E402

base, ROOT, P = h2.base, h2.ROOT, h2.P
SRC2 = ROOT / 'HumanWAM-h2'
CODE3 = ROOT / 'HumanWAM-h2w3'
CAMS3 = ('observation.images.head', 'observation.images.chest', 'observation.images.wrist')
MICRO3, ACC3 = 4, 2
W3_PATCHES = {
    'src/imagewam/datasets/h2_pour.py': [
        ("CAMERAS = ('observation.images.head', 'observation.images.chest')",
         "CAMERAS = ('observation.images.head', 'observation.images.chest', 'observation.images.wrist')  # HB 2026-11-13: 3 cameras"),
    ],
    'src/imagewam/h2_pour_delta.py': [
        ('"input_size": [224, 448]', '"input_size": [224, 672]'),
        ('== (392, 128)', '== (588, 128)'),
        ('"Two224px views must form native224x448 FLUX.2 current/future token grids"',
         '"Three 224px views (head, chest, wrist) must form 224x672 FLUX.2 current/future token grids (14x42)"'),
        ('{(3, 224, 448), (1, 3, 224, 448)}, "Expected two-view224x448 current image"',
         '{(3, 224, 672), (1, 3, 224, 672)}, "Expected three-view 224x672 current image (head | chest | wrist)"'),
    ],
}


def build_code3():
    record = CODE3 / 'deltaflow-code.json'
    if record.is_file():
        for rel, digest in json.loads(record.read_text())['sha256'].items():
            if h2.sha(CODE3 / rel) != digest:
                raise SystemExit(f'{CODE3 / rel} changed after preparation; rename {CODE3} aside and rerun')
        h2.log('H2 3-camera code copy present and unchanged:', CODE3)
        return
    if CODE3.exists():
        raise SystemExit(f'{CODE3} exists without deltaflow-code.json (incomplete); rename it aside and rerun')
    rec2 = json.loads((SRC2 / 'deltaflow-code.json').read_text())
    for rel, digest in rec2['sha256'].items():
        if h2.sha(SRC2 / rel) != digest:
            raise SystemExit(f'2-camera H2 code copy changed: {SRC2 / rel}')
    tmp = ROOT / f'.HumanWAM-h2w3.tmp{os.getpid()}'
    shutil.copytree(SRC2, tmp, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', 'deltaflow-code.json'))
    for rel, pairs in W3_PATCHES.items():
        f = tmp / rel
        text = f.read_text(encoding='utf-8')
        for old, new in pairs:
            if text.count(old) != 1:
                raise SystemExit(f'{rel}: 3-camera patch anchor found {text.count(old)}x: {old!r}')
            text = text.replace(old, new)
        f.write_text(text, encoding='utf-8')
    files = sorted(set(rec2['sha256']) | set(W3_PATCHES))
    h2.dump(tmp / 'deltaflow-code.json', dict(
        source=str(SRC2), created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        changes=rec2['changes'] + [
            'h2_pour.py: CAMERAS head, chest, wrist (2026-11-13 user rule: every H2 task trains with 3 cameras)',
            'h2_pour_delta.py: three side-by-side 224px views -> 224x672 input, 14x42 = 588 FLUX.2 tokens per image '
            '(only the shape checks / protocol changed; the model code path is the native multi-view one)'],
        sha256={rel: h2.sha(tmp / rel) for rel in files}))
    tmp.rename(CODE3)
    h2.log('H2 3-camera code copy built:', CODE3)


PREPARE_PATCHES = [
    ("if list(h2_pour.CAMERAS) != ['observation.images.head', 'observation.images.chest']:",
     "if tuple(h2_pour.CAMERAS) != CAMS3:"),
    ("raise SystemExit(f'{task}: unexpected batch / accumulation / validation in the H2 config')\n",
     "raise SystemExit(f'{task}: unexpected batch / accumulation / validation in the H2 config')\n"
     "    cfg.update(batch_size=MICRO3, gradient_accumulation_steps=ACC3)  # 3 cameras: 4 x 2 per GPU, global 256\n"),
    ("raw_loader_batches=math.ceil(FRAMES / base.MICRO)", "raw_loader_batches=math.ceil(FRAMES / MICRO3)"),
    ("per_rank_microbatches=u, world_size=base.WORLD, microbatch=base.MICRO,",
     "per_rank_microbatches=u * ACC3, world_size=base.WORLD, microbatch=MICRO3,"),
    ("gradient_accumulation=1),", "gradient_accumulation=ACC3),"),
    ("(3, 2, 224, 448)", "(3, 2, 224, 672)"),
    ("name=f'deltaflow-{task}-40ep-augv2-aligned'", "name=f'deltaflow-{task}-8000steps-augv2-aligned-3cam'"),
]


def install():
    h2.CODE = CODE3
    h2.CAMS3, h2.MICRO3, h2.ACC3 = CAMS3, MICRO3, ACC3
    h2.TASKS['a14'] = dict(data=P / 'a14-unscrew-bulb-v1-b4390fc' / 'data', frames=185638, qwen='compute',
                           source='a14-unscrew-bulb-expert-v1 @b4390fc')
    src = inspect.getsource(h2.prepare)
    for old, new in PREPARE_PATCHES:
        if src.count(old) != 1:
            raise SystemExit(f'prepare(): patch anchor found {src.count(old)}x: {old!r}')
        src = src.replace(old, new)
    exec(compile(src, str(H / 'prepare_deltaflow_h2.py') + ' [3-camera]', 'exec'), h2.__dict__)


if __name__ == '__main__':
    tasks = sys.argv[1:] or ['a18', 'a17', 'a14']
    install()
    for t in tasks:
        if t not in h2.TASKS:
            raise SystemExit(f'unknown H2 task {t}; choose from {list(h2.TASKS)}')
    build_code3()
    for t in tasks:
        R = ROOT / t
        dvf = R / 'data-verified.json'
        if R.exists() and (not dvf.is_file() or json.loads(dvf.read_text()).get('code') != str(CODE3)):
            if (R / 'run/production').exists():
                raise SystemExit(f'{t}: {R} already has a training run; not moving it')
            aside = ROOT / f'{t}.2cam-20261113'
            if aside.exists():
                raise SystemExit(f'{aside} exists; resolve by hand')
            R.rename(aside)
            h2.log(t, '2-camera preparation kept as', aside)
        h2.prepare(t)
        dv = json.loads((R / 'data-verified.json').read_text())
        dv.update(cameras=list(CAMS3), per_gpu_micro_batch=MICRO3, gradient_accumulation=ACC3, global_batch=base.WORLD * MICRO3 * ACC3)
        (R / 'data-verified.json').write_text(json.dumps(dv, indent=2) + '\n')
        print('H2W3_PREPARED', t, flush=True)
