"""2026-11-14 20:3x user: every H2 task (A14 included) trains with head + chest only, no wrist.
Delta Flow A14 on the unchanged 2-camera H2 code tree (HumanWAM-h2, same as a18c2 / a17c2): per-GPU 8 x accumulation 1
= 256 on 32 GPUs; 16000 steps (2026-11-13 user: C6 / A14 train 2x the basket budget), checkpoints 14000 / 16000,
warmup 5 %. Replaces the 3-camera a14 preparation (per-GPU 4 x accumulation 2, forbidden since 2026-11-14)."""
import json
import math
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow as base  # noqa: E402
import prepare_deltaflow_h2 as h2  # noqa: E402

LONG = '2026-11-13 user: C6 / A14 train 2x the basket budget (16000 steps)'
CAMS = ['observation.images.head', 'observation.images.chest']
h2.TASKS['a14c2'] = dict(data=h2.P / 'a14-unscrew-bulb-v1-b4390fc' / 'data', frames=185638, qwen='compute',
                         source='a14-unscrew-bulb-expert-v1 @b4390fc, 2 cameras head + chest (2026-11-14)')


def long_schedule(u):
    total = 2 * base.FIXED_STEPS
    save_steps = [total - base.CKPT_EVERY * k for k in range(base.KEEP_CKPTS - 1, -1, -1)]
    return total, save_steps, round(base.WARMUP_FRAC * total), math.ceil(total / u)


base.fixed_schedule = long_schedule

if __name__ == '__main__':
    h2.build_code()
    h2.prepare('a14c2')
    R = h2.ROOT / 'a14c2'
    d = json.loads((R / 'data-verified.json').read_text())
    if d['max_steps'] != 16000 or d['save_steps'] != [14000, 16000] or d['state_dim'] != 78 or d['action_dim'] != 75:
        raise SystemExit(f'a14c2: unexpected receipt {d}')
    cfg = (R / 'run/production.yaml').read_text()
    import yaml
    c = yaml.safe_load(cfg)
    if c['batch_size'] != 8 or c['gradient_accumulation_steps'] != 1 or c['max_steps'] != 16000:
        raise SystemExit(f"a14c2: config batch {c['batch_size']} accum {c['gradient_accumulation_steps']} steps {c['max_steps']}")
    if list(c['wholebody_contract']['cameras']) != CAMS:
        raise SystemExit(f"a14c2: cameras {c['wholebody_contract']['cameras']}")
    d.update(long_task_2x=LONG, code=str(h2.CODE), cameras=CAMS, per_gpu_micro_batch=base.MICRO,
             gradient_accumulation=1, global_batch=base.MICRO * base.WORLD)
    (R / 'data-verified.json').write_text(json.dumps(d, indent=2))
    print('A14C2_DF_OK', json.dumps({k: d[k] for k in ('frames', 'max_steps', 'warmup_steps', 'save_steps', 'cameras',
                                                       'per_gpu_micro_batch', 'gradient_accumulation', 'global_batch',
                                                       'code')}), flush=True)
