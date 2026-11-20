"""[2026-11-20 copy of prepare_deltaflow_b6b5.py: B6 v6 = fixed first 500 of 542 episodes (user rule 11-18); the rest unchanged]
2026-11-15 17:xx user: Delta Flow on
  b6v6: B6 v6 (HF b6-pedal-bin @ad2f5bf0, repo root, all 542 episodes) as a weights-only CONTINUATION of the B6 v5
        Delta-flow run (deltaflow-runs/b6 step_010200: first 100 v5 episodes, 40 epochs, 256 x 10200), 8000 steps
        (6.2 epochs of v6), peak lr 3e-5 (= 0.3 x the original 1e-4, user rule), warmup 400 (5 %), cosine, checkpoints
        6000 / 8000; normalization recomputed from v6 (as for the GR00T / Psi-Zero continuations);
  b5:   B5 chair push v1 (HF b5-chair-push-expert-v1 @b85b7e9, all 100 episodes) from the Delta-flow initialization.
Both on the all-episodes G1 tree HumanWAM-allep with the unchanged G1 recipe of prepare_deltaflow.py (aug v2, 44-D
proprio without velocity, proprio aligned with GR00T A20, 32 GPUs x 8 = 256, no accumulation). Usage: <task>"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow as base  # noqa: E402
import prepare_deltaflow_a10v3 as allep  # noqa: E402  (build_all: the HumanWAM-allep tree, verified)

SPEC = {
    'b6v6': dict(data=base.P / 'b6-pedal-bin-v6-ad2f5bf/data', n=500, total=542, frames=305667,
                 source='humanoid-benchmark/b6-pedal-bin @ad2f5bf0 (v6 at the repo root, fixed first 500 of 542 episodes)'),
    'b5': dict(data=base.P / 'b5-chair-push-v1-b85b7e9/data', n=100, frames=43075,
               source='humanoid-benchmark/b5-chair-push-expert-v1 @b85b7e9 (all 100 episodes)'),
    # 2026-11-15 23:xx user: Delta Flow aligned with the newest data GR00T trains on (A7 v4 replaces A7 v2)
    'a7v4': dict(data=base.P / 'a7-cabinet-close-v4-90cc2b5/data', n=100, frames=26693,
                 source='humanoid-benchmark/a7-cabinet-close-expert-v4 @90cc2b5 (all 100 episodes)'),
    'a3v4': dict(data=base.P / 'a3-fridge-open-v4-6393d4a/data', n=125, frames=37226,
                 source='humanoid-benchmark/a3-fridge-open-expert-v4 @6393d4a (all 125 episodes)'),
}
INIT = base.ROOT / 'b6' / 'run/production/checkpoints/weights/step_010200.pt'

if __name__ == '__main__':
    task = sys.argv[1]
    s = SPEC[task]
    N = s['n']
    base.TASKS[task] = dict(data=s['data'], old=None, control_hz=50, total_episodes=s.get('total', N), source=s['source'])
    base.build_code = allep.build_all
    base.EPISODES = N
    sys.argv = [sys.argv[0], task]
    base.main()
    R = base.ROOT / task
    d = json.loads((R / 'data-verified.json').read_text())
    if (d['train_episode_ids'] != list(range(N)) or d['shelved_episode_ids'] != list(range(N, s.get('total', N))) or d['frames'] != s['frames']
            or d['max_steps'] != 8000 or d['code'] != str(allep.CODE_ALL) or d['proprio_dim'] != 44 or d['joint_velocity']):
        raise SystemExit(f'{task}: unexpected receipt {d}')
    yp = R / 'run/production.yaml'
    y = yp.read_text()
    if y.count('train_episode_ids: 0-99\n') == 1:
        y = y.replace('train_episode_ids: 0-99\n', f'train_episode_ids: 0-{N - 1}\n')
    if task == 'b6v6':
        if y.count('\nlearning_rate: 0.0001\n') == 1:
            y = y.replace('\nlearning_rate: 0.0001\n', '\nlearning_rate: 3.0e-05\n')
        elif y.count('\nlearning_rate: 3.0e-05\n') != 1:
            raise SystemExit('b6v6: learning_rate line not found')
    yp.write_text(y)
    import yaml
    c = yaml.safe_load(yp.read_text())
    if c['batch_size'] != 8 or c['gradient_accumulation_steps'] != 1 or c['max_steps'] != 8000 or c['warmup_steps'] != 400:
        raise SystemExit(f"{task}: config batch {c['batch_size']} accum {c['gradient_accumulation_steps']} warmup {c['warmup_steps']}")
    if task == 'b6v6' and abs(float(c['learning_rate']) - 3e-5) > 1e-12:
        raise SystemExit(f"b6v6: learning_rate {c['learning_rate']}")
    d.update(episode_rule=('2026-11-18 user: B6 v6 only the fixed first 500 episodes (0-499), 500-541 shelved' if task == 'b6v6' else '2026-11-14 user: all episodes, no forced 100'), per_gpu_micro_batch=base.MICRO,
             gradient_accumulation=1, global_batch=base.MICRO * base.WORLD, dataset_source=s['source'])
    if task == 'b6v6':
        if not INIT.is_file():
            raise SystemExit(f'missing {INIT}')
        d.update(init_weights=dict(path=str(INIT), bytes=INIT.stat().st_size),
                 continued_from='deltaflow-runs/b6 step_010200 (B6 v5, first 100 episodes, 65,127 frames, 40 epochs, '
                                '256 x 10200, peak lr 1e-4)',
                 continuation='weights only (fresh optimizer, LR schedule and global step); peak lr 3e-5 = 0.3 x 1e-4, '
                              'warmup 400 (5 %), cosine to 0 over 8000 steps (6.2 epochs of v6); normalization '
                              'recomputed from the v6 data', learning_rate=3e-5)
    (R / 'data-verified.json').write_text(json.dumps(d, indent=2))
    print('B6B5_DF_OK', task, json.dumps({k: d.get(k) for k in ('frames', 'episodes', 'updates_per_epoch', 'max_steps',
                                                              'save_steps', 'warmup_steps', 'global_batch', 'code',
                                                              'init_weights', 'learning_rate')}), flush=True)
