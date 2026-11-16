"""2026-11-15 (H2 training guides hb_training/docs/h2, user list item 4): Delta Flow on H2 tasks with the configuration
of deltaflow/legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/epoch-040 (31/100 on a18), changing only what the
guide changes, with the user's 2026-11-14 Delta-flow decisions (global batch 256 = 32 x 8, no accumulation, 8000 steps,
checkpoints 4000 / 8000, warmup 5 % = 400, cosine to 0):
  same loader imagewam.datasets.h2_pour.H2PourDataset on the task's export (head + chest, 224 x 448), same visual
  augmentation b6-visual-aug-v2 (policy_id renamed per task), no proprio augmentation, same model / losses / optimizer;
  normalization refit on the task (q01 / q99), the task's own Qwen3 instruction cache.
The run's data-verified.json carries recipe 'legacy-a18-e40', which the shared train_deltaflow.py accepts.
Usage: <venv python> prepare_deltaflow_h2legacy.py a17lg"""
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import yaml

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow as base  # noqa: E402
import prepare_deltaflow_h2 as h2  # noqa: E402

OLD = h2.OLD
STEPS, SAVES, WARMUP = 8000, [4000, 8000], 400
TASKS = {
    'a17lg': dict(data=h2.TASKS['a17']['data'], frames=115333, code=h2.CODE, action_dim=75,
                  qwen_from=base.ROOT / 'a17c2' / 'qwen-text', policy_id='a17-h2-rgb-v1',
                  source='humanoid-benchmark/a17-unscrew-bottle-expert-v1 @ 407e6e20db2debc9b77fc2813e3445f549d51a33'),
}
sha, dump, log = base.sha, base.dump, base.log


def prepare(task):
    s = TASKS[task]
    R = base.ROOT / task
    if (R / 'prepared.ok').is_file():
        log(task, 'already prepared, skipping')
        return
    if R.exists():
        aside = base.ROOT / f'{task}.incomplete-{time.strftime("%Y%m%dT%H%M%S")}'
        R.rename(aside)
        log(task, 'half-prepared directory moved to', aside)
    (R / 'run').mkdir(parents=True)
    code = Path(s['code'])
    sys.path.insert(0, str(code / 'src'))
    from imagewam.datasets import h2_pour
    if not h2_pour.__file__.startswith(str(code / 'src')):
        raise SystemExit(f'imported the wrong imagewam: {h2_pour.__file__}')
    if (h2_pour.ACTION_DIM, h2_pour.STATE_DIM, list(h2_pour.CAMERAS)) != (
            s['action_dim'], 78, ['observation.images.head', 'observation.images.chest']):
        raise SystemExit(f'{task}: loader dims / cameras {h2_pour.ACTION_DIM} {h2_pour.STATE_DIM} {h2_pour.CAMERAS}')
    data = Path(s['data'])
    episodes, instruction, names = h2_pour.load_collection(data)
    frames = sum(int(e['length']) for e in episodes)
    if frames != s['frames'] or [e['episode_index'] for e in episodes] != list(range(100)):
        raise SystemExit(f'{task}: {frames} frames / episodes differ from the reviewed collection')
    log(task, 'normalization refit (q01 / q99 over episodes 0-99)')
    h2_pour.fit(data, R / 'normalization.json')
    shutil.copytree(s['qwen_from'], R / 'qwen-text')
    prompt = h2_pour.DEFAULT_PROMPT.format(task=instruction)
    cache = R / 'qwen-text' / f'{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt'
    if not cache.is_file():
        raise SystemExit(f'{task}: instruction cache for this task text missing ({cache.name})')
    u = math.ceil(frames / (base.WORLD * base.MICRO))
    cfg = yaml.safe_load((OLD / 'run/production.yaml').read_text())
    if cfg['batch_size'] != base.MICRO or cfg['gradient_accumulation_steps'] != 1 or cfg['data']['val'] is not None:
        raise SystemExit(f'{task}: unexpected batch / accumulation / validation in the legacy a18 config')
    old_sched = {k: cfg[k] for k in ('num_epochs', 'max_steps', 'warmup_steps', 'save_every', 'keep_latest_weights')}
    cfg.update(output_dir=str(R / 'run/production'), num_epochs=math.ceil(STEPS / u), max_steps=STEPS,
               lr_scheduler_total_steps=STEPS, warmup_steps=WARMUP, save_every=0, save_steps=SAVES, save_at_end=True,
               keep_latest_weights=len(SAVES), keep_latest_state_only=True, resume=None, eval_every=0)
    cfg['wandb'].update(enabled=False, mode='disabled', name=f'deltaflow-{task}-legacy-a18-e40-8k')
    d = cfg['data']['train']
    if d['_target_'] != 'imagewam.datasets.h2_pour.H2PourDataset' or d['image_augmentation'].get('version') != 'b6-visual-aug-v2':
        raise SystemExit(f'{task}: legacy config is not H2PourDataset + b6-visual-aug-v2')
    old_policy = d['image_augmentation'].get('policy_id')
    d.update(dataset_dir=str(data), stats_path=str(R / 'normalization.json'), qwen_text_cache_dir=str(R / 'qwen-text'))
    d['image_augmentation']['policy_id'] = s['policy_id']
    m = cfg['model']
    if (m['expected_action_dim'], m['action_dit_config']['action_dim'], len(m['loss']['action_a2a']['source_noise_dim_mask']),
            m['proprio_dim'], m['action_state_dim'], m['expected_action_horizon']) != (s['action_dim'],) * 3 + (78, 78, 16):
        raise SystemExit(f'{task}: model dims differ from action{s["action_dim"]} / state78 / horizon16')
    cfg['hb_legacy_contract'] = dict(
        task=task, recipe='legacy-a18-e40', source_run=str(OLD), source_config=str(OLD / 'run/production.yaml'),
        changed=dict(dataset=s['source'], normalization='refit on this task', qwen_text_cache='this task text',
                     image_augmentation_policy_id=[old_policy, s['policy_id']],
                     schedule=dict(old=old_sched, new=dict(max_steps=STEPS, warmup_steps=WARMUP, save_steps=SAVES,
                                                          cosine_to_zero_at=STEPS))),
        unchanged='loader, b6-visual-aug-v2 effects, no proprio augmentation, model / losses / optimizer, global batch 256 '
                  '(32 x 8, no accumulation)',
        user='2026-11-15 H2 training guides + user: Delta Flow 256 / 8000 steps, publish 4000 and 8000')

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and value.startswith(str(OLD) + '/') and 'hb_legacy_contract' not in value:
            if not (value.endswith('production.yaml') or value == str(OLD)):
                raise SystemExit(f'{task}: config still points into the old experiment: {value}')
    walk({k: v for k, v in cfg.items() if k != 'hb_legacy_contract'})
    (R / 'run/production.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    dump(R / 'data-verified.json', dict(
        task=task, recipe='legacy-a18-e40', dataset_dir=str(data), dataset_source=s['source'],
        total_episodes_in_collection=100, episodes=100, train_episode_ids=list(range(100)), shelved_episode_ids=[],
        frames=frames, updates_per_epoch=u, epochs=None, fixed_steps=True, epochs_equivalent=round(STEPS / u, 2),
        max_steps=STEPS, warmup_steps=WARMUP, save_steps=SAVES, control_hz=30, recording_hz=30,
        statistics_sha256=sha(R / 'normalization.json'), augmentation='b6-visual-aug-v2 (legacy a18 e40 recipe)',
        proprio_augmentation='none (legacy a18 e40 recipe, H2 training guide)', joint_velocity=False,
        state_dim=78, action_dim=s['action_dim'], cameras=list(h2_pour.CAMERAS), per_gpu_micro_batch=base.MICRO,
        gradient_accumulation=1, global_batch=base.MICRO * base.WORLD, code=str(code),
        code_record_sha256=sha(code / 'deltaflow-code.json'), qwen_cache=f'copied from {s["qwen_from"]}'))
    log(task, 'loading real augmented samples (CPU)')
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    import torch
    ds = instantiate(OmegaConf.load(R / 'run/production.yaml').data.train)
    if len(ds) != frames or list(ds.episode_ids) != list(range(100)):
        raise SystemExit(f'{task}: dataset length {len(ds)} differs from {frames}')
    shapes = []
    for i in (0, 20, frames // 2, frames - 1):
        x = ds[i]
        shapes.append([tuple(x['video'].shape), tuple(x['action'].shape), tuple(x['proprio'].shape)])
        if (tuple(x['video'].shape) != (3, 2, 224, 448) or tuple(x['action'].shape) != (16, s['action_dim'])
                or tuple(x['proprio'].shape) != (16, 78) or not torch.isfinite(x['action']).all()
                or not torch.isfinite(x['proprio']).all() or not torch.isfinite(x['video']).all()):
            raise SystemExit(f'{task}: sample {i} schema check failed: {shapes[-1]}')
    dump(R / 'prepared.ok', dict(frames=frames, updates_per_epoch=u, max_steps=STEPS, save_steps=SAVES, warmup=WARMUP,
                                 recipe='legacy-a18-e40', sample_shapes=shapes))
    log(task, 'PREPARED', json.dumps(dict(frames=frames, updates_per_epoch=u, max_steps=STEPS, save_steps=SAVES,
                                          warmup=WARMUP, action_dim=s['action_dim'], shapes=shapes[0])))


if __name__ == '__main__':
    for t in sys.argv[1:]:
        if t not in TASKS:
            raise SystemExit(f'unknown task {t}; choose from {list(TASKS)}')
        prepare(t)
