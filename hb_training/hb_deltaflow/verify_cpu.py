"""CPU verification of the Delta-flow 40-epoch package (run on a node after prepare_deltaflow.py; no GPU used).

    $P/venv/bin/python $G/hb_deltaflow/verify_cpu.py                 # aug checks + b6 a3 a7
    $P/venv/bin/python $G/hb_deltaflow/verify_cpu.py b6 --samples 48
    $P/venv/bin/python $G/hb_deltaflow/verify_cpu.py --aug-only      # before prepare: hb_aug_v2 from $G/hb_groot
Checks
  1. hb_aug_v2 parameter statistics: each of the 15 effects is the 1.5x boost ~1/15 of the time, trigger rates match
     the spec, 1x and 1.5x ranges are respected and the boosted range is actually reached.
  2. hb_aug_v2.apply per effect in isolation: every effect changes the observation frame; colour / rotation-crop /
     light / white balance also change the t+16 target, sensor/optics effects leave the target untouched.
  3. per task: manifest / statistics cover exactly episodes 0-99 (B6 shelves 100-148), YAML has the
     40-epoch schedule and epoch 10/20/30/40 saves and NO joint velocity anywhere (user rule 2026-11-10), then one
     real augmented batch: shapes, labels unchanged, target frame = colour/geometry/light/WB only, proprio 44-D
     (G1: body 29 + hands 12 + gravity 3) / 78-D (H2: 75 joints + gravity 3) with untouched hands/gravity when not
     delayed.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

H = Path(__file__).resolve().parent
G = H.parent
CODE = G / 'deltaflow-runs' / 'HumanWAM'
CODE_H2 = G / 'deltaflow-runs' / 'HumanWAM-h2'
ROOT = G / 'deltaflow-runs'
CONTROL_HZ = {'b6': 50, 'a3': 30, 'a7': 30}
TOTAL_EPISODES = {'b6': 149, 'a3': 100, 'a7': 100}
# src/imagewam/wholebody_delta.py of the 44-D reference run $P/real30-from-sim60-20261108/HumanWAM
REF_DELTA_SHA256 = '523244263496679f7046035f5df3d34590b7086e25173938d8a175dad963b9f3'
FAILURES = []


def check(condition, message):
    print(('  ok   ' if condition else '  FAIL ') + message, flush=True)
    if not condition:
        FAILURES.append(message)


def velocity_entries(value, path='cfg'):
    """Config entries that would add joint velocity (non-empty velocity_* settings or velocity file references)."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if 'velocity' in str(key).lower() and item not in ([], None, False, 0):
                found.append(f'{path}.{key}')
            found += velocity_entries(item, f'{path}.{key}')
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found += velocity_entries(item, f'{path}[{i}]')
    elif isinstance(value, str) and 'velocity' in value.lower() and 'no velocity' not in value.lower():
        found.append(path)
    return found


SENSOR = ('noise', 'gaussian_blur', 'jpeg', 'motion_blur', 'sharpen', 'defocus')
PROB = {'noise': .5, 'gaussian_blur': .15, 'jpeg': .35, 'motion_blur': .15, 'sharpen': .15, 'defocus': .10,
        'light_shadow': .30, 'wb_channel': .30}


def active(p, name):
    value = p[name]
    return value is not None and value != 0.0


def check_parameters(A, n=15000):
    print('[1] hb_aug_v2 parameter statistics, n=%d' % n)
    check(len(A.EFFECTS) == 15 and A.VERSION == 'hb-aug-v2-20261109', f'15 effects, version {A.VERSION}')
    rng = np.random.default_rng(1234)
    draws = [A.sample(rng, 2, 224) for _ in range(n)]
    counts = {e: sum(p['boost'] == e for p in draws) for e in A.EFFECTS}
    band = 5 * math.sqrt(n / 15)
    check(all(abs(c - n / 15) < band for c in counts.values()), f'boost uniform over 15 effects: {counts}')
    for name, prob in PROB.items():
        rate = sum(active(p, name) for p in draws) / n
        check(abs(rate - prob) < 0.03, f'{name} trigger rate {rate:.3f} (spec {prob})')

    def span(name, value_fn, lo1, hi1, lo15, hi15):
        normal = [v for p in draws if p['boost'] != name for v in value_fn(p)]
        boosted = [v for p in draws if p['boost'] == name for v in value_fn(p)]
        ok = (normal and boosted and lo1 - 1e-9 <= min(normal) and max(normal) <= hi1 + 1e-9
              and lo15 - 1e-9 <= min(boosted) and max(boosted) <= hi15 + 1e-9
              and (min(boosted) < lo1 - 1e-6 or max(boosted) > hi1 + 1e-6))
        check(bool(ok), f'{name}: x1 in [{lo1},{hi1}], x1.5 in [{lo15},{hi15}] and beyond x1 '
                        f'(seen x1 {min(normal):.4g}..{max(normal):.4g}, x1.5 {min(boosted):.4g}..{max(boosted):.4g})')

    span('exposure', lambda p: [p['exposure']], -.22, .22, -.33, .33)
    span('contrast', lambda p: [p['contrast']], .88, 1.12, .82, 1.18)
    span('saturation', lambda p: [p['saturation']], .76, 1.24, .64, 1.36)
    span('hue', lambda p: [p['hue']], -.015, .015, -.0225, .0225)
    span('gamma', lambda p: [p['gamma']], .89, 1.11, .835, 1.165)
    span('temperature', lambda p: [p['temperature']], -.066, .066, -.099, .099)
    span('rotation_crop', lambda p: [a for a, _ in p['geometry']], -1.5, 1.5, -2.25, 2.25)
    span('rotation_crop', lambda p: [s - 1 - abs(a) * .015 for a, s in p['geometry']], 0, .02, 0, .03)
    span('noise', lambda p: [p['noise']] if p['noise'] else [], 4 / 255, 4 / 255, 6 / 255, 6 / 255)
    span('gaussian_blur', lambda p: [p['gaussian_blur']] if p['gaussian_blur'] else [], .25, .65, .375, .975)
    span('jpeg', lambda p: [p['jpeg']] if p['jpeg'] is not None else [], 70, 95, 55, 92)
    span('motion_blur', lambda p: [p['motion_blur'][0]] if p['motion_blur'] else [], 3, 7, 4.5, 10.5)
    span('sharpen', lambda p: [p['sharpen']] if p['sharpen'] else [], .2, .6, .3, .9)
    span('defocus', lambda p: [p['defocus']] if p['defocus'] else [], 1, 2, 1.5, 3)
    span('light_shadow', lambda p: [b[3] for cam in (p['light_shadow'] or []) for b in cam], .7, 1.3, .55, 1.45)
    span('wb_channel', lambda p: [g for cam in (p['wb_channel'] or []) for g in cam if g != 1.0], .9, 1.1, .85, 1.15)
    angles = [p['motion_blur'][1] for p in draws if p['motion_blur']]
    check(min(angles) >= 0 and max(angles) <= 180, 'motion blur direction in 0-180 deg')
    blobs = [cam for p in draws if p['light_shadow'] for cam in p['light_shadow']]
    check(all(1 <= len(c) <= 2 and all(.15 <= b[2] <= .35 for b in c) for c in blobs),
          'light: 1-2 patches per camera, size 15-35 % of the frame')
    wb = [cam for p in draws if p['wb_channel'] for cam in p['wb_channel']]
    check(all(sum(g != 1.0 for g in cam) <= 1 for cam in wb), 'white balance: one channel per camera')
    geometry_differs = sum(p['geometry'][0] != p['geometry'][1] for p in draws) / n
    check(geometry_differs > .99, 'rotation/crop drawn independently per camera')


def identity(views=2):
    return {'boost': None, 'exposure': 0.0, 'contrast': 1.0, 'saturation': 1.0, 'hue': 0.0, 'gamma': 1.0,
            'temperature': 0.0, 'geometry': [(0.0, 1.0)] * views, 'noise': 0.0, 'gaussian_blur': 0.0, 'jpeg': None,
            'motion_blur': None, 'sharpen': 0.0, 'defocus': 0.0, 'light_shadow': None, 'wb_channel': None}


def test_image():
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 224), torch.linspace(0, 1, 224), indexing='ij')
    gen = torch.Generator().manual_seed(7)
    frames = []
    for t in range(2):
        cams = []
        for v in range(2):
            base = torch.stack([0.2 + 0.6 * xx, 0.2 + 0.6 * yy, 0.5 + 0.3 * torch.sin(12 * xx + 7 * yy + t + v)])
            checker = ((torch.floor(xx * 16) + torch.floor(yy * 16)) % 2) * 0.15
            cams.append((base * 0.8 + checker + 0.05 * torch.rand(3, 224, 224, generator=gen)).clamp(0, 1))
        frames.append(torch.stack(cams))
    return torch.stack(frames)                                     # [T=2, V=2, 3, 224, 224]


def check_apply(A):
    print('[2] hb_aug_v2.apply: every effect fires, observation-only vs target rules')
    x = test_image()
    obs = torch.tensor([True, False])
    ref = A.apply(x, identity(), obs)
    check((ref - x).abs().max().item() < 2e-3, 'identity parameters leave the frames unchanged')
    settings = {
        'exposure': {'exposure': .3}, 'contrast': {'contrast': 1.15}, 'saturation': {'saturation': 1.3},
        'hue': {'hue': .02}, 'gamma': {'gamma': 1.15}, 'temperature': {'temperature': .09},
        'rotation_crop': {'geometry': [(2.0, 1.05), (-2.0, 1.04)]}, 'noise': {'noise': 6 / 255},
        'gaussian_blur': {'gaussian_blur': .9}, 'jpeg': {'jpeg': 55}, 'motion_blur': {'motion_blur': (9.0, 30.0)},
        'sharpen': {'sharpen': .8}, 'defocus': {'defocus': 2.5},
        'light_shadow': {'light_shadow': [[(.5, .5, .25, 1.4)], [(.3, .6, .2, .6)]]},
        'wb_channel': {'wb_channel': [[1.0, 1.12, 1.0], [0.88, 1.0, 1.0]]},
    }
    check(sorted(settings) == sorted(A.EFFECTS), 'all 15 effects exercised')
    for name, change in settings.items():
        p = dict(identity(), **change)
        y = A.apply(x, p, obs, noise_seed=3)
        d_obs = [(y[0, v] - ref[0, v]).abs().mean().item() for v in range(2)]
        d_tgt = [(y[1, v] - ref[1, v]).abs().max().item() for v in range(2)]
        fires = min(d_obs) > 5e-4
        if name in SENSOR:
            check(fires and max(d_tgt) < 1e-6, f'{name}: observation changed {min(d_obs):.4f}, target untouched')
        else:
            check(fires and min((y[1, v] - ref[1, v]).abs().mean().item() for v in range(2)) > 5e-4,
                  f'{name}: observation and t+16 target both changed')
    p = dict(identity(), exposure=.3, saturation=1.3)
    y = A.apply(x, p, obs)
    same_colour = all(torch.allclose(y[t, v], A.apply(x[t:t + 1, v:v + 1], dict(p, geometry=[(0.0, 1.0)]),
                                                      torch.tensor([t == 0]))[0, 0], atol=1e-5)
                      for t in range(2) for v in range(2))
    check(same_colour, 'one colour parameter set applied identically to both cameras and both times')


def check_task(task, A, samples):
    import yaml
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from imagewam.datasets.wholebody import WholeBodyLeRobotDataset
    R = ROOT / task
    print(f'[3] task {task}: {R}')
    check((R / 'prepared.ok').is_file(), 'prepared.ok present')
    receipt = json.loads((R / 'data-verified.json').read_text())
    u, frames = receipt['updates_per_epoch'], receipt['frames']
    train = list(range(100))
    shelved = list(range(100, TOTAL_EPISODES[task]))
    manifest = json.loads((R / 'split-manifest.json').read_text())
    stats = json.loads((R / 'stats/dataset_stats.json').read_text())
    check(manifest['train_episode_ids'] == train and manifest['validation_episode_ids'] == shelved,
          f'split manifest trains 0-99, shelves {len(shelved)} episodes')
    check(stats['train_episode_ids'] == train and stats['heldout_episode_ids'] == shelved
          and stats['train_frames'] == frames and stats['train_split_mode'] == 'first_100',
          f'normalization statistics fit on episodes 0-99 only ({frames} frames)')
    check(not (R / 'stats/velocity.npz').exists() and not list(R.rglob('*velocity*')), 'no velocity cache / statistics files')
    cfg = yaml.safe_load((R / 'run/production.yaml').read_text())
    left = velocity_entries(cfg)
    check(not left, f'config has no joint-velocity setting ({left or "none"})')
    d = cfg['data']['train']
    saves = [e * u for e in (10, 20, 30, 40)]
    check(cfg['num_epochs'] == 40 and cfg['max_steps'] == cfg['lr_scheduler_total_steps'] == 40 * u
          and cfg['warmup_steps'] == 2 * u and cfg['save_steps'] == saves and cfg['save_every'] == 0
          and cfg['keep_latest_weights'] == 4 and cfg['keep_latest_state_only'] is True and cfg['save_at_end'] is True,
          f'schedule: {u} updates/epoch, max_steps {40 * u}, warmup {2 * u}, saves {saves}, newest state only')
    check(d['proprio_augmentation']['control_hz'] == CONTROL_HZ[task] and d['proprio_augmentation']['recording_hz'] == 30,
          f'control {CONTROL_HZ[task]} Hz / record 30 Hz')
    check(d['_target_'].endswith('WholeBodyHBAugV2LeRobotDataset')
          and d['image_augmentation']['version'] == A.VERSION, 'dataset class uses hb_aug_v2')
    check(cfg['model']['proprio_dim'] == cfg['model']['action_state_dim'] == 44 and cfg['model']['expected_action_dim'] == 76,
          'model 44-D state (no velocity) / 76-D action, as reference real30-from-sim60-20261108')
    check(d['_target_'].endswith('WholeBodyHBAugV2LeRobotDataset') and not d['proprio_augmentation'].get('velocity_channels'),
          'loader = WholeBodyRobust-based HB aug v2, velocity_channels empty')
    # the model factory itself (2026-11-10: config + data said 44 but the factory of the B6 60-epoch copy required 73)
    import hashlib
    import inspect
    import re
    import imagewam.wholebody_delta as model_module
    src = Path(model_module.__file__).read_bytes()
    sig = inspect.signature(model_module.create_imagewam_flux2_wholebody_delta)
    check(cfg['model']['_target_'] == 'imagewam.wholebody_delta.create_imagewam_flux2_wholebody_delta'
          and sig.parameters['action_state_dim'].default == 44 and 'state44' in model_module.SCHEMA_ID
          and b'(None, 44)' in src and re.search(rb'\b73\b', src) is None
          and hashlib.sha256(src).hexdigest() == REF_DELTA_SHA256,
          'model factory wholebody_delta.py = state44 copy of real30-from-sim60-20261108 (no 73 anywhere)')

    dataset = instantiate(OmegaConf.load(R / 'run/production.yaml').data.train)
    check(len(dataset) == frames and list(dataset.episode_ids) == train, f'dataset = episodes 0-99, {len(dataset)} samples')
    rng = np.random.default_rng(99)
    index = sorted(set(rng.integers(0, len(dataset), samples).tolist()) | {0})
    batch, boosts, target_ok, obs_ok, label_ok, hands_ok, episodes = [], [], 0, 0, 0, 0, set()
    not_delayed = 0
    for i in index:
        sample = dataset[i]
        info = dataset.last_augmentation
        base = WholeBodyLeRobotDataset.__getitem__(dataset, i)
        p = info['visual']
        boosts.append(p['boost'])
        episodes.add(sample['episode_index'])
        views = torch.stack((((base['video'].permute(1, 0, 2, 3) + 1) * .5).clamp(0, 1)).chunk(2, dim=-1), dim=1)
        clean = A.apply(views, p, torch.tensor([False, False]))
        got = torch.stack((((sample['video'].permute(1, 0, 2, 3) + 1) * .5)).chunk(2, dim=-1), dim=1)
        target_ok += bool(torch.allclose(got[1], clean[1], atol=1e-4))
        sensor = any(active(p, k) for k in SENSOR)
        d0 = (got[0] - clean[0]).abs().max().item()
        obs_ok += (d0 > 1e-6) if sensor else (d0 < 1e-4)
        label_ok += bool(torch.equal(sample['action'], base['action']))
        if info['numeric']['delay_recorded_frames'] == 0:
            not_delayed += 1
            hands_ok += bool(torch.allclose(sample['proprio'][:, 29:44], base['proprio'][:, 29:44])
                             and not torch.allclose(sample['proprio'][:, :29], base['proprio'][:, :29]))
        batch.append(sample)
    n = len(index)
    check(target_ok == n, f't+16 target = colour/geometry/light/WB only, no sensor effects ({target_ok}/{n})')
    check(obs_ok == n, f'observation frame gets the sensor/optics effects when drawn ({obs_ok}/{n})')
    check(label_ok == n, f'action labels identical to the un-augmented loader ({label_ok}/{n})')
    check(hands_ok == not_delayed, f'undelayed samples: hands/gravity untouched, body joints perturbed ({hands_ok}/{not_delayed})')
    check(max(episodes) < 100, f'samples only from episodes < 100 (max {max(episodes)})')
    check(len(set(boosts)) >= min(8, n // 3), f'boosted effects seen: {sorted(set(boosts))}')
    check(all(s['proprio'].shape[-1] == 44 for s in batch), 'every sample proprio is 44-D (no velocity)')
    b = default_collate(batch[:8])
    k = len(batch[:8])
    shapes = {key: tuple(b[key].shape) for key in ('video', 'action', 'past_action', 'proprio', 'text_hidden_states')}
    check(shapes == {'video': (k, 3, 2, 224, 448), 'action': (k, 16, 76), 'past_action': (k, 16, 76),
                     'proprio': (k, 16, 44), 'text_hidden_states': (k, 128, 7680)}, f'batch shapes {shapes}')
    check(bool(torch.isfinite(b['proprio']).all()) and b['video'].min() >= -1 and b['video'].max() <= 1,
          'finite proprio, video in [-1,1]')


def check_task_h2(A, samples):
    """A18 / H2: 75-D action, 78-D proprio (75 joints + gravity 3, NO velocity), cameras head + chest."""
    import yaml
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from imagewam.datasets.h2_pour import CAMERAS, H2PourDataset
    import imagewam.h2_pour_delta as model_module
    R = ROOT / 'a18'
    print(f'[3] task a18 (H2): {R}')
    check((R / 'prepared.ok').is_file(), 'prepared.ok present')
    receipt = json.loads((R / 'data-verified.json').read_text())
    u, frames = receipt['updates_per_epoch'], receipt['frames']
    check(frames == 72752 and u == 285 and receipt['train_episode_ids'] == list(range(100)),
          f'A18: 100 episodes, {frames} frames, {u} updates/epoch')
    check(not list(R.rglob('*velocity*')), 'no velocity cache / statistics files')
    cfg = yaml.safe_load((R / 'run/production.yaml').read_text())
    d, m = cfg['data']['train'], cfg['model']
    left = velocity_entries(cfg)
    check(not left and not d['proprio_augmentation'].get('velocity_channels'),
          f'config has no joint-velocity setting ({left or "none"})')
    saves = [e * u for e in (10, 20, 30, 40)]
    check(cfg['num_epochs'] == 40 and cfg['max_steps'] == cfg['lr_scheduler_total_steps'] == 40 * u
          and cfg['warmup_steps'] == 2 * u and cfg['save_steps'] == saves and cfg['save_every'] == 0
          and cfg['keep_latest_weights'] == 4 and cfg['keep_latest_state_only'] is True and cfg['save_at_end'] is True,
          f'schedule: max_steps {40 * u}, warmup {2 * u}, saves {saves}, newest state only')
    pa = d['proprio_augmentation']
    check(d['_target_'].endswith('H2HBAugV2Dataset') and d['image_augmentation']['version'] == A.VERSION
          and pa['affected_angle_channels'] == list(range(31)) and pa['control_hz'] == pa['recording_hz'] == 30,
          'H2 dataset class, aug v2, bias/noise on state[0:31], 30 Hz delay conversion')
    check('h2_pour_delta' in m['_target_'] and m['proprio_dim'] == m['action_state_dim'] == 78
          and m['expected_action_dim'] == 75, f'model: original H2 h2_pour_delta, state 78 (no velocity) / action 75')
    sig = __import__('inspect').signature(model_module.create_imagewam_flux2_wholebody_delta)
    check(sig.parameters['action_state_dim'].default == 78, 'h2_pour_delta factory default state 78')
    check(list(CAMERAS) == ['observation.images.head', 'observation.images.chest'], f'cameras {list(CAMERAS)}')

    dataset = instantiate(OmegaConf.load(R / 'run/production.yaml').data.train)
    check(len(dataset) == frames, f'dataset {len(dataset)} samples')
    rng = np.random.default_rng(99)
    index = sorted(set(rng.integers(0, len(dataset), samples).tolist()) | {0})
    batch, boosts, target_ok, obs_ok, label_ok, hands_ok, not_delayed = [], [], 0, 0, 0, 0, 0
    for i in index:
        sample = dataset[i]
        info = dataset.last_augmentation
        base = H2PourDataset.__getitem__(dataset, i)          # parent path: augmentation disabled, 78-D state
        p = info['visual']
        boosts.append(p['boost'])
        views = torch.stack((((base['video'].permute(1, 0, 2, 3) + 1) * .5).clamp(0, 1)).chunk(2, dim=-1), dim=1)
        clean = A.apply(views, p, torch.tensor([False, False]))
        got = torch.stack((((sample['video'].permute(1, 0, 2, 3) + 1) * .5)).chunk(2, dim=-1), dim=1)
        target_ok += bool(torch.allclose(got[1], clean[1], atol=1e-4))
        d0 = (got[0] - clean[0]).abs().max().item()
        obs_ok += (d0 > 1e-6) if any(active(p, k) for k in SENSOR) else (d0 < 1e-4)
        label_ok += bool(torch.equal(sample['action'], base['action']))
        numeric = info['numeric']
        if numeric['delay_recorded_frames'] == 0:
            not_delayed += 1
            hands_ok += bool(torch.allclose(sample['proprio'][:, 31:78], base['proprio'][:, 31:78])
                             and not torch.allclose(sample['proprio'][:, :31], base['proprio'][:, :31]))
        batch.append(sample)
    n = len(index)
    check(target_ok == n, f't+16 target = colour/geometry/light/WB only ({target_ok}/{n})')
    check(obs_ok == n, f'observation frame gets the sensor/optics effects when drawn ({obs_ok}/{n})')
    check(label_ok == n, f'action labels identical to the un-augmented H2 loader ({label_ok}/{n})')
    check(all(s['proprio'].shape[-1] == 78 for s in batch), 'every sample proprio is 78-D (no velocity)')
    check(hands_ok == not_delayed, f'undelayed: hands/gravity untouched, 31 body joints perturbed ({hands_ok}/{not_delayed})')
    check(len(set(boosts)) >= min(8, n // 3), f'boosted effects seen: {sorted(set(boosts))}')
    b = default_collate(batch[:8])
    k = len(batch[:8])
    shapes = {key: tuple(b[key].shape) for key in ('video', 'action', 'past_action', 'proprio', 'proprio_dim_is_pad')}
    check(shapes == {'video': (k, 3, 2, 224, 448), 'action': (k, 16, 75), 'past_action': (k, 16, 75),
                     'proprio': (k, 16, 78), 'proprio_dim_is_pad': (k, 78)}, f'batch shapes {shapes}')
    check(bool(torch.isfinite(b['proprio']).all()) and b['video'].min() >= -1 and b['video'].max() <= 1,
          'finite proprio, video in [-1,1]')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('tasks', nargs='*', default=['b6', 'a3', 'a7', 'a18'])
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--aug-only', action='store_true')
    parser.add_argument('--skip-aug', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    torch.set_num_threads(4)
    g1 = [t for t in args.tasks if t != 'a18']
    h2_rc = 0
    if 'a18' in args.tasks and g1 and not args.aug_only:
        # The H2 code tree has its own imagewam package: check A18 in a separate interpreter.
        import subprocess
        h2_rc = subprocess.call([sys.executable, __file__, 'a18', '--samples', str(args.samples), '--skip-aug'])
    h2_only = args.tasks == ['a18'] and not args.aug_only
    if args.aug_only:
        sys.path.insert(0, str(G / 'hb_groot'))
        import hb_aug_v2 as A
    else:
        sys.path.insert(0, str((CODE_H2 if h2_only else CODE) / 'src'))
        from imagewam.datasets import hb_aug_v2 as A
    print('hb_aug_v2 from', A.__file__)
    if not args.skip_aug:
        check_parameters(A)
        check_apply(A)
    if not args.aug_only:
        if h2_only:
            check_task_h2(A, args.samples)
        else:
            for task in g1:
                check_task(task, A, args.samples)
    if h2_rc:
        FAILURES.append('a18 sub-check failed (see its output above)')
    print('VERIFY_FAIL %d' % len(FAILURES) if FAILURES else 'VERIFY_OK', flush=True)
    return 1 if FAILURES else 0


if __name__ == '__main__':
    sys.exit(main())
