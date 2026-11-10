"""One-time CPU preparation of the ImageWAM Delta-flow 40-epoch runs (B6 / A3 / A7, 100 episodes each, aug v2).

NO joint velocity (user rule 2026-11-10): the model input is 44-D (body 29 + hands 12 + gravity 3), exactly like the
reference HF humanoid-benchmark/b6-imagewam-deltaflow76 real30-from-sim60-20261108 (proprio_dim = action_state_dim
= 44, velocity_channels []). The previous run's YAML is a 73-D (velocity) config: its velocity keys are removed and
the model dims are set to 44; no velocity cache or statistics are produced, and no_velocity() rejects any leftover.

Run on node 0 (no GPU needed), after deploying this directory to $G/hb_deltaflow:
    $P/venv/bin/python $G/hb_deltaflow/prepare_deltaflow.py            # all of b6 a3 a7 a18
    $P/venv/bin/python $G/hb_deltaflow/prepare_deltaflow.py b6         # one task
    (a18 = H2 is delegated to prepare_deltaflow_a18.py in a separate interpreter; see that file)
Creates (never touches the old experiments or the datasets):
    $G/deltaflow-runs/HumanWAM/                 code copy of the last B6 Delta-flow run + 100-episode loader patch,
                                                hb_aug_v2.py, wholebody_hbaug.py, train_deltaflow.py
    $G/deltaflow-runs/<task>/{split-manifest.json, stats/, qwen-text/, run/production.yaml, data-verified.json,
                              prepared.ok}
Idempotent: finished tasks (prepared.ok) are skipped; a half-prepared task directory is renamed aside and redone.
"""
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

H = Path(__file__).resolve().parent          # $G/hb_deltaflow
G = H.parent                                 # $P/policies-20261109
P = G.parent                                 # preflight-20261104T1648Z
OLD_B6 = P / 'b6-v5t7-ready-start-60ep-20261109'
OLD_A = P / 'a3-a7-20261109'
SRC_CODE = OLD_B6 / 'HumanWAM'
AUG_SRC = G / 'hb_groot' / 'hb_aug_v2.py'
ROOT = G / 'deltaflow-runs'
CODE = ROOT / 'HumanWAM'

EPISODES, EPOCHS, SAVE_EVERY_EPOCHS, WARMUP_EPOCHS = 100, 40, 10, 2
WORLD, MICRO = 32, 8
DATASET_TARGET = 'imagewam.datasets.wholebody_hbaug.WholeBodyHBAugV2LeRobotDataset'
AUG_VERSION = 'hb-aug-v2-20261109'
AUG_SHA256 = '9f7713cea4f846efcfc924234267ced48210357f419d67915c10b504316bedde'
TASKS = {
    'b6': dict(data=OLD_B6 / 'data/deliveries/g1/b6-v5t7-ready-start-20261109', old=OLD_B6 / 'train',
               control_hz=50, total_episodes=149),
    'a3': dict(data=OLD_A / 'a3-fridge-open-expert-v1/780fcf1e4bd7db947a8c2d8f7b92e61f24f71b45', old=OLD_A / 'a3',
               control_hz=30, total_episodes=100),
    'a7': dict(data=OLD_A / 'a7-cabinet-close-expert-v1/1935a98e8f55d4d1d2d1615ddb97c2679713595c', old=OLD_A / 'a7',
               control_hz=30, total_episodes=100),
}
# Files of the last B6 Delta-flow code copy that this package builds on (hashes of the reviewed copy).
SOURCE_SHA256 = {
    'src/imagewam/datasets/wholebody.py': 'e6460388758187180a28515cb15b666d948387b1b6aff7b026baa2f2e9402ccc',
    'src/imagewam/datasets/wholebody_velocity.py': 'a5788736652772aea2fcf305b47ca6a6c11e1ea798fe8e2cf7bcdd1a242b23ea',
    'src/imagewam/datasets/wholebody_augmented_v3.py': '0e34ea3a71dde2f05bd4bd314ec97787bf3d1809c1cedc3dff5e8610b7723366',
    'src/imagewam/datasets/wholebody_augmented.py': '641bb399f54206fe1c923c4b24fd1b9797787616154915523f10a9480daf7da3',
    'src/imagewam/trainer.py': 'cdcd48e09d1d7b81cae3a829bd415ed4b53ff1b290aacac1ce2217876f475b1a',
    'src/imagewam/runtime.py': '3ebc45f61b0c22fb659012a7d76bea671a42e1e528b5c917ae1e8827e88b890b',
    'src/imagewam/wholebody_delta.py': '20135b5a6cc0378c76d3eb8d958cbc4c3aeb773909b1453081f4fcd6238d36ea',
    'scripts/wholebody/train_a3a7.py': '2cf154f91de516f96da9b93f4b8e235fab13344461b2aeb5d048e4f78a63dec8',
    'scripts/accelerate_configs/accelerate_zero1_ds.yaml': '5b9e24f38c199fdff6bbd17981d86e2624f855f707660094c99f711efc8e0833',
}
# wholebody.py of the B6 run accepts exactly 149/0 episodes; make it accept exactly episodes 0-99 for training,
# with any later episodes listed as shelved (validation_episode_ids, never loaded for training or statistics).
LOADER_PATCHES = [
    ('STATS_SCHEMA = "wholebody-state41to44-action76-chunk16-train149val0-v1"',
     'STATS_SCHEMA = "wholebody-state41to44-action76-chunk16-first100-v1"', 1),
    ('    if (len(train) != 149 or len(val) != 0 or train != sorted(set(train))\n'
     '            or val != sorted(set(val)) or set(train) & set(val)\n'
     '            or sorted(train + val) != ids):\n'
     '        raise ValueError("Require exact disjoint 149/0 episode partition of 149 episodes")\n',
     '    if (train != list(range(100)) or val != sorted(set(val)) or set(train) & set(val)\n'
     '            or sorted(train + val) != ids):\n'
     '        raise ValueError("Require training episodes exactly 0-99; later episodes may only be listed as shelved")\n',
     1),
    ('    if train_split_mode != "all_149" or len(meta["train"]) != 149 or len(meta["validation"]) != 0:\n'
     '        raise ValueError("Require reviewed all-149 training partition")\n',
     '    if train_split_mode != "first_100" or list(meta["train"]) != list(range(100)):\n'
     '        raise ValueError("Require the fixed first-100 training partition (episodes 0-99)")\n', 1),
    ('"all_149"', '"first_100"', 7),
]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(16 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def augmentation_config():
    return dict(
        version=AUG_VERSION, enabled=True, seed=20261110, spec='project CLAUDE.md augmentation default spec v2 (2026-11-09)',
        implementation='imagewam/datasets/hb_aug_v2.py = unchanged copy of $G/hb_groot/hb_aug_v2.py',
        implementation_sha256=AUG_SHA256, boost='one of 15 effects uniformly per sample, magnitude x1.5',
        colour='exposure/contrast/saturation/hue/gamma/temperature shared by both cameras and both times',
        per_camera='rotation+crop, light/shadow patches, single-channel white balance; same on t and t+16',
        observation_only='noise, gaussian blur, JPEG, motion blur, sharpen, defocus (never on the t+16 target)',
        native_model_image_augmentation='none', action_label_noise=False, state_label_noise=False)


def build_code():
    record = CODE / 'deltaflow-code.json'
    if record.is_file():
        for rel, digest in json.loads(record.read_text())['sha256'].items():
            if sha(CODE / rel) != digest:
                raise SystemExit(f'{CODE / rel} changed after preparation; rename {CODE} aside and rerun')
        log('code copy present and unchanged:', CODE)
        return
    if CODE.exists():
        raise SystemExit(f'{CODE} exists without deltaflow-code.json (incomplete); rename it aside and rerun')
    for rel, digest in SOURCE_SHA256.items():
        if sha(SRC_CODE / rel) != digest:
            raise SystemExit(f'Source code differs from the reviewed copy: {SRC_CODE / rel}')
    if sha(AUG_SRC) != AUG_SHA256:
        raise SystemExit(f'{AUG_SRC} differs from the reviewed hb_aug_v2 ({AUG_SHA256[:12]})')
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / f'.HumanWAM.tmp{os.getpid()}'
    shutil.copytree(SRC_CODE, tmp, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    loader = tmp / 'src/imagewam/datasets/wholebody.py'
    text = loader.read_text(encoding='utf-8')
    for old, new, count in LOADER_PATCHES:
        if text.count(old) != count:
            raise SystemExit(f'Loader patch anchor found {text.count(old)}x (want {count}): {old[:60]!r}')
        text = text.replace(old, new)
    if '149' in text:
        raise SystemExit('Loader still mentions 149 after patching')
    loader.write_text(text, encoding='utf-8')
    added = {'src/imagewam/datasets/hb_aug_v2.py': AUG_SRC,
             'src/imagewam/datasets/wholebody_hbaug.py': H / 'wholebody_hbaug.py',
             'scripts/wholebody/train_deltaflow.py': H / 'train_deltaflow.py'}
    for rel, source in added.items():
        shutil.copyfile(source, tmp / rel)
    files = ['src/imagewam/datasets/wholebody.py', *added, *[k for k in SOURCE_SHA256 if k != 'src/imagewam/datasets/wholebody.py']]
    dump(tmp / 'deltaflow-code.json', dict(
        source=str(SRC_CODE), hb_aug_v2_source=str(AUG_SRC), created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        changes=['wholebody.py: train exactly episodes 0-99 ("first_100"), later episodes shelved',
                 'hb_aug_v2.py: unchanged copy of the shared spec-v2 implementation',
                 'wholebody_hbaug.py: WholeBodyHBAugV2LeRobotDataset(WholeBodyRobustLeRobotDataset), visual aug -> '
                 'hb_aug_v2, 44-D proprio, NO joint velocity (wholebody_velocity.py stays in the tree, unused)',
                 'train_deltaflow.py: 40-epoch entry, full state at every save, --resume'],
        sha256={rel: sha(tmp / rel) for rel in files}))
    tmp.rename(CODE)
    log('code copy built:', CODE)


PROPRIO_DIM = 44   # body 29 + hands 12 + gravity 3 (NO velocity)


def no_velocity(cfg, where):
    """Reject any joint-velocity remnant in a training config (user rule 2026-11-10)."""
    def walk(value, path):
        if isinstance(value, dict):
            for key, item in value.items():
                if 'velocity' in str(key).lower() and item not in ([], None, False, 0):
                    raise SystemExit(f'{where}: velocity setting left in the config: {path}.{key}={item!r}')
                walk(item, f'{path}.{key}')
        elif isinstance(value, list):
            for i, item in enumerate(value):
                walk(item, f'{path}[{i}]')
        elif isinstance(value, str) and 'velocity' in value.lower() and 'no velocity' not in value.lower():
            raise SystemExit(f'{where}: velocity reference left in the config: {path}={value!r}')
    walk(cfg, 'cfg')


def prepare_task(task, wb):
    spec = TASKS[task]
    R = ROOT / task
    if (R / 'prepared.ok').is_file():
        log(task, 'already prepared, skipping')
        return
    if R.exists():
        aside = ROOT / f'{task}.incomplete-{time.strftime("%Y%m%dT%H%M%S")}'
        R.rename(aside)
        log(task, 'half-prepared directory moved to', aside)
    D, old = spec['data'], spec['old']
    (R / 'run').mkdir(parents=True)
    log(task, 'dataset', D)

    meta = wb.load_metadata(D)
    ids = [int(e['episode_index']) for e in meta['episodes']]
    lengths = {int(e['episode_index']): int(e['length']) for e in meta['episodes']}
    train = list(range(EPISODES))
    if len(ids) != spec['total_episodes'] or ids[:EPISODES] != train:
        raise SystemExit(f'{task}: expected {spec["total_episodes"]} episodes 0..N-1, found {len(ids)}')
    shelved = ids[EPISODES:]
    old_manifest = json.loads((old / 'split-manifest.json').read_text())
    if sorted(old_manifest['train_episode_ids'] + old_manifest['validation_episode_ids']) != ids:
        raise SystemExit(f'{task}: previous split manifest does not cover this dataset')
    basis = ('Fixed first 100 episodes (episode_index 0-99) per project rule 2026-11-09; '
             + (f'episodes {shelved[0]}-{shelved[-1]} are shelved (listed as validation ids, never trained, '
                'no validation run)' if shelved else 'the collection has exactly 100 episodes'))
    manifest = dict(old_manifest, train_episode_ids=train, validation_episode_ids=shelved, shelved_episode_ids=shelved,
                    split_basis=basis, previous_manifest=str(old / 'split-manifest.json'))
    dump(R / 'split-manifest.json', manifest)
    wb.apply_split(meta, str(R / 'split-manifest.json'))     # provenance hashes, joint names, groups

    log(task, 'fitting normalization statistics on episodes 0-99')
    wb.prepare_statistics(str(D), str(R / 'stats'), train_split_mode='first_100',
                          split_manifest_path=str(R / 'split-manifest.json'))
    stats = json.loads((R / 'stats/dataset_stats.json').read_text())
    frames = sum(lengths[i] for i in train)
    if (stats['train_episode_ids'] != train or stats['heldout_episode_ids'] != shelved
            or stats['train_frames'] != frames or stats['schema'] != wb.STATS_SCHEMA):
        raise SystemExit(f'{task}: statistics do not cover exactly episodes 0-99')

    u = math.ceil(frames / (WORLD * MICRO))
    if u != math.ceil(math.ceil(frames / MICRO) / WORLD):
        raise SystemExit('updates_per_epoch mismatch')
    total = EPOCHS * u
    save_steps = [e * u for e in range(SAVE_EVERY_EPOCHS, EPOCHS + 1, SAVE_EVERY_EPOCHS)]

    shutil.copytree(old / 'qwen-text', R / 'qwen-text')
    prompt = wb.DEFAULT_PROMPT.format(task=meta['tasks'][0]['task'])
    cache = R / 'qwen-text' / f'{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt'
    if not cache.is_file():
        raise SystemExit(f'{task}: instruction cache missing in copied qwen-text: {cache.name}')

    source_yaml = old / 'run/production.yaml'
    cfg = yaml.safe_load(source_yaml.read_text())
    cfg.update(output_dir=str(R / 'run/production'), num_epochs=EPOCHS, max_steps=total,
               lr_scheduler_total_steps=total, warmup_steps=WARMUP_EPOCHS * u, save_every=0, save_steps=save_steps,
               save_at_end=True, keep_latest_weights=4, keep_latest_state_only=True, resume=None, eval_every=0)
    if cfg['batch_size'] != MICRO or cfg['gradient_accumulation_steps'] != 1 or cfg['data']['val'] is not None:
        raise SystemExit(f'{task}: unexpected batch / accumulation / validation in {source_yaml}')
    cfg['wandb'].update(enabled=False, mode='disabled', name=f'deltaflow-{task}-40ep-augv2')
    d = cfg['data']['train']
    d.update(_target_=DATASET_TARGET, dataset_dir=str(D), stats_path=str(R / 'stats/dataset_stats.json'),
             qwen_text_cache_dir=str(R / 'qwen-text'), split_manifest_path=str(R / 'split-manifest.json'),
             train_split_mode='first_100')
    for key in [k for k in d if 'velocity' in k]:   # the previous run's 73-D loader keys (velocity_cache, noise)
        del d[key]
    d['image_augmentation'] = augmentation_config()
    pa = d['proprio_augmentation']
    pa.update(control_hz=spec['control_hz'], recording_hz=30, velocity_channels=[])
    for key, value in dict(enabled=True, joint_bias_degrees=2.0, joint_noise_std_degrees=0.25, delay_probability=0.15,
                           max_delay_control_cycles=2, history_drop_probability=0.15,
                           affected_angle_channels=list(range(29)), bias_scope='fixed_per_episode').items():
        if pa.get(key) != value:
            raise SystemExit(f'{task}: proprio_augmentation.{key}={pa.get(key)!r} differs from spec v2 ({value!r})')
    m = cfg['model']
    if (m['proprio_dim'], m['action_state_dim'], m['expected_action_dim'], m['expected_action_horizon']) not in (
            (73, 73, 76, 16), (44, 44, 76, 16)):
        raise SystemExit(f'{task}: source model is not the 76-D action / horizon-16 Delta-flow')
    m.update(proprio_dim=PROPRIO_DIM, action_state_dim=PROPRIO_DIM)   # = reference real30-from-sim60 (no velocity)
    c = cfg['wholebody_contract']
    for key in [k for k in c if 'velocity' in k]:
        del c[key]
    c.update(stats_schema=wb.STATS_SCHEMA, statistics_sha256=sha(R / 'stats/dataset_stats.json'),
             smoke_output_dir=None, production_output_dir=str(R / 'run/production'), updates_per_epoch=u,
             augmentation=AUG_VERSION, proprio_layout='body29 + hands12 + gravity3 = 44 (no velocity)',
             train_episode_ids='0-99', shelved_episode_ids=(f'{shelved[0]}-{shelved[-1]}' if shelved else 'none'))
    c['measured_loader'].update(dataset_frames=frames, raw_loader_batches=math.ceil(frames / MICRO),
                                per_rank_microbatches=u, world_size=WORLD, microbatch=MICRO)
    old_prefix = str(old) + '/'

    def walk(value):
        if isinstance(value, dict):
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and (value.startswith(old_prefix) or value == str(old)):
            raise SystemExit(f'{task}: config still points into the old experiment: {value}')
    walk(cfg)
    no_velocity(cfg, task)
    c['source_config'] = str(source_yaml)
    (R / 'run/production.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    dump(R / 'data-verified.json', dict(
        task=task, dataset_dir=str(D), total_episodes_in_collection=len(ids), episodes=EPISODES,
        train_episode_ids=train, shelved_episode_ids=shelved, frames=frames, updates_per_epoch=u, epochs=EPOCHS,
        max_steps=total, warmup_steps=WARMUP_EPOCHS * u, save_steps=save_steps, control_hz=spec['control_hz'],
        recording_hz=30, statistics_sha256=c['statistics_sha256'], proprio_dim=PROPRIO_DIM, joint_velocity=False,
        augmentation=AUG_VERSION, code=str(CODE), code_record_sha256=sha(CODE / 'deltaflow-code.json')))

    log(task, 'loading two real augmented samples (CPU)')
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    dataset = instantiate(OmegaConf.load(R / 'run/production.yaml').data.train)
    if len(dataset) != frames or list(dataset.episode_ids) != train:
        raise SystemExit(f'{task}: dataset length {len(dataset)} / episodes differ from {frames} / 0-99')
    import torch
    first, sample = dataset[0], dataset[min(20, len(dataset) - 1)]
    if (tuple(first['video'].shape) != (3, 2, 224, 448) or tuple(first['proprio'].shape) != (16, PROPRIO_DIM)
            or tuple(sample['proprio'].shape) != (16, PROPRIO_DIM) or tuple(sample['action'].shape) != (16, 76)
            or not torch.isfinite(sample['proprio']).all() or dataset.last_augmentation['visual']['boost'] is None):
        raise SystemExit(f'{task}: sample schema check failed')
    dump(R / 'prepared.ok', dict(frames=frames, updates_per_epoch=u, max_steps=total, save_steps=save_steps))
    log(task, 'PREPARED', json.dumps(dict(frames=frames, updates_per_epoch=u, max_steps=total, save_steps=save_steps,
                                          warmup=WARMUP_EPOCHS * u, control_hz=spec['control_hz'])))


def main():
    tasks = sys.argv[1:] or [*TASKS, 'a18']
    for task in tasks:
        if task not in TASKS and task != 'a18':
            raise SystemExit(f'unknown task {task}; choose from {[*TASKS, "a18"]}')
    g1 = [task for task in tasks if task in TASKS]
    if g1:
        build_code()
        sys.path.insert(0, str(CODE / 'src'))
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        import imagewam.datasets.wholebody as wb
        if not wb.__file__.startswith(str(CODE / 'src')):
            raise SystemExit(f'imported the wrong imagewam: {wb.__file__}')
        for task in g1:
            prepare_task(task, wb)
    if 'a18' in tasks:
        # The H2 branch ships its own imagewam package: prepare it in a separate interpreter.
        import subprocess
        rc = subprocess.call([sys.executable, str(H / 'prepare_deltaflow_a18.py')])
        if rc != 0:
            raise SystemExit(f'a18 preparation failed (rc {rc})')
    log('PREPARE_DONE', ' '.join(tasks))


if __name__ == '__main__':
    main()
