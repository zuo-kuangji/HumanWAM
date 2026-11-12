"""H2 (A18 pour water, A17 unscrew bottle) preparation for the Delta-flow 40-epoch runs. CPU only, node 0, idempotent.

    $P/venv/bin/python $G/hb_deltaflow/prepare_deltaflow_h2.py <a18|a17>   (also called by prepare_deltaflow.py)
Generalizes prepare_deltaflow_a18.py (2026-11-12): proprio aligned with GR00T's A20 recipe (per-sample
hb_aug_v2.perturb_joints on the 31 H2 body joints, no delay, no history drop) and A17 added.
NO joint velocity (user rule 2026-11-10): the model is the original 78-D H2 model of a18-h2-pour-from-scratch-20261108
(imagewam/h2_pour_delta.py, unchanged; proprio_dim 78 = 75 joints + gravity 3).
Creates:
    $G/deltaflow-runs/HumanWAM-h2/   copy of $P/a18-h2-pour-from-scratch-20261108/HumanWAM (hash-checked) plus
        src/imagewam/datasets/hb_aug_v2.py   unchanged copy of $G/hb_groot/hb_aug_v2.py
        src/imagewam/datasets/h2_hbaug.py    H2HBAugV2Dataset (aug v2 + aligned 31-joint proprio aug, 78-D)
        scripts/wholebody/train_deltaflow.py the shared 40-epoch entry
        src/imagewam/datasets/h2_pour.py     frame total read from meta/info.json instead of the A18 constant 72752
    $G/deltaflow-runs/<task>/{stats/normalization.json, qwen-text/, run/production.yaml, data-verified.json, prepared.ok}
"""
import hashlib
import inspect
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

import prepare_deltaflow as base

H, G, P, ROOT = base.H, base.G, base.P, base.ROOT
OLD = P / 'a18-h2-pour-from-scratch-20261108'
SRC_CODE = OLD / 'HumanWAM'
CODE = ROOT / 'HumanWAM-h2'
TASKS = {'a18': dict(data=P / 'a18-h2-dataset-audit-8e956a10' / 'dataset', frames=72752, qwen='copy',
                     source='a18 H2 pour water (audited 8e956a10)'),
         'a17': dict(data=P / 'a17-unscrew-bottle-v1-407e6e2' / 'data', frames=115333, qwen='compute',
                     source='a17-unscrew-bottle-expert-v1 @407e6e2')}
EPISODES, BODY, STATE, ACTION = 100, 31, 78, 75
DATASET_TARGET = 'imagewam.datasets.h2_hbaug.H2HBAugV2Dataset'
SOURCE_SHA256 = {   # = h2-run-manifest.json file_sha256 / reviewed local copy
    'src/imagewam/datasets/h2_pour.py': 'd61ec58b83c20621488899c023a348fed52a7881f8faa4f6ba277fcb9c399168',
    'src/imagewam/h2_pour_delta.py': '69bf26169559b082a0a45152512497998163b63c69723d1af606dcf82eca47fc',
    'src/imagewam/datasets/wholebody.py': '4d81de5a73109752128a41a65b01121afe12de15cac42f215157c4e30008ccf9',
    'src/imagewam/trainer.py': 'cdcd48e09d1d7b81cae3a829bd415ed4b53ff1b290aacac1ce2217876f475b1a',
    'src/imagewam/runtime.py': '3ebc45f61b0c22fb659012a7d76bea671a42e1e528b5c917ae1e8827e88b890b',
    'scripts/wholebody/train_h2_pour.py': '69c6974418b9ed02cae4c9cca7177e3c32a213d9cc833ecd3d0c95489d901a5e',
    'scripts/accelerate_configs/accelerate_zero1_ds.yaml': '5b9e24f38c199fdff6bbd17981d86e2624f855f707660094c99f711efc8e0833',
}
H2_POUR_PATCHES = [  # make the A18-specific frame total generic (A17 has 115333 frames); everything else unchanged
    ("sum(row['length'] for row in episodes)!=72752", "sum(row['length'] for row in episodes)!=int(info['total_frames'])"),
    ("frames=72752,", "frames=sum(int(row['length']) for row in episodes),"),
]
ALIGNED = dict(delay_probability=0.0, max_delay_control_cycles=0, history_drop_probability=0.0,
               bias_scope='per_sample', implementation='hb_aug_v2.perturb_joints',
               aligned_with='GR00T A20 basket-transfer recipe (hb_groot/hb_launch.py), user 2026-11-12')
sha, dump, log = base.sha, base.dump, base.log


def build_code():
    record = CODE / 'deltaflow-code.json'
    if record.is_file():
        for rel, digest in json.loads(record.read_text())['sha256'].items():
            if sha(CODE / rel) != digest:
                raise SystemExit(f'{CODE / rel} changed after preparation; rename {CODE} aside and rerun')
        log('H2 code copy present and unchanged:', CODE)
        return
    if CODE.exists():
        raise SystemExit(f'{CODE} exists without deltaflow-code.json (incomplete); rename it aside and rerun')
    for rel, digest in SOURCE_SHA256.items():
        if sha(SRC_CODE / rel) != digest:
            raise SystemExit(f'H2 source differs from the reviewed copy: {SRC_CODE / rel}')
    if sha(base.AUG_SRC) != base.AUG_SHA256:
        raise SystemExit(f'{base.AUG_SRC} differs from the reviewed hb_aug_v2')
    ROOT.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / f'.HumanWAM-h2.tmp{os.getpid()}'
    shutil.copytree(SRC_CODE, tmp, symlinks=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    added = {'src/imagewam/datasets/hb_aug_v2.py': base.AUG_SRC,
             'src/imagewam/datasets/h2_hbaug.py': H / 'h2_hbaug.py',
             'scripts/wholebody/train_deltaflow.py': H / 'train_deltaflow.py'}
    for rel, source in added.items():
        shutil.copyfile(source, tmp / rel)
    pour = tmp / 'src/imagewam/datasets/h2_pour.py'
    text = pour.read_text(encoding='utf-8')
    for old, new in H2_POUR_PATCHES:
        if text.count(old) != 1:
            raise SystemExit(f'h2_pour patch anchor found {text.count(old)}x: {old!r}')
        text = text.replace(old, new)
    pour.write_text(text, encoding='utf-8')
    files = [*added, *SOURCE_SHA256]
    dump(tmp / 'deltaflow-code.json', dict(
        source=str(SRC_CODE), hb_aug_v2_source=str(base.AUG_SRC),
        created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        changes=['h2_pour_delta.py: unchanged original 78-D H2 model (no joint velocity)',
                 'hb_aug_v2.py: unchanged copy of the shared spec-v2 implementation',
                 'h2_hbaug.py: H2HBAugV2Dataset (aug v2 + 31-joint proprio aug aligned with GR00T A20: per-sample '
                 'perturb_joints, no delay, no history drop; 78-D state, no velocity)',
                 'h2_pour.py: total frame count read from meta/info.json (was the A18 constant 72752) so A17 loads',
                 'train_deltaflow.py: shared 40-epoch entry, full state at every save, --resume'],
        sha256={rel: sha(tmp / rel) for rel in files}))
    tmp.rename(CODE)
    log('H2 code copy built:', CODE)


def prepare(task):
    spec = TASKS[task]
    DATA, FRAMES = spec['data'], spec['frames']
    R = ROOT / task
    if (R / 'prepared.ok').is_file():
        log(task, 'already prepared, skipping')
        return
    if R.exists():
        aside = ROOT / f'{task}.incomplete-{time.strftime("%Y%m%dT%H%M%S")}'
        R.rename(aside)
        log(task, 'half-prepared directory moved to', aside)
    (R / 'run').mkdir(parents=True)
    (R / 'stats').mkdir()
    sys.path.insert(0, str(CODE / 'src'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from imagewam.datasets import h2_pour
    if not h2_pour.__file__.startswith(str(CODE / 'src')):
        raise SystemExit(f'imported the wrong imagewam: {h2_pour.__file__}')
    if h2_pour.STATE_DIM != STATE or h2_pour.ACTION_DIM != ACTION:
        raise SystemExit(f'h2_pour dims {h2_pour.STATE_DIM}/{h2_pour.ACTION_DIM}, expected {STATE}/{ACTION}')
    episodes, instruction, names = h2_pour.load_collection(DATA)
    lengths = [int(e['length']) for e in episodes]
    if [e['episode_index'] for e in episodes] != list(range(EPISODES)) or sum(lengths) != FRAMES:
        raise SystemExit(f'{task} collection is not episodes 0-99 / {FRAMES} frames')
    if list(h2_pour.CAMERAS) != ['observation.images.head', 'observation.images.chest']:
        raise SystemExit(f'Unexpected H2 cameras {h2_pour.CAMERAS}')

    log(task, 'normalization (q01/q99 of state78 / action75 over episodes 0-99)')
    h2_pour.fit(DATA, R / 'stats/normalization.json')

    u = math.ceil(FRAMES / (base.WORLD * base.MICRO))
    total, save_steps, warmup, n_epochs = base.fixed_schedule(u)
    prompt = h2_pour.DEFAULT_PROMPT.format(task=instruction)
    qinfo = None
    if spec['qwen'] == 'copy':
        shutil.copytree(OLD / 'qwen-text', R / 'qwen-text')
    else:
        import qwen_cache
        ref_episodes, ref_instruction, _ = h2_pour.load_collection(TASKS['a18']['data'])
        _, qinfo = qwen_cache.make(prompt, R / 'qwen-text', CODE, base.QWEN3, base.FLUX2_SRC,
                                   h2_pour.DEFAULT_PROMPT.format(task=ref_instruction), OLD / 'qwen-text')
        log(task, 'qwen cache computed', qinfo)
    cache = R / 'qwen-text' / f'{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt'
    if not cache.is_file():
        raise SystemExit(f'{task}: instruction cache missing: {cache.name}')

    source_yaml = OLD / 'run/production.yaml'
    cfg = yaml.safe_load(source_yaml.read_text())
    cfg.update(output_dir=str(R / 'run/production'), num_epochs=n_epochs, max_steps=total,
               lr_scheduler_total_steps=total, warmup_steps=warmup, save_every=0,
               save_steps=save_steps, save_at_end=True, keep_latest_weights=base.KEEP_CKPTS, keep_latest_state_only=True,
               resume=None, eval_every=0)
    if cfg['batch_size'] != base.MICRO or cfg['gradient_accumulation_steps'] != 1 or cfg['data']['val'] is not None:
        raise SystemExit(f'{task}: unexpected batch / accumulation / validation in the H2 config')
    cfg['wandb'].update(enabled=False, mode='disabled', name=f'deltaflow-{task}-40ep-augv2-aligned',
                        group='h2-action75-state78')
    old_train = cfg['data']['train']
    cfg['data']['train'] = dict(
        _target_=DATASET_TARGET, dataset_dir=str(DATA), stats_path=str(R / 'stats/normalization.json'),
        qwen_text_cache_dir=str(R / 'qwen-text'), qwen_context_len=old_train['qwen_context_len'],
        qwen_text_dim=old_train['qwen_text_dim'], video_cache_frames=64,
        image_augmentation=base.augmentation_config(),
        proprio_augmentation=dict(
            enabled=True, seed=20261106, joint_bias_degrees=2.0, joint_noise_std_degrees=0.25, control_hz=30,
            recording_hz=30, affected_angle_channels=list(range(BODY)), velocity_channels=[],
            action_label_noise=False, body_joints='observation.state[0:31] incl. head',
            start_pose_library='not_applied_offline', render_randomization='not_applied_offline', **ALIGNED))
    m = cfg['model']
    if (m['expected_action_dim'], m['action_dit_config']['action_dim'], len(m['loss']['action_a2a']['source_noise_dim_mask']),
            m['proprio_dim'], m['action_state_dim'], m['expected_action_horizon']) != (ACTION, ACTION, ACTION, STATE, STATE, 16):
        raise SystemExit(f'{task}: the H2 config is not the 78-D state / 75-D action Delta-flow it should be')
    if 'h2_pour_delta' not in str(m.get('_target_', '')):
        raise SystemExit(f"{task}: unexpected model target {m.get('_target_')!r} (want the original h2_pour_delta)")
    cfg['wholebody_contract'] = dict(
        task=task, robot='h2_sharpa', world_size=base.WORLD, zero_stage=1, fresh_base=True,
        production_output_dir=str(R / 'run/production'), updates_per_epoch=u, train_episode_ids='0-99',
        statistics_sha256=sha(R / 'stats/normalization.json'), augmentation=base.AUG_VERSION,
        proprio_augmentation='aligned with GR00T A20 (per-sample perturb_joints, no delay, no history drop)',
        state_layout='joints75 + gravity3 = 78 (no velocity)',
        action_layout='body_target31 + hands44 = 75', cameras=list(h2_pour.CAMERAS),
        measured_loader=dict(dataset_frames=FRAMES, raw_loader_batches=math.ceil(FRAMES / base.MICRO),
                             per_rank_microbatches=u, world_size=base.WORLD, microbatch=base.MICRO,
                             gradient_accumulation=1),
        source_config=str(source_yaml))
    for value in (cfg['output_dir'], cfg['data']['train']['stats_path'], cfg['data']['train']['qwen_text_cache_dir']):
        if value.startswith(str(OLD) + '/'):
            raise SystemExit(f'{task}: config still points into the old experiment: {value}')
    base.no_velocity(cfg, task)
    (R / 'run/production.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    previous = OLD / 'normalization.json'
    dump(R / 'data-verified.json', dict(
        task=task, dataset_dir=str(DATA), dataset_source=spec['source'], total_episodes_in_collection=EPISODES,
        episodes=EPISODES, train_episode_ids=list(range(EPISODES)), shelved_episode_ids=[], frames=FRAMES,
        updates_per_epoch=u, epochs=None, fixed_steps=True,
        epochs_equivalent=round(total / u, 2), max_steps=total, warmup_steps=warmup,
        save_steps=save_steps, control_hz=30, recording_hz=30, statistics_sha256=sha(R / 'stats/normalization.json'),
        statistics_equal_previous_a18_run=(task == 'a18' and previous.is_file()
                                           and sha(previous) == sha(R / 'stats/normalization.json')),
        augmentation=base.AUG_VERSION, proprio_augmentation=ALIGNED, qwen_cache=(qinfo or 'copied from the A18 run'),
        state_dim=STATE, action_dim=ACTION, joint_velocity=False,
        code=str(CODE), code_record_sha256=sha(CODE / 'deltaflow-code.json')))

    log(task, 'model module check (no weights loaded)')
    import imagewam.h2_pour_delta as model_module
    factory = inspect.signature(model_module.create_imagewam_flux2_wholebody_delta)
    if (factory.parameters['action_state_dim'].default != STATE
            or factory.parameters['expected_action_dim'].default != ACTION):
        raise SystemExit(f'{task}: original H2 model module is not 78-D state / 75-D action')

    log(task, 'loading real augmented samples (CPU)')
    import numpy as np
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    dataset = instantiate(OmegaConf.load(R / 'run/production.yaml').data.train)
    if len(dataset) != FRAMES or list(dataset.episode_ids) != list(range(EPISODES)):
        raise SystemExit(f'{task}: dataset length {len(dataset)} differs from {FRAMES}')
    first, sample = dataset[0], dataset[145]
    if (tuple(first['video'].shape) != (3, 2, 224, 448) or tuple(first['proprio'].shape) != (16, STATE)
            or tuple(sample['proprio'].shape) != (16, STATE) or tuple(sample['proprio_dim_is_pad'].shape) != (STATE,)
            or tuple(sample['action'].shape) != (16, ACTION) or tuple(sample['past_action'].shape) != (16, ACTION)
            or not torch.isfinite(sample['proprio']).all() or dataset.last_augmentation['visual']['boost'] is None):
        raise SystemExit(f'{task}: sample schema check failed')
    perturb = []
    for i in range(0, FRAMES, max(1, FRAMES // 40)):
        dataset[i]
        num = dataset.last_augmentation['numeric']
        if num['delay_recorded_frames'] != 0 or num['history_drop_rows'] or num['state_frame'] != num['anchor_frame']:
            raise SystemExit(f'{task}: delay / history drop still active: {num}')
        perturb.append(num['joint_perturbation_degrees'])
    perturb = np.asarray(perturb)
    if perturb.shape[1] != BODY or np.abs(perturb).max() > 2.0 + 0.75 + 1e-3 or perturb.std(axis=0).min() < 0.5:
        raise SystemExit(f'{task}: joint perturbation is not per-sample +-2 deg + 0.25 deg noise')
    dump(R / 'prepared.ok', dict(frames=FRAMES, updates_per_epoch=u, max_steps=total, save_steps=save_steps,
                                 proprio_check=dict(samples=len(perturb), max_abs_deg=float(np.abs(perturb).max()),
                                                    std_deg=float(perturb.std()))))
    log(task, 'PREPARED', json.dumps(dict(frames=FRAMES, updates_per_epoch=u, max_steps=total, save_steps=save_steps,
                                          warmup=warmup, state_dim=STATE, action_dim=ACTION)))


if __name__ == '__main__':
    tasks = sys.argv[1:] or list(TASKS)
    for t in tasks:
        if t not in TASKS:
            raise SystemExit(f'unknown H2 task {t}; choose from {list(TASKS)}')
    build_code()
    for t in tasks:
        prepare(t)
