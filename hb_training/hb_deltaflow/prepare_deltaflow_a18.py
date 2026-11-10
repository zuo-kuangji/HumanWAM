"""A18 (H2 pour water) preparation for the Delta-flow 40-epoch runs. CPU only, node 0, idempotent.

    $P/venv/bin/python $G/hb_deltaflow/prepare_deltaflow_a18.py      (also called by prepare_deltaflow.py a18)
Creates:
    $G/deltaflow-runs/HumanWAM-h2/   copy of $P/a18-h2-pour-from-scratch-20261108/HumanWAM (hash-checked) plus
        src/imagewam/h2_vel_delta.py         h2_pour_delta.py with the current-state width 78 -> 109
                                             (75 joints + gravity 3 + body velocity 31) and its own schema id
        src/imagewam/datasets/hb_aug_v2.py   unchanged copy of $G/hb_groot/hb_aug_v2.py
        src/imagewam/datasets/h2_hbaug.py    H2HBAugV2Dataset (aug v2, 31-joint proprio aug and velocity)
        scripts/wholebody/train_deltaflow.py the shared 40-epoch entry
    $G/deltaflow-runs/a18/{stats/normalization.json, stats/velocity.npz, stats/velocity-normalization.json,
                           qwen-text/, run/production.yaml, data-verified.json, prepared.ok}
"""
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

import prepare_deltaflow as base

H, G, P, ROOT = base.H, base.G, base.P, base.ROOT
OLD = P / 'a18-h2-pour-from-scratch-20261108'
SRC_CODE = OLD / 'HumanWAM'
DATA = P / 'a18-h2-dataset-audit-8e956a10' / 'dataset'
CODE = ROOT / 'HumanWAM-h2'
R = ROOT / 'a18'
EPISODES, FRAMES, BODY, RAW_STATE, STATE, ACTION = 100, 72752, 31, 78, 109, 75
DATASET_TARGET = 'imagewam.datasets.h2_hbaug.H2HBAugV2Dataset'
MODEL_TARGET = 'imagewam.h2_vel_delta.create_imagewam_flux2_wholebody_delta'
SCHEMA_ID = 'a18-h2-sharpa-body31-hand44-state78-vel31-chunk16-v1'
CHECKPOINT_FORMAT = 'h2_sharpa_body31_hand44_vel31_flux2_a18_chunk16_v1'
SOURCE_SHA256 = {   # = h2-run-manifest.json file_sha256 / reviewed local copy
    'src/imagewam/datasets/h2_pour.py': 'd61ec58b83c20621488899c023a348fed52a7881f8faa4f6ba277fcb9c399168',
    'src/imagewam/h2_pour_delta.py': '69bf26169559b082a0a45152512497998163b63c69723d1af606dcf82eca47fc',
    'src/imagewam/datasets/wholebody.py': '4d81de5a73109752128a41a65b01121afe12de15cac42f215157c4e30008ccf9',
    'src/imagewam/trainer.py': 'cdcd48e09d1d7b81cae3a829bd415ed4b53ff1b290aacac1ce2217876f475b1a',
    'src/imagewam/runtime.py': '3ebc45f61b0c22fb659012a7d76bea671a42e1e528b5c917ae1e8827e88b890b',
    'scripts/wholebody/train_h2_pour.py': '69c6974418b9ed02cae4c9cca7177e3c32a213d9cc833ecd3d0c95489d901a5e',
    'scripts/accelerate_configs/accelerate_zero1_ds.yaml': '5b9e24f38c199fdff6bbd17981d86e2624f855f707660094c99f711efc8e0833',
}
HEX = re.compile(r'([0-9a-f]{64})')
sha, dump, log = base.sha, base.dump, base.log


def widen_state(text):
    """78 -> 109 everywhere the H2 model names its current-state width, never inside pinned sha256 strings."""
    parts = HEX.split(text)
    count = 0
    for i in range(0, len(parts), 2):
        parts[i], n = re.subn(r'(?<!\d)78(?!\d)', str(STATE), parts[i])
        count += n
    if count != 21:
        raise SystemExit(f'h2_pour_delta.py: expected 21 state-width occurrences of 78, found {count}')
    out = ''.join(parts)
    for name, value in (('SCHEMA_ID', SCHEMA_ID), ('CHECKPOINT_FORMAT', CHECKPOINT_FORMAT)):
        out, n = re.subn(rf'^{name} = .*$', f'{name} = "{value}"', out, flags=re.M)
        if n != 1:
            raise SystemExit(f'h2_pour_delta.py: {name} line not found')
    if HEX.findall(out) != HEX.findall(text):
        raise SystemExit('h2_pour_delta.py: pinned upstream hashes changed while widening the state')
    return out


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
    model = widen_state((tmp / 'src/imagewam/h2_pour_delta.py').read_text(encoding='utf-8'))
    (tmp / 'src/imagewam/h2_vel_delta.py').write_text(model, encoding='utf-8')
    added = {'src/imagewam/datasets/hb_aug_v2.py': base.AUG_SRC,
             'src/imagewam/datasets/h2_hbaug.py': H / 'h2_hbaug.py',
             'scripts/wholebody/train_deltaflow.py': H / 'train_deltaflow.py'}
    for rel, source in added.items():
        shutil.copyfile(source, tmp / rel)
    files = ['src/imagewam/h2_vel_delta.py', *added, *SOURCE_SHA256]
    dump(tmp / 'deltaflow-code.json', dict(
        source=str(SRC_CODE), hb_aug_v2_source=str(base.AUG_SRC),
        created=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        changes=['h2_vel_delta.py: copy of h2_pour_delta.py, current-state width 78 -> 109 (+31 body velocity), '
                 f'schema {SCHEMA_ID}, checkpoint format {CHECKPOINT_FORMAT}; h2_pour_delta.py left unchanged',
                 'hb_aug_v2.py: unchanged copy of the shared spec-v2 implementation',
                 'h2_hbaug.py: H2HBAugV2Dataset (aug v2 + 31-joint proprio aug + 31-joint velocity)',
                 'train_deltaflow.py: shared 40-epoch entry, full state at every save, --resume'],
        sha256={rel: sha(tmp / rel) for rel in files}))
    tmp.rename(CODE)
    log('H2 code copy built:', CODE)


def compute_velocity(names):
    velocities, dts = {}, []
    for episode in range(EPISODES):
        path = DATA / f'data/chunk-000/episode_{episode:06d}.parquet'
        table = pq.read_table(path, columns=['observation.state', 'timestamp', 'frame_index'])
        q = np.asarray(table['observation.state'].to_pylist(), dtype=np.float64)[:, :BODY]
        dt = np.diff(table['timestamp'].to_numpy().astype(np.float64))
        if table['frame_index'].to_pylist() != list(range(len(q))) or not (dt > 0).all():
            raise SystemExit(f'Bad frame index / timestamps in {path}')
        velocity = np.zeros_like(q, dtype=np.float32)
        velocity[1:] = np.diff(q, axis=0) / dt[:, None]
        if not np.isfinite(velocity).all():
            raise SystemExit(f'Non-finite velocity in {path}')
        velocities[episode] = velocity
        dts.append(dt)
    valid = np.concatenate([v[1:] for v in velocities.values()])
    mean = valid.mean(axis=0)
    scale = np.maximum(valid.std(axis=0), 0.05)
    dt = np.concatenate(dts)
    meta = dict(order=names[:BODY], mean=mean.tolist(), std=scale.tolist(), std_floor_rad_s=0.05,
                training_noise_std_rad_s=0.05, normalized_clip=5, train_episode_ids=list(range(EPISODES)),
                method='(q[t] - q[t-1]) / (timestamp[t] - timestamp[t-1]), observation.state[:, 0:31]',
                first_frame='normalized zero', peak_abs_velocity_rad_s=float(np.abs(valid).max()),
                timestamp_step_s=dict(min=float(dt.min()), max=float(dt.max()), mean=float(dt.mean())),
                model_input_order='state78 (75 joints + gravity3, q01/q99) then velocity31 -> 109')
    return velocities, mean, scale, meta


def prepare():
    if (R / 'prepared.ok').is_file():
        log('a18 already prepared, skipping')
        return
    if R.exists():
        aside = ROOT / f'a18.incomplete-{time.strftime("%Y%m%dT%H%M%S")}'
        R.rename(aside)
        log('a18 half-prepared directory moved to', aside)
    (R / 'run').mkdir(parents=True)
    (R / 'stats').mkdir()
    sys.path.insert(0, str(CODE / 'src'))
    os.environ.setdefault('HF_HUB_OFFLINE', '1')
    from imagewam.datasets import h2_pour
    if not h2_pour.__file__.startswith(str(CODE / 'src')):
        raise SystemExit(f'imported the wrong imagewam: {h2_pour.__file__}')
    episodes, instruction, names = h2_pour.load_collection(DATA)   # exactly 100 episodes / 72752 frames
    lengths = [int(e['length']) for e in episodes]
    if [e['episode_index'] for e in episodes] != list(range(EPISODES)) or sum(lengths) != FRAMES:
        raise SystemExit('A18 collection is not episodes 0-99 / 72752 frames')
    if list(h2_pour.CAMERAS) != ['observation.images.head', 'observation.images.chest']:
        raise SystemExit(f'Unexpected H2 cameras {h2_pour.CAMERAS}')

    log('a18 normalization (q01/q99 of state78 / action75 over episodes 0-99)')
    h2_pour.fit(DATA, R / 'stats/normalization.json')
    log('a18 31-joint velocity cache / statistics')
    velocities, mean, scale, vmeta = compute_velocity(names)
    np.savez(R / 'stats/velocity.npz', mean=mean, scale=scale, **{f'episode_{i}': v for i, v in velocities.items()})
    dump(R / 'stats/velocity-normalization.json', vmeta)

    u = math.ceil(FRAMES / (base.WORLD * base.MICRO))              # 285
    total = base.EPOCHS * u
    save_steps = [e * u for e in range(base.SAVE_EVERY_EPOCHS, base.EPOCHS + 1, base.SAVE_EVERY_EPOCHS)]
    shutil.copytree(OLD / 'qwen-text', R / 'qwen-text')
    prompt = h2_pour.DEFAULT_PROMPT.format(task=instruction)
    cache = R / 'qwen-text' / f'{hashlib.sha256(prompt.encode()).hexdigest()}.qwen3_flux2_len128.pt'
    if not cache.is_file():
        raise SystemExit(f'a18: instruction cache missing in copied qwen-text: {cache.name}')

    source_yaml = OLD / 'run/production.yaml'
    cfg = yaml.safe_load(source_yaml.read_text())
    cfg.update(output_dir=str(R / 'run/production'), num_epochs=base.EPOCHS, max_steps=total,
               lr_scheduler_total_steps=total, warmup_steps=base.WARMUP_EPOCHS * u, save_every=0,
               save_steps=save_steps, save_at_end=True, keep_latest_weights=4, keep_latest_state_only=True,
               resume=None, eval_every=0)
    if cfg['batch_size'] != base.MICRO or cfg['gradient_accumulation_steps'] != 1 or cfg['data']['val'] is not None:
        raise SystemExit('a18: unexpected batch / accumulation / validation in the H2 config')
    cfg['wandb'].update(enabled=False, mode='disabled', name='deltaflow-a18-40ep-augv2',
                        group='h2-action75-state109')
    old_train = cfg['data']['train']
    cfg['data']['train'] = dict(
        _target_=DATASET_TARGET, dataset_dir=str(DATA), stats_path=str(R / 'stats/normalization.json'),
        qwen_text_cache_dir=str(R / 'qwen-text'), qwen_context_len=old_train['qwen_context_len'],
        qwen_text_dim=old_train['qwen_text_dim'], video_cache_frames=64,
        image_augmentation=base.augmentation_config(),
        proprio_augmentation=dict(
            enabled=True, seed=20261106, joint_bias_degrees=2.0, joint_noise_std_degrees=0.25, delay_probability=0.15,
            max_delay_control_cycles=2, control_hz=30, recording_hz=30, history_drop_probability=0.15,
            affected_angle_channels=list(range(BODY)), velocity_channels=list(range(RAW_STATE, STATE)),
            action_label_noise=False, bias_scope='fixed_per_episode', body_joints='observation.state[0:31] incl. head',
            start_pose_library='not_applied_offline', render_randomization='not_applied_offline'),
        velocity_cache=str(R / 'stats/velocity.npz'), velocity_noise_std=0.05)
    m = cfg['model']
    if (m['expected_action_dim'], m['action_dit_config']['action_dim'], len(m['loss']['action_a2a']['source_noise_dim_mask']),
            m['proprio_dim'], m['action_state_dim'], m['expected_action_horizon']) != (ACTION, ACTION, ACTION, RAW_STATE, RAW_STATE, 16):
        raise SystemExit('a18: the H2 config is not the 78-D state / 75-D action Delta-flow it should be')
    m.update(_target_=MODEL_TARGET, proprio_dim=STATE, action_state_dim=STATE, state_only_schema_id=SCHEMA_ID)
    cfg['wholebody_contract'] = dict(
        task='a18', robot='h2_sharpa', world_size=base.WORLD, zero_stage=1, fresh_base=True,
        production_output_dir=str(R / 'run/production'), updates_per_epoch=u, train_episode_ids='0-99',
        statistics_sha256=sha(R / 'stats/normalization.json'), velocity_statistics_sha256=sha(R / 'stats/velocity.npz'),
        augmentation=base.AUG_VERSION, state_layout='joints75 + gravity3 + body_velocity31 = 109',
        action_layout='body_target31 + hands44 = 75', cameras=list(h2_pour.CAMERAS),
        measured_loader=dict(dataset_frames=FRAMES, raw_loader_batches=math.ceil(FRAMES / base.MICRO),
                             per_rank_microbatches=u, world_size=base.WORLD, microbatch=base.MICRO,
                             gradient_accumulation=1),
        source_config=str(source_yaml))
    for value in (cfg['output_dir'], cfg['data']['train']['stats_path'], cfg['data']['train']['qwen_text_cache_dir']):
        if value.startswith(str(OLD) + '/'):
            raise SystemExit(f'a18: config still points into the old experiment: {value}')
    (R / 'run/production.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    previous = OLD / 'normalization.json'
    dump(R / 'data-verified.json', dict(
        task='a18', dataset_dir=str(DATA), total_episodes_in_collection=EPISODES, episodes=EPISODES,
        train_episode_ids=list(range(EPISODES)), shelved_episode_ids=[], frames=FRAMES, updates_per_epoch=u,
        epochs=base.EPOCHS, max_steps=total, warmup_steps=base.WARMUP_EPOCHS * u, save_steps=save_steps,
        control_hz=30, recording_hz=30, statistics_sha256=sha(R / 'stats/normalization.json'),
        statistics_equal_previous_a18_run=previous.is_file() and sha(previous) == sha(R / 'stats/normalization.json'),
        velocity_statistics_sha256=sha(R / 'stats/velocity.npz'), augmentation=base.AUG_VERSION,
        state_dim=STATE, action_dim=ACTION, code=str(CODE), code_record_sha256=sha(CODE / 'deltaflow-code.json')))

    log('a18 model module check (no weights loaded)')
    import imagewam.h2_vel_delta as model_module
    factory = inspect.signature(model_module.create_imagewam_flux2_wholebody_delta)
    if (model_module.SCHEMA_ID != SCHEMA_ID or factory.parameters['action_state_dim'].default != STATE
            or factory.parameters['expected_action_dim'].default != ACTION):
        raise SystemExit('a18: widened model module is inconsistent')

    log('a18 loading two real augmented samples (CPU)')
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    dataset = instantiate(OmegaConf.load(R / 'run/production.yaml').data.train)
    if len(dataset) != FRAMES or list(dataset.episode_ids) != list(range(EPISODES)):
        raise SystemExit(f'a18: dataset length {len(dataset)} differs from {FRAMES}')
    first, sample = dataset[0], dataset[145]
    if (tuple(first['video'].shape) != (3, 2, 224, 448) or tuple(first['proprio'].shape) != (16, STATE)
            or tuple(sample['action'].shape) != (16, ACTION) or tuple(sample['past_action'].shape) != (16, ACTION)
            or torch.any(first['proprio'][:, RAW_STATE:]) or not torch.isfinite(sample['proprio']).all()
            or dataset.last_augmentation['visual']['boost'] is None):
        raise SystemExit('a18: sample schema check failed')
    dump(R / 'prepared.ok', dict(frames=FRAMES, updates_per_epoch=u, max_steps=total, save_steps=save_steps))
    log('a18 PREPARED', json.dumps(dict(frames=FRAMES, updates_per_epoch=u, max_steps=total, save_steps=save_steps,
                                        warmup=base.WARMUP_EPOCHS * u, state_dim=STATE, action_dim=ACTION)))


if __name__ == '__main__':
    build_code()
    prepare()
