#!/usr/bin/env python3
"""Build instruction-only Qwen3 embeddings with the existing FLUX.2 encoder."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-dir', required=True, type=Path)
    parser.add_argument('--cache-dir', required=True, type=Path)
    parser.add_argument('--qwen-model', required=True, type=Path, help='Local Qwen3-4B directory')
    parser.add_argument('--flux2-src', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0', help='One visible CUDA device, e.g. cuda:0 or cuda:3')
    args = parser.parse_args()
    data = args.dataset_dir.resolve(strict=True)
    model = args.qwen_model.resolve(strict=True)
    flux2 = args.flux2_src.resolve(strict=True)
    cache = args.cache_dir.resolve()
    if not model.is_dir() or not flux2.is_dir():
        raise ValueError('Use downloaded local Qwen weights and a local FLUX.2 source checkout')
    if cache == data or data in cache.parents or cache in data.parents:
        raise ValueError('Text cache must be outside the immutable dataset')
    tasks = [json.loads(line) for line in (data/'meta/tasks.jsonl').read_text().splitlines() if line.strip()]
    if len(tasks) != 1 or tasks[0].get('task_index') != 0 or not str(tasks[0].get('task', '')).strip():
        raise ValueError('Whole-body adapter expects one recorded nonempty instruction with task_index=0')
    if not args.device.startswith('cuda:') or not args.device[5:].isdigit():
        raise ValueError('--device must be cuda:N; this stage uses one GPU but does not train')
    from omegaconf import OmegaConf
    with tempfile.TemporaryDirectory(prefix='wholebody_qwen_') as folder:
        config = {'data': {'train': {'dataset_dirs': [str(data)], 'qwen_text_cache_dir': str(cache),
            'qwen_context_len': 128, 'override_instruction': tasks[0]['task']}},
            'model': {'variant': 'klein-base-4b', 'flux2_src_path': str(flux2), 'qwen3_model_spec': str(model)},
            'qwen_cache_batch_size': 1, 'qwen_cache_overwrite': False, 'qwen_cache_save_workers': 1,
            'hydra': {'run': {'dir': str(Path(folder)/'logs')}, 'output_subdir': None}}
        OmegaConf.save(OmegaConf.create(config), Path(folder)/'qwen_cache.yaml')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.device[5:], LOCAL_RANK='0', RANK='0', WORLD_SIZE='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        env['PYTHONPATH'] = str(REPO/'src')+os.pathsep+env.get('PYTHONPATH', '')
        subprocess.run([sys.executable, '-B', str(REPO/'scripts/flux2/precompute_flux2_qwen3_embeds.py'),
            '--config-path', folder, '--config-name', 'qwen_cache'], cwd=REPO, env=env, check=True)


if __name__ == '__main__':
    main()
