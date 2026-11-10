"""Verify a finished Delta-flow 40-epoch run; node 0 then writes R/deltaflow-complete.json.

    python deltaflow_check_final.py <run_root R> [--write]      exit 0 = complete, 1 = not complete
Checks: 32 finish receipts at global_step = 40 x updates_per_epoch; weights for epochs 10/20/30/40 present and
non-empty; exactly one training state left (the final step) with the full ZeRO-1 shard set.
"""
import argparse
import json
from pathlib import Path
import sys

WORLD, EPOCHS, EVERY = 32, 40, 10


def problems(root):
    found = []
    receipt = json.loads((root / 'data-verified.json').read_text())
    u = int(receipt['updates_per_epoch'])
    total = EPOCHS * u
    out = root / 'run' / 'production'
    for rank in range(WORLD):
        path = out / f'finish-rank-{rank:02d}.json'
        if not path.is_file():
            found.append(f'missing {path.name}')
            continue
        status = json.loads(path.read_text())
        if status.get('rank') != rank or status.get('global_step') != total:
            found.append(f'bad receipt {path.name}: {status}')
        elif status.get('resumed_from_step', 0) == 0 and status.get('optimizer_steps') != total:
            found.append(f'optimizer step count differs in {path.name}: {status}')
    weights = [out / 'checkpoints' / 'weights' / f'step_{e * u:06d}.pt' for e in range(EVERY, EPOCHS + 1, EVERY)]
    for path in weights:
        if not path.is_file() or path.stat().st_size <= 0:
            found.append(f'missing weights {path.name}')
    state_root = out / 'checkpoints' / 'state'
    states = sorted(p.name for p in state_root.iterdir() if p.is_dir()) if state_root.is_dir() else []
    final = state_root / f'step_{total:06d}'
    if states != [final.name]:
        found.append(f'expected only state {final.name}, found {states}')
    shards = list((final / 'pytorch_model').glob('*_optim_states.pt'))
    model = final / 'pytorch_model' / 'mp_rank_00_model_states.pt'
    if (len(shards) != WORLD or not model.is_file() or not (final / 'trainer_state.json').is_file()
            or any(p.stat().st_size <= 0 for p in [*shards, model] if p.is_file())):
        found.append(f'final training state incomplete ({len(shards)} optimizer shards)')
    return found, dict(weights=[str(p) for p in weights], state=str(final), optimizer_shards=len(shards),
                       updates_per_epoch=u, max_steps=total, train_episode_ids='0-99')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--write', action='store_true', help='write deltaflow-complete.json when complete')
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        found, summary = problems(root)
    except (OSError, ValueError, KeyError) as error:
        print(f'DELTAFLOW_INCOMPLETE {root.name}: {type(error).__name__}: {error}', flush=True)
        return 1
    if found:
        print(f'DELTAFLOW_INCOMPLETE {root.name}: ' + '; '.join(found[:8]), flush=True)
        return 1
    if args.write:
        target = root / 'deltaflow-complete.json'
        pending = root / 'deltaflow-complete.json.tmp'
        pending.write_text(json.dumps(summary, indent=2) + '\n')
        pending.replace(target)
    print(f'DELTAFLOW_COMPLETE {root.name}: epoch 10/20/30/40 weights + final state verified', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
