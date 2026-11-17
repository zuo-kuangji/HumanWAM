"""2026-11-17 11:3x user: Delta Flow B7 black on the trimmed data (HF humanoid-benchmark/b7-black-clothes-hamper @949da46,
107 episodes / 83,660 frames; the fixed first 100 trained as for every B7 run, 78,170 frames). Identical to the B7 run it
replaces (deltaflow-runs/b7: prepare_deltaflow.py G1 recipe, code tree deltaflow-runs/HumanWAM, aug v2, 44-D proprio
without velocity, proprio aligned with GR00T A20, 32 GPUs x 8 = 256, no accumulation, 8000 steps); only the data
changes. Usage: prepare_deltaflow_b7t.py"""
import json
import sys
from pathlib import Path

H = Path(__file__).resolve().parent
sys.path.insert(0, str(H))
import prepare_deltaflow as base  # noqa: E402

base.TASKS['b7t'] = dict(data=base.P / 'b7-black-clothes-hamper-trim-949da46/data', old=None, control_hz=50,
                         total_episodes=107,
                         source='humanoid-benchmark/b7-black-clothes-hamper @949da46 (trimmed; first 100 of 107 episodes)')

if __name__ == '__main__':
    sys.argv = [sys.argv[0], 'b7t']
    base.main()
    R = base.ROOT / 'b7t'
    d = json.loads((R / 'data-verified.json').read_text())
    ref = json.loads((base.ROOT / 'b7' / 'data-verified.json').read_text())
    if (d['train_episode_ids'] != list(range(100)) or d['shelved_episode_ids'] != list(range(100, 107))
            or d['frames'] != 78170 or d['max_steps'] != 8000 or d['proprio_dim'] != 44 or d.get('joint_velocity')
            or d['code'] != ref['code'] or d.get('save_steps') != ref.get('save_steps')):
        raise SystemExit(f'b7t: unexpected receipt {d}')
    print('B7T_PREPARED_OK frames', d['frames'], 'updates_per_epoch', d['updates_per_epoch'], 'steps', d['max_steps'],
          'save', d.get('save_steps'), 'code', d['code'], flush=True)
