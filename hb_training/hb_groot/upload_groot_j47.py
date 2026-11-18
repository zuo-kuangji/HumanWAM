"""2026-11-18 user: upload the GR00T joint47 B6 v5 run (groot-runs-j47/b6v5j47) to the public humanoid-benchmark/groot
under b6-pedal-bin/joint47-v5-150/step-NNNNNN (the last two checkpoints, weights + processor + a protocol.json; no
optimizer / resume state), with the data receipt, modality, the training code and the logs; sha256-verified."""
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

G = Path('/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109')
R = G / 'groot-runs-j47/b6v5j47'
D = G / 'groot-data/b6v5j47'
H = G / 'hb_groot'
REPO, BASE = 'humanoid-benchmark/groot', 'b6-pedal-bin/joint47-v5-150'
FILES = ('config.json', 'model.safetensors.index.json', 'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
         'processor_config.json', 'statistics.json', 'embodiment_id.json', 'trainer_state.json')
api = HfApi(token=Path('/root/.hb-secrets/hf_token').read_text().strip())


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


done = json.loads((R / 'production-complete.json').read_text())
receipt = json.loads((D / 'hb-dataset-receipt.json').read_text())
ops, check = [], {}
for step in (3500, 4000):
    ck = R / f'production/checkpoint-{step}'
    dst = f'{BASE}/step-{step:06d}'
    files = {n: dict(size=(ck / n).stat().st_size, sha256=sha(ck / n)) for n in FILES}
    proto = dict(model='GR00T N1.7 (Gr00tN1d7), full fine-tune (tune_llm, tune_visual)', task='B6 pedal-bin (G1)',
                 task_dir=dst, robot='Unitree G1', global_step=step, epoch_completed=None,
                 training_data='b6-pedal-bin v5/ (HF humanoid-benchmark/b6-pedal-bin @ad2f5bf0), all 150 episodes',
                 episodes=receipt['episodes'], frames=receipt['frames'],
                 action_contract='HumanWAM joint47 v2: body29 (measured joints) | Inspire left6 | right6 | root rotation6D '
                                 'relative to the current heading; 40 rows at offsets 0,2,...,78 (stride 2); execute 16',
                 action_groups=dict(body29=29, left_hand_target=6, right_hand_target=6, root_rot6d=6), action_chunk=40,
                 embodiment_tag='new_embodiment',
                 state_groups=dict(left_leg=6, right_leg=6, waist=3, left_arm=7, right_arm=7, left_hand=6, right_hand=6,
                                   projected_gravity=3),
                 cameras=['observation.images.ego_view', 'observation.images.chest'],
                 training=done, augmentation='Humanoid Benchmark augmentation spec v2 + proprio perturbation (hb_aug_v2)',
                 files=files, uploaded_utc=time.strftime('%FT%TZ', time.gmtime()))
    pj = R / f'protocol-{step}.json'
    pj.write_text(json.dumps(proto, indent=1))
    check[dst] = files
    ops += [CommitOperationAdd(f'{dst}/{n}', str(ck / n)) for n in FILES]
    ops.append(CommitOperationAdd(f'{dst}/protocol.json', str(pj)))
extra = {'dataset_meta/hb-dataset-receipt.json': D / 'hb-dataset-receipt.json', 'dataset_meta/modality.json': D / 'meta/modality.json',
         'dataset_meta/info.json': D / 'meta/info.json', 'production-complete.json': R / 'production-complete.json',
         'code/hb_j47.py': H / 'hb_j47.py', 'code/g1_joint47_h40.py': H / 'g1_joint47_h40.py',
         'code/prepare_groot_j47.py': H / 'prepare_groot_j47.py', 'code/groot_quad16_j47.sh': H / 'groot_quad16_j47.sh',
         'code/verify_j47_cpu.py': H / 'verify_j47_cpu.py', 'code/hb_launch.py': H / 'hb_launch.py', 'code/hb_aug_v2.py': H / 'hb_aug_v2.py',
         'logs/production.log': R / 'logs/production.log'}
ops += [CommitOperationAdd(f'{BASE}/{k}', str(v)) for k, v in extra.items() if Path(v).is_file()]
c = api.create_commit(REPO, operations=ops, commit_message=f'{BASE}: GR00T joint47 on B6 v5 (150 episodes), steps 3500 and 4000')
tree = {f.path: f for f in api.list_repo_tree(REPO, path_in_repo=BASE, recursive=True, expand=True) if hasattr(f, 'size')}
bad = [f'{d}/{n}' for d, files in check.items() for n, v in files.items()
       if f'{d}/{n}' not in tree or tree[f'{d}/{n}'].size != v['size']
       or (tree[f'{d}/{n}'].lfs is not None and tree[f'{d}/{n}'].lfs.sha256 != v['sha256'])]
print('UPLOADED' if not bad else 'VERIFY_FAILED', c.oid, bad, flush=True)
