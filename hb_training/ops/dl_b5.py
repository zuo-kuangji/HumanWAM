import json, sys, time
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
P = sys.argv[1]
R = 'humanoid-benchmark/b5-chair-push-expert-v1'
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
api = HfApi(token=tok)
info = api.dataset_info(R)
sha = info.sha
names = [s.rfilename for s in info.siblings]
top = {}
for x in names:
    k = '/'.join(x.split('/')[:2]) if x.count('/') >= 2 else x
    top[k] = top.get(k, 0) + 1
print('B5_SHA', sha, info.last_modified, 'files', len(names), json.dumps(top)[:800], flush=True)
out = f'{P}/b5-chair-push-v1-{sha[:7]}/data'
for attempt in range(60):
    try:
        snapshot_download(R, repo_type='dataset', revision=sha, local_dir=out, token=tok, max_workers=8,
                          allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', '.gitattributes'])
        print('B5_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(120)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
j = json.load(open(out + '/meta/info.json'))
ref = json.load(open(P + '/basket-current-dynamics-100-v21-b0ff522/data/meta/info.json'))
USED = ["observation.state", "observation.projected_gravity", "action.motion_token", "observation.inspire_left_hand_state",
        "observation.inspire_right_hand_state", "observation.images.ego_view", "observation.images.chest"]
key = lambda i, ks: {k: (i['features'][k].get('dtype'), i['features'][k].get('shape'), i['features'][k].get('names')) for k in ks if k in i['features']}
print('INFO', j.get('codebase_version'), j['total_episodes'], j['total_frames'], j['fps'], j.get('dataset_version'), j.get('robot_type'))
print('USED_SAME_AS_BASKET', key(j, USED) == key(ref, USED), [k for k in USED if k not in j['features']])
print('EXTRA', sorted(set(j['features']) - set(ref['features'])), 'MISSING', sorted(set(ref['features']) - set(j['features'])))
print('TASKS', open(out + '/meta/tasks.jsonl').read()[:300])
print('README', open(out + '/README.md').read().replace('\n', ' | ')[:3000])
