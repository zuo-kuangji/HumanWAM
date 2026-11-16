# 2026-11-14: wait until humanoid-benchmark/b1-riceball-microwave-expert-v1 is fully uploaded (meta/info.json present,
# all episode parquet files present, no new commit for 20 min), then download that revision (no sim/ sidecar).
import json, sys, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, snapshot_download
P = sys.argv[1]
R = 'humanoid-benchmark/b1-riceball-microwave-expert-v1'
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
api = HfApi(token=tok)
t0 = time.time()
while True:
    try:
        info = api.dataset_info(R)
        files = [s.rfilename for s in info.siblings]
        age = time.time() - info.last_modified.timestamp()
        nparq = sum(f.startswith('data/') and f.endswith('.parquet') for f in files)
        nvid = sum(f.startswith('videos/') for f in files)
        ready = False
        if 'meta/info.json' in files and age > 1200:
            m = json.load(open(hf_hub_download(R, 'meta/info.json', repo_type='dataset', revision=info.sha, token=tok,
                                               local_dir='/tmp/b1meta-' + info.sha[:8])))
            ready = nvid > 0 and nparq >= int(m.get('total_chunks', 1)) and len(files) > 10
            print(time.strftime('%H:%M:%S'), 'sha', info.sha[:8], 'files', len(files), 'parquet', nparq, 'videos', nvid,
                  'episodes', m.get('total_episodes'), 'frames', m.get('total_frames'), 'age_min', round(age / 60),
                  'ready', ready, flush=True)
        else:
            print(time.strftime('%H:%M:%S'), 'waiting: sha', info.sha[:8], 'files', len(files), 'age_min', round(age / 60),
                  flush=True)
        if ready:
            out = f'{P}/b1-riceball-microwave-v1-{info.sha[:7]}/data'
            snapshot_download(R, repo_type='dataset', revision=info.sha, local_dir=out, token=tok, max_workers=4,
                              allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', 'SHA256SUMS.txt',
                                              '.gitattributes'])
            print('B1_DONE', info.sha, out, flush=True)
            break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'error', type(e).__name__, str(e)[:160], flush=True)
    if time.time() - t0 > 36 * 3600:
        print('GAVE_UP', flush=True)
        sys.exit(1)
    time.sleep(300)
