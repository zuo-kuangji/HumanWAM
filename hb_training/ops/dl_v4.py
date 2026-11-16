"""A3 v4 / A7 v4 download with a completeness check (a rate-limited snapshot_download silently returns a partial dir):
compare the expected file list with the disk and fetch what is missing with 8 threads; prints V4_FULL <task> per task."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
api = HfApi(token=tok)
REPOS = (('a3v4', 'a3-fridge-open-v4-6393d4a', 'humanoid-benchmark/a3-fridge-open-expert-v4', '6393d4aaf8886f053130345d718b85d9d35160de'),
         ('a7v4', 'a7-cabinet-close-v4-90cc2b5', 'humanoid-benchmark/a7-cabinet-close-expert-v4', '90cc2b52c3436daaef7de7c7840a6e73a31c9f18'))
for task, name, repo, rev in REPOS:
    out = Path(f'{P}/{name}/data')
    want = None
    for attempt in range(30):
        try:
            if want is None:
                info = api.dataset_info(repo, revision=rev)
                want = [s.rfilename for s in info.siblings if s.rfilename.startswith(('data/', 'videos/', 'meta/'))
                        or s.rfilename in ('README.md', '.gitattributes')]
            miss = [f for f in want if not (out / f).is_file()]
            print(time.strftime('%H:%M:%S'), task, 'attempt', attempt, 'expected', len(want), 'missing', len(miss), flush=True)
            if not miss:
                print('V4_FULL', task, out, flush=True); break

            def get(f):
                hf_hub_download(repo, f, repo_type='dataset', revision=rev, local_dir=str(out), token=tok)
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(get, miss))
        except Exception as e:
            print(time.strftime('%H:%M:%S'), task, 'retry', attempt, type(e).__name__, str(e)[:120], flush=True)
            time.sleep(310)
    else:
        print('GAVE_UP', task, flush=True)
