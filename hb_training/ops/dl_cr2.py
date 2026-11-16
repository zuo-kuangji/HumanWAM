"""A18 / A17 cropped v2 download with a completeness check (a rate-limited snapshot_download silently returns a partial
dir): compare the expected file list with the disk and fetch what is missing with 8 threads; prints CR2_FULL <task>."""
import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
api = HfApi(token=tok)
REPOS = (('a18cr2', 'a18-pour-water-cropped-v2-44cdd57', 'humanoid-benchmark/a18-pour-water-expert-cropped-v2', '44cdd5715f4681038bd0bdf377e7705585a831c3'),
         ('a17cr2', 'a17-unscrew-bottle-cropped-v2-3520b70', 'humanoid-benchmark/a17-unscrew-bottle-expert-cropped-v2', '3520b70f0cc30fec10c5d806167c791cbbfea718'))
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
                print('CR2_FULL', task, out, flush=True); break

            def get(f):
                hf_hub_download(repo, f, repo_type='dataset', revision=rev, local_dir=str(out), token=tok)
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(get, miss))
        except Exception as e:
            print(time.strftime('%H:%M:%S'), task, 'retry', attempt, type(e).__name__, str(e)[:120], flush=True)
            time.sleep(310)
    else:
        print('GAVE_UP', task, flush=True)
