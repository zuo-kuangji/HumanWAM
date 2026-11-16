"""B6 v6 download that checks completeness: snapshot_download silently returns the local dir when the Hub rate-limits
(429, 1000 API requests / 5 min), so after each attempt the expected file list is compared with the disk and the
missing files are fetched again after the rate-limit window."""
import sys, time
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
P = sys.argv[1]
R, REV = 'humanoid-benchmark/b6-pedal-bin', 'ad2f5bf07f2eeef765b8b401b401a8e4dc133505'
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
out = Path(P + '/b6-pedal-bin-v6-ad2f5bf/data')
want = None
for attempt in range(40):
    try:
        if want is None:
            info = HfApi(token=tok).dataset_info(R, revision=REV)
            want = [s.rfilename for s in info.siblings if s.rfilename.startswith(('data/', 'videos/', 'meta/'))
                    or s.rfilename in ('README.md', '.gitattributes')]
        miss = [f for f in want if not (out / f).is_file()]
        print(time.strftime('%H:%M:%S'), 'attempt', attempt, 'expected', len(want), 'missing', len(miss), flush=True)
        if not miss:
            print('B6V6_FULL', out, flush=True); break
        for i, f in enumerate(miss):
            hf_hub_download(R, f, repo_type='dataset', revision=REV, local_dir=str(out), token=tok)
            if i % 50 == 0:
                print(time.strftime('%H:%M:%S'), 'fetched', i + 1, '/', len(miss), flush=True)
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True)
        time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
