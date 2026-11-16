import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
out = P + '/b6-pedal-bin-v6-ad2f5bf/data'
for attempt in range(60):
    try:
        snapshot_download('humanoid-benchmark/b6-pedal-bin', repo_type='dataset',
                          revision='ad2f5bf07f2eeef765b8b401b401a8e4dc133505', local_dir=out, token=tok, max_workers=8,
                          allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', '.gitattributes'])
        print('B6V6_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(120)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
