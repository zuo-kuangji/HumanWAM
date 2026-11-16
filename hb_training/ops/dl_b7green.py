# 2026-11-13: B7 green clothes -> hamper v1 (100 episodes) download, review extras skipped, HF rate-limit back-off.
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
repo, rev, out = 'humanoid-benchmark/b7-green-clothes-hamper-expert-v1', 'ada288c6bc57f2e54f862248da56eba880d4f811', P + '/b7-green-clothes-hamper-v1-ada288c/data'
for attempt in range(60):
    try:
        snapshot_download(repo, repo_type='dataset', revision=rev, local_dir=out, token=tok, max_workers=4,
                          allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', '.gitattributes'])
        print('B7GREEN_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
