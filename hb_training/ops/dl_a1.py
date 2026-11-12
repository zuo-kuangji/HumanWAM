# 2026-11-13: A1 ground-object pickup (100 episodes) download, review extras/ skipped, HF rate-limit back-off.
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
repo, rev, out = 'humanoid-benchmark/a1-ground-object-pickup-expert-v1', 'f47f07ee79bc1ca47f802d531fba68b49565f1b6', P + '/a1-ground-object-pickup-v1-f47f07e/data'
for attempt in range(60):
    try:
        snapshot_download(repo, repo_type='dataset', revision=rev, local_dir=out, token=tok, max_workers=4,
                          allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', '.gitattributes'])
        print('A1_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
