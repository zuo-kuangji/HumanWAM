# 2026-11-13: A10 trolley push v2 (150 episodes) download, review extras/ skipped, HF rate-limit back-off.
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
repo, rev, out = 'humanoid-benchmark/a10-trolley-push-expert-v2', 'd3abb4cdd2f2d5db01a69f7e913e85182ca701b0', P + '/a10-trolley-push-v2-d3abb4c/data'
for attempt in range(60):
    try:
        snapshot_download(repo, repo_type='dataset', revision=rev, local_dir=out, token=tok, max_workers=4,
                          allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', '.gitattributes'])
        print('A10V2_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
