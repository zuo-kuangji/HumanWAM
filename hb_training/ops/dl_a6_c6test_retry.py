# 2026-11-12: resumable download with HF rate-limit back-off (5915 small files hit the per-5-minute resolver limit).
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
for repo, rev, out, tag in (('humanoid-benchmark/door-pull-expert-v1', 'cf3ee9b5c7', P + '/a6-door-pull-v1-cf3ee9b/data', 'A6_DONE'),
                            ('humanoid-benchmark/c6-bathroom-door-fetch-expert-v1', '06c231d2ba', P + '/c6-test8-06c231d/data', 'C6TEST_DONE')):
    for attempt in range(60):
        try:
            snapshot_download(repo, repo_type='dataset', revision=rev, local_dir=out, token=tok, max_workers=4)
            print(tag, out, flush=True); break
        except Exception as e:
            print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
    else:
        print('GAVE_UP', repo, flush=True); sys.exit(1)
