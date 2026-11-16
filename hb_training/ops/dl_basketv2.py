# 2026-11-14 user: A20 basket transfer pre-reach trim v2 (100 episodes, 54,433 frames) for GR00T / Psi-Zero continuation.
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download, HfApi
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
repo = 'humanoid-benchmark/a20-basket-transfer-pre-reach-trim-expert-v2'
sha = HfApi(token=tok).dataset_info(repo).sha
assert sha.startswith('c4df4d40'), sha
out = P + '/basket-trimv2-c4df4d4/data'
print('SHA', sha, flush=True)
for attempt in range(60):
    try:
        snapshot_download(repo, repo_type='dataset', revision=sha, local_dir=out, token=tok, max_workers=4)
        print('BASKETV2_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
