# 2026-11-12: resumable A14 (100 episodes) download with HF rate-limit back-off.
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
repo, rev, out = 'humanoid-benchmark/a14-unscrew-bulb-expert-v1', 'b4390fc5a351ee4ede5cfced83da250e0e3b95d9', P + '/a14-unscrew-bulb-v1-b4390fc/data'
for attempt in range(60):
    try:
        snapshot_download(repo, repo_type='dataset', revision=rev, local_dir=out, token=tok, max_workers=4)
        print('A14_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
