import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
out = P + '/a10-trolley-push-v3-6e410c5/data'
for attempt in range(60):
    try:
        snapshot_download('humanoid-benchmark/a10-trolley-push-expert-v3', repo_type='dataset',
                          revision='6e410c559b08095d8ff94980201dd6ff74f761f2', local_dir=out, token=tok, max_workers=4,
                          allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', 'SHA256SUMS', 'RELEASE.json', '.gitattributes'])
        print('A10V3_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
