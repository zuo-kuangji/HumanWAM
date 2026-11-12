# 2026-11-12: A17 wrist-camera videos (all H2 tasks now train with head + chest + wrist); HF rate-limit back-off.
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
for attempt in range(60):
    try:
        snapshot_download('humanoid-benchmark/a17-unscrew-bottle-expert-v1', repo_type='dataset', revision='407e6e2',
                          local_dir=P + '/a17-unscrew-bottle-v1-407e6e2/data', token=tok, max_workers=4,
                          allow_patterns=['videos/*/observation.images.wrist/*'])
        print('A17_WRIST_DONE', flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)
