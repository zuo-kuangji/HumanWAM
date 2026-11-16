# 2026-11-14: A14 v2 (planner command added) @8411a6da: data, meta, head + chest videos only (no wrist, no third_person, no sim/).
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
out = P + '/a14-unscrew-bulb-v2-8411a6d/data'
for attempt in range(60):
    try:
        snapshot_download('humanoid-benchmark/a14-unscrew-bulb-expert-v2', repo_type='dataset', revision='8411a6dac45d0097d6068a60307de0c2e3b621a7',
                          local_dir=out, token=tok, max_workers=4,
                          allow_patterns=['data/*', 'meta/*', 'README.md', 'SHA256SUMS.txt', '.gitattributes',
                                          'videos/*/observation.images.head/*', 'videos/*/observation.images.chest/*'])
        print('A14V2_DONE', out, flush=True); break
    except Exception as e:
        print(time.strftime('%H:%M:%S'), 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
else:
    print('GAVE_UP', flush=True); sys.exit(1)

