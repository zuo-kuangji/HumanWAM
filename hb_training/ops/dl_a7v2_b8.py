# 2026-11-14: A7 cabinet close v2 + B8 table wipe v1 (100 episodes each), sim/ sidecar skipped, HF rate-limit back-off.
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
for repo, rev, out, tag in (('humanoid-benchmark/a7-cabinet-close-expert-v2', 'aecf070ba960db618d51c20f2a63556e8749198e', P + '/a7-cabinet-close-v2-aecf070/data', 'A7V2'),
                            ('humanoid-benchmark/b8-table-wipe-expert-v1', '3b2e6180508f84f70fda8b33cd3e41c14225f0b5', P + '/b8-table-wipe-v1-3b2e618/data', 'B8')):
    for attempt in range(60):
        try:
            snapshot_download(repo, repo_type='dataset', revision=rev, local_dir=out, token=tok, max_workers=4,
                              allow_patterns=['data/*', 'videos/*', 'meta/*', 'README.md', 'SHA256SUMS.txt', '.gitattributes'])
            print(tag + '_DONE', out, flush=True); break
        except Exception as e:
            print(time.strftime('%H:%M:%S'), tag, 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
    else:
        print('GAVE_UP', tag, flush=True); sys.exit(1)
print('ALL_DONE', flush=True)
