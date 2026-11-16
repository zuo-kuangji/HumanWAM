import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download
P = sys.argv[1]
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
for name, repo, rev in (('a3-fridge-open-v3-24882d5', 'humanoid-benchmark/a3-fridge-open-expert-v3', '24882d5b902e3b541471284de1bcfc611f35cac5'),
                        ('a7-cabinet-close-v3-c2e15b0', 'humanoid-benchmark/a7-cabinet-close-expert-v3', 'c2e15b01c643d4612591ea13a6a9fafef37a2c79')):
    out = P + '/' + name + '/data'
    for attempt in range(60):
        try:
            snapshot_download(repo, repo_type='dataset', revision=rev, local_dir=out, token=tok, max_workers=4,
                              allow_patterns=['data/*', 'videos/*', 'meta/*', 'reproduction/*', 'README.md', '.gitattributes'])
            print('DONE', name, out, flush=True); break
        except Exception as e:
            print(time.strftime('%H:%M:%S'), name, 'retry', attempt, type(e).__name__, str(e)[:120], flush=True); time.sleep(310)
    else:
        print('GAVE_UP', name, flush=True); sys.exit(1)
print('A37V3_DONE', flush=True)
