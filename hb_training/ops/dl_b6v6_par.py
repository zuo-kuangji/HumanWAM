import sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
P = sys.argv[1]
R, REV = 'humanoid-benchmark/b6-pedal-bin', 'ad2f5bf07f2eeef765b8b401b401a8e4dc133505'
tok = Path('/root/.hb-secrets/hf_token').read_text().strip()
out = Path(P + '/b6-pedal-bin-v6-ad2f5bf/data')
info = HfApi(token=tok).dataset_info(R, revision=REV)
miss = [s.rfilename for s in info.siblings if s.rfilename.startswith('videos/') and not (out / s.rfilename).is_file()]
miss = miss[::-1]
print(time.strftime('%H:%M:%S'), 'parallel missing', len(miss), flush=True)
def get(f):
    for a in range(5):
        try:
            if not (out / f).is_file():
                hf_hub_download(R, f, repo_type='dataset', revision=REV, local_dir=str(out), token=tok)
            return 1
        except Exception as e:
            print('err', f, type(e).__name__, str(e)[:80], flush=True); time.sleep(60)
    return 0
with ThreadPoolExecutor(8) as ex:
    n = sum(ex.map(get, miss))
print(time.strftime('%H:%M:%S'), 'PAR_DONE', n, len(miss), flush=True)
