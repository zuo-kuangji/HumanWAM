import time
from huggingface_hub import snapshot_download
for i in range(30):
    try:
        p = snapshot_download("lerobot/pi05_base", revision="22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0", local_dir="/mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109/models/lerobot-pi05_base", max_workers=8)
        print("OK", p); break
    except Exception as e:
        print("retry", i, repr(e)[:300]); time.sleep(20)
