#!/bin/bash
# Build the isolated GR00T N1.7 training env (Python 3.12, torch 2.9.0 cu128) on NAS, pinned to Isaac-GR00T 7aa0bca.
set -euo pipefail
G=$(dirname "$(realpath "$0")")
R=$G/HumanWAM
ENV=$G/envs/groot-py312
export PIP_CACHE_DIR=/tmp/pipcache-groot PIP_DISABLE_PIP_VERSION_CHECK=1
cd "$R"
git submodule update --init third_party/Psi0
git -C third_party/Isaac-GR00T rev-parse HEAD
PYBIN=$(command -v python3.12 || true)
if [ -z "$PYBIN" ]; then
  PYBIN=$(command -v python3); "$PYBIN" -c 'import sys; assert sys.version_info[:2]==(3,12), sys.version'
fi
"$PYBIN" --version
[ -x "$ENV/bin/python" ] || "$PYBIN" -m venv "$ENV"
PIP="$ENV/bin/pip"
"$PIP" install -U pip wheel setuptools
"$PIP" install torch==2.9.0 torchvision==0.24.0
if [ -f /mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl.done ]; then "$PIP" install /mnt/nas/cross/Team/RobotAgent/Personal/baijiaqi/humanoid-benchmark-work/preflight-20261104T1648Z/policies-20261109/wheels/flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl; else echo FLASH_ATTN_SKIPPED_SDPA_FALLBACK; fi
"$PIP" install albumentations==1.4.18 "huggingface-hub[cli]" "opencv-python-headless>=4.5,<4.13" diffusers==0.35.1 dm-tree \
  lmdb==1.7.5 msgpack==1.1.0 msgpack-numpy==0.4.8 pandas==2.2.3 peft==0.17.1 termcolor==3.2.0 transformers==4.57.3 \
  tyro==0.9.17 click==8.1.8 datasets==3.6.0 "cryptography>=46.0.7" einops==0.8.1 gitpython==3.1.50 jsonlines==4.0.0 \
  gymnasium==1.2.2 matplotlib==3.10.1 numpy==1.26.4 omegaconf==2.3.0 scipy==1.15.3 torchcodec==0.8.0 wandb==0.23.0 \
  pyzmq==27.0.1 deepspeed==0.17.6
"$PIP" install --no-deps -e third_party/Isaac-GR00T
"$PIP" install --no-deps -e packages/humanwam
"$ENV/bin/python" - <<'PY'
import torch, transformers, flash_attn, deepspeed, torchcodec, gr00t
print("torch", torch.__version__, "cuda", torch.version.cuda, "transformers", transformers.__version__,
      "flash_attn", flash_attn.__version__, "deepspeed", deepspeed.__version__, "torchcodec", torchcodec.__version__)
from gr00t.experiment import launch_finetune  # noqa: F401
print("GROOT_ENV_OK")
PY
"$ENV/bin/pip" freeze > "$G/groot-env-freeze.txt"
