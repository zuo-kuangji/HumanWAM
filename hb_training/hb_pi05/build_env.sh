#!/bin/bash
# Build the isolated OpenPI env at $HB_ENV (default $G/envs/openpi) with Python 3.11, exactly the versions in
# OpenPI's uv.lock (pinned commit), downloaded through the Aliyun PyPI mirror, then apply transformers_replace.
# Run once on ONE node (NAS is shared):  bash build_env.sh
set -euo pipefail
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
MIRROR=${HB_PYPI_MIRROR:-http://mirrors.cloud.aliyuncs.com/pypi/simple/}
MIRROR_HOST=$(echo "$MIRROR" | sed -E 's#^[a-z]+://([^/:]+).*#\1#')
export UV_DEFAULT_INDEX=$MIRROR UV_INDEX_URL=$MIRROR UV_INSECURE_HOST=$MIRROR_HOST
export PIP_INDEX_URL=$MIRROR PIP_TRUSTED_HOST=$MIRROR_HOST
export UV_LINK_MODE=copy                         # README warns: hardlinks + transformers_replace would poison the uv cache
export UV_CACHE_DIR=$HB_G/envs/.uv-cache UV_PYTHON_INSTALL_DIR=$HB_G/envs/.uv-python
mkdir -p "$HB_G/envs"

# 0. OpenPI source at the pinned commit, without the A2A patch (must be a clean checkout).
if [ ! -d "$HB_OPENPI/.git" ] && [ ! -f "$HB_OPENPI/.git" ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "$HB_OPENPI"
  git -C "$HB_OPENPI" checkout "$HB_OPENPI_COMMIT"
fi
HEAD=$(git -C "$HB_OPENPI" rev-parse HEAD)
[ "$HEAD" = "$HB_OPENPI_COMMIT" ] || { echo "openpi HEAD $HEAD != $HB_OPENPI_COMMIT (git -C $HB_OPENPI checkout $HB_OPENPI_COMMIT)"; exit 2; }
git -C "$HB_OPENPI" diff --quiet HEAD || { echo "openpi checkout has modified tracked files (A2A patch?). Use a clean checkout:"; git -C "$HB_OPENPI" status --short | head; exit 3; }
echo "untracked files in openpi (should be empty or harmless):"; git -C "$HB_OPENPI" status --short --untracked-files=normal | head || true

# 1. uv (bootstrap via system pip into a private target dir).
UVT=$HB_G/envs/.uv-tool
if [ ! -x "$UVT/bin/uv" ]; then python3 -m pip install --upgrade --target "$UVT" uv; fi
UV="$UVT/bin/uv"; "$UV" --version

# 2. Python 3.11 (uv-managed, stored on NAS so every node sees the same interpreter) + venv.
"$UV" python install 3.11
[ -x "$HB_ENV/bin/python" ] || "$UV" venv --python 3.11 --seed "$HB_ENV"
PY="$HB_ENV/bin/python"; "$PY" -V

# 3. Locked third-party deps (uv.lock -> requirements, no dev group, workspace members installed separately).
REQ=$HB_ENV/openpi-locked-requirements.txt
( cd "$HB_OPENPI" && "$UV" export --frozen --no-hashes --no-dev --no-emit-project --no-emit-workspace \
    --format requirements-txt -o "$REQ" )
grep -E '^(torch|transformers|jax|lerobot|av|torchvision)[ =@]' "$REQ" || true
"$UV" pip install --python "$PY" -r "$REQ"
"$UV" pip install --python "$PY" --no-deps -e "$HB_OPENPI/packages/openpi-client" -e "$HB_OPENPI"

# 4. transformers_replace (README "PyTorch Support > Setup" step 3), copied into this venv's transformers only.
TF_DIR=$("$PY" -c 'import os, transformers; print(os.path.dirname(transformers.__file__))')
cp -r "$HB_OPENPI/src/openpi/models_pytorch/transformers_replace/"* "$TF_DIR/"
grep -q adarms "$TF_DIR/models/gemma/modeling_gemma.py" || { echo "transformers_replace not applied"; exit 4; }

# 5. Sanity.
"$PY" - <<'EOF'
import importlib, torch, jax, transformers
from transformers.models.siglip import check
assert check.check_whether_transformers_replace_is_installed_correctly(), transformers.__version__
for m in ["openpi.training.config", "openpi.training.data_loader", "openpi.models_pytorch.pi0_pytorch",
          "lerobot.common.datasets.lerobot_dataset", "av", "torchvision", "pyarrow.parquet", "sentencepiece", "safetensors"]:
    importlib.import_module(m)
print("torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(),
      "| jax", jax.__version__, jax.devices()[0].platform, "| transformers", transformers.__version__)
try:
    import hb_aug_v2; print("hb_aug_v2", hb_aug_v2.VERSION)
except ImportError:
    print("WARNING: hb_aug_v2.py not importable yet (copy it next to hb_pi05 scripts)")
EOF
echo BUILD_ENV_OK "$HB_ENV"
