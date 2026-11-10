#!/bin/bash
# Build the Psi0 training env at $G/envs/psi0 (Python 3.11, versions pinned by the upstream uv.lock),
# install flash-attn 2.7.4.post1 (prebuilt wheel), download the public CLIP text encoder and write the
# non-secret $PSI/.env that scripts/train.py requires (assert load_dotenv()).
# Run once on ONE node (NAS is shared):   bash $G/hb_psi0/build_env.sh
set -euo pipefail
H=$(dirname "$(realpath "$0")")
G=${G:-$(dirname "$H")}
PSI=$G/HumanWAM/third_party/Psi0
ENV=$G/envs/psi0
PSI_COMMIT=cda3fdfafe47cb6fa75400d013de1bd36e3cea05
CLIP_REV=32bd64288804d66eefd0ccbe215aa642df71cc41          # openai/clip-vit-large-patch14 (public)
FA_WHEEL_VER=2.7.4.post1                                     # README: flash_attn==2.7.4.post1
MIRROR=${PIP_MIRROR:-http://mirrors.cloud.aliyuncs.com/pypi/simple/}
MHOST=$(echo "$MIRROR" | sed -E 's#^[a-z]+://([^/:]+).*#\1#')
export UV_INDEX_URL=$MIRROR UV_DEFAULT_INDEX=$MIRROR UV_INSECURE_HOST=$MHOST PIP_INDEX_URL=$MIRROR PIP_TRUSTED_HOST=$MHOST
export UV_CACHE_DIR=$G/uv-cache UV_PYTHON_INSTALL_DIR=$G/uv-python UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600
export GIT_LFS_SKIP_SMUDGE=1 HF_HOME=$G/hf-home
mkdir -p "$G/envs" "$G/wheels" "$G/models" "$G/hf-home"
echo "== G=$G PSI=$PSI ENV=$ENV mirror=$MIRROR"

# ---- 0. pinned Psi0 source ------------------------------------------------------------------------
if [ ! -f "$PSI/pyproject.toml" ]; then
  echo "== cloning Psi0 into $PSI"
  mkdir -p "$PSI"
  git clone https://github.com/physical-superintelligence-lab/Psi0.git "$PSI"
  git -C "$PSI" checkout "$PSI_COMMIT"
fi
if [ "$(git -C "$PSI" rev-parse --show-toplevel 2>/dev/null || true)" = "$(realpath "$PSI")" ]; then
  head=$(git -C "$PSI" rev-parse HEAD)
  [ "$head" = "$PSI_COMMIT" ] || { echo "FATAL: Psi0 at $head, expected $PSI_COMMIT"; exit 3; }
  echo "== Psi0 commit OK ($head)"
else
  echo "WARN: $PSI is not its own git checkout; commit $PSI_COMMIT NOT verified (check HumanWAM third_party pin)"
fi
grep -q 'requires-python = "==3.11' "$PSI/pyproject.toml" || echo "WARN: unexpected requires-python"

# ---- 1. uv (bootstrapped with the system python3.12, never touching system site-packages) ----------
UVBOOT=$G/envs/uv-boot
if [ ! -x "$UVBOOT/bin/uv" ]; then
  python3 -m venv "$UVBOOT"
  "$UVBOOT/bin/pip" install -q -i "$MIRROR" --trusted-host "$MHOST" uv
fi
UV=$UVBOOT/bin/uv
"$UV" --version

# ---- 2. Python 3.11 ---------------------------------------------------------------------------------
if ! PY311=$("$UV" python find 3.11 2>/dev/null); then
  echo "== uv python install 3.11 (python-build-standalone from GitHub releases)"
  if ! "$UV" python install 3.11; then
    echo "== fallback 1: npmmirror copy of python-build-standalone"
    UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone "$UV" python install 3.11 || {
      echo "== fallback 2: micromamba (conda-forge) python=3.11"
      MM=$G/envs/micromamba
      mkdir -p "$MM"
      curl -fsSL -o "$MM/micromamba" https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64
      chmod +x "$MM/micromamba"
      "$MM/micromamba" create -y -r "$MM/root" -p "$G/envs/py311-base" -c conda-forge python=3.11
      export PATH="$G/envs/py311-base/bin:$PATH"
    }
  fi
  PY311=$("$UV" python find 3.11)
fi
echo "== python: $PY311"

# ---- 3. venv + pinned requirements (exported offline from upstream uv.lock) ----------------------------
[ -x "$ENV/bin/python" ] || "$UV" venv --python "$PY311" "$ENV"
REQ=$ENV/psi0-requirements.txt
if "$UV" export --project "$PSI" --frozen --no-hashes --no-emit-project --no-default-groups \
      --group psi --group serve --format requirements-txt -o "$REQ.raw" >/dev/null; then
  echo "== exported $(grep -c '==' "$REQ.raw") pins from uv.lock"
else
  echo "WARN: uv export failed; using hand-pinned fallback (looser transitive pins)"
  cp "$H/requirements-fallback.txt" "$REQ.raw"
fi
# evdev (teleop keyboard, needs kernel headers to build) is not used by training
SKIP=${HB_SKIP_PKGS:-evdev}
cp "$REQ.raw" "$REQ"
for p in $SKIP; do sed -i -E "/^${p}([=<> @;]|$)/d" "$REQ"; done
"$UV" pip install --python "$ENV/bin/python" -r "$REQ"
"$UV" pip install --python "$ENV/bin/python" --no-deps -e "$PSI" || echo "WARN: editable psi install failed; PYTHONPATH=$PSI/src is used anyway"

# ---- 4. flash-attn (Qwen3VL is loaded with attn_implementation=flash_attention_2) ----------------------
ABI=$("$ENV/bin/python" -c "import torch;print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
WHL=flash_attn-${FA_WHEEL_VER}+cu12torch2.7cxx11abi${ABI}-cp311-cp311-linux_x86_64.whl
if ! "$ENV/bin/python" -c "import flash_attn" 2>/dev/null; then
  [ -s "$G/wheels/$WHL" ] || curl -fL -o "$G/wheels/$WHL" \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_WHEEL_VER}/${WHL//+/%2B}"
  "$UV" pip install --python "$ENV/bin/python" --no-deps "$G/wheels/$WHL"
fi

# ---- 5. CLIP text encoder for the pooled instruction embedding (finetune.py:93-107) ---------------
CLIP=$G/models/clip-vit-large-patch14
if [ ! -s "$CLIP/model.safetensors" ]; then
  "$ENV/bin/hf" download openai/clip-vit-large-patch14 --revision "$CLIP_REV" --local-dir "$CLIP" \
    --include config.json model.safetensors tokenizer.json tokenizer_config.json vocab.json merges.txt \
              special_tokens_map.json preprocessor_config.json
fi

# ---- 6. non-secret .env (scripts/train.py: assert load_dotenv(); never overrides exported vars) -------
cat > "$PSI/.env" <<EOF
# Humanoid Benchmark Psi0 .env -- NO secrets (no HF_TOKEN / WANDB_API_KEY). Generated by hb_psi0/build_env.sh
# PSI_HOME: base for relative paths in Psi0ModelConfig/LerobotDataConfig validators (we pass absolute paths)
PSI_HOME=$G/psi0-home
# DATA_HOME: psi.utils.resolve_path fallback for relative data paths
DATA_HOME=$G/psi0-data
# HF_HOME / TORCH_HOME: shared HF cache on NAS (models are loaded from local dirs; HF_HUB_OFFLINE=1 in the launcher)
HF_HOME=$G/hf-home
TORCH_HOME=$G/hf-home/torch
# HF_LEROBOT_HOME: lerobot default root (unused: views are opened with an explicit root)
HF_LEROBOT_HOME=$G/psi0-data
UV_CACHE_DIR=$G/uv-cache
OMP_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false
TF_CPP_MIN_LOG_LEVEL=3
AV_LOG_LEVEL=quiet
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
NO_ALBUMENTATIONS_UPDATE=1
WANDB_MODE=disabled
# A800 = sm_80. (.env.sample's CUDA_LAUNCH_BLOCKING=true is deliberately NOT set: it serializes every kernel.)
TORCH_CUDA_ARCH_LIST=8.0
EOF
echo "== wrote $PSI/.env"

# ---- 7. import check ----------------------------------------------------------------------------------
PYTHONPATH="$PSI/src:$H" HB_AUG_DIR="${HB_AUG_DIR:-$G/hb_groot}" "$ENV/bin/python" - <<'EOF'
import sys, numpy, torch, transformers, accelerate, diffusers, datasets, tyro, flash_attn, qwen_vl_utils
from psi.data.lerobot.compat import LEROBOT_LAYOUT
import psi.trainers.finetune, hb_psi0_config
print("python", sys.version.split()[0], "numpy", numpy.__version__, "torch", torch.__version__, "cuda", torch.version.cuda,
      "abi", torch._C._GLIBCXX_USE_CXX11_ABI, "transformers", transformers.__version__, "accelerate", accelerate.__version__,
      "diffusers", diffusers.__version__, "datasets", datasets.__version__, "flash_attn", flash_attn.__version__,
      "lerobot layout", LEROBOT_LAYOUT, "gpu", torch.cuda.is_available())
assert sys.version_info[:2] == (3, 11) and numpy.__version__.startswith("1.") and torch.__version__.startswith("2.7.0")
assert transformers.__version__ == "4.57.1"
EOF
for d in "$G/models/psi0/psi0/postpre.sonic1.1.unifolm.2609181726.40k" "$G/models/qwen3-vl-2b-instruct"; do
  [ -d "$d" ] || echo "WARN: $d not present yet"
done
echo "BUILD_ENV_OK $ENV"
