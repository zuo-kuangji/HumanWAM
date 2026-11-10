#!/bin/bash
# Build the DiT4DiT training env at $HB_ENV (default $G/envs/dit4dit). Run once on ONE node (NAS is shared):
#   mkdir -p $G/envs && bash $G/hb_dit4dit/build_env.sh 2>&1 | tee $G/envs/build_dit4dit.log      # -> BUILD_ENV_OK
# 0. DiT4DiT source at the HumanWAM pin (Mondo-Robotics/DiT4DiT @ 66a6f3a) in $HB_D4D_SRC (sparse: no media/robot code)
# 1. uv (reused from the other builds when present)  2. Python 3.11 (uv-managed, reused) or the system python3.12
# 3. torch 2.7.0 / torchvision 0.22.0 (upstream requirements.txt pins; PyPI cu126 wheels from the Aliyun mirror)
# 4. the rest of upstream requirements.txt with exact pins, minus packages not used by training (decord, pytorch3d,
#    retinaface, junk pins) and minus the torch/nvidia/triton lines (the list mixes cu124 CUDA libs with torch 2.7.0)
# 5. deepspeed 0.18.4 (no build isolation, no prebuilt ops)  6. diffusers @ 3996788b (codeload tarball / git; fallback
#    diffusers==0.37.0 from the mirror, which contains that commit)  7. import checks.
set -euo pipefail
unset RANK WORLD_SIZE MASTER_ADDR MASTER_PORT LOCAL_RANK LOCAL_WORLD_SIZE GROUP_RANK ROLE_RANK
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
MIRROR=${HB_PYPI_MIRROR:-http://mirrors.cloud.aliyuncs.com/pypi/simple/}
MHOST=$(echo "$MIRROR" | sed -E 's#^[a-z]+://([^/:]+).*#\1#')
export UV_DEFAULT_INDEX=$MIRROR UV_INDEX_URL=$MIRROR UV_INSECURE_HOST=$MHOST PIP_INDEX_URL=$MIRROR PIP_TRUSTED_HOST=$MHOST
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=600 UV_CACHE_DIR=${UV_CACHE_DIR:-$HB_G/uv-cache}
export GIT_LFS_SKIP_SMUDGE=1
mkdir -p "$HB_G/envs" "$HB_G/wheels" "$HB_G/models"
echo "== G=$HB_G SRC=$HB_D4D_SRC ENV=$HB_ENV mirror=$MIRROR"

# ---- 0. pinned DiT4DiT source -------------------------------------------------------------------------------------
if [ ! -f "$HB_D4D_SRC/DiT4DiT/training/train.py" ]; then
  echo "== fetching DiT4DiT $HB_D4D_COMMIT (sparse, partial clone) into $HB_D4D_SRC"
  mkdir -p "$HB_D4D_SRC"
  ( cd "$HB_D4D_SRC"
    [ -d .git ] || git init -q
    git remote get-url origin >/dev/null 2>&1 || git remote add origin https://github.com/Mondo-Robotics/DiT4DiT.git
    git config core.sparseCheckout true
    printf '/DiT4DiT/\n/deployment/\n/requirements.txt\n/pyproject.toml\n/LICENSE\n/README.md\n' > .git/info/sparse-checkout
    timeout 3600 git fetch --depth 1 --filter=blob:none origin "$HB_D4D_COMMIT"
    git checkout -q FETCH_HEAD ) || {
    echo "== git fetch failed; trying the codeload tarball (~260 MB incl. media)"
    TB=$HB_G/wheels/DiT4DiT-$HB_D4D_COMMIT.tar.gz
    [ -s "$TB" ] || curl -fL --retry 5 --max-time 7200 -o "$TB" "https://codeload.github.com/Mondo-Robotics/DiT4DiT/tar.gz/$HB_D4D_COMMIT"
    tar -xzf "$TB" -C "$HB_D4D_SRC" --strip-components=1 "DiT4DiT-$HB_D4D_COMMIT/DiT4DiT" "DiT4DiT-$HB_D4D_COMMIT/deployment" \
      "DiT4DiT-$HB_D4D_COMMIT/requirements.txt" "DiT4DiT-$HB_D4D_COMMIT/pyproject.toml" "DiT4DiT-$HB_D4D_COMMIT/LICENSE"
    echo "$HB_D4D_COMMIT" > "$HB_D4D_SRC/.hb_tarball_commit"
  }
fi
if git -C "$HB_D4D_SRC" rev-parse HEAD >/dev/null 2>&1 && [ "$(git -C "$HB_D4D_SRC" rev-parse --show-toplevel)" = "$(realpath "$HB_D4D_SRC")" ]; then
  HEAD=$(git -C "$HB_D4D_SRC" rev-parse HEAD)
  [ "$HEAD" = "$HB_D4D_COMMIT" ] || { echo "FATAL: DiT4DiT at $HEAD, expected $HB_D4D_COMMIT"; exit 3; }
  git -C "$HB_D4D_SRC" diff --quiet HEAD -- DiT4DiT || { echo "FATAL: modified tracked files in $HB_D4D_SRC/DiT4DiT:"; git -C "$HB_D4D_SRC" status --short DiT4DiT | head; exit 3; }
  echo "== DiT4DiT commit OK ($HEAD), clean"
elif [ "$(cat "$HB_D4D_SRC/.hb_tarball_commit" 2>/dev/null)" = "$HB_D4D_COMMIT" ]; then
  echo "== DiT4DiT from codeload tarball $HB_D4D_COMMIT"
else
  echo "WARN: $HB_D4D_SRC is not its own git checkout; commit $HB_D4D_COMMIT NOT verified (check the HumanWAM submodule pin)"
fi
for f in DiT4DiT/training/train.py DiT4DiT/model/framework/DiT4DiT.py DiT4DiT/model/modules/vlm/Cosmos25.py \
         DiT4DiT/config/real_robot/dit4dit_g1.yaml requirements.txt; do
  [ -f "$HB_D4D_SRC/$f" ] || { echo "FATAL: $HB_D4D_SRC/$f missing"; exit 3; }
done

# ---- 1. uv --------------------------------------------------------------------------------------------------------
UV=""
for c in "$HB_G/envs/uv-boot/bin/uv" "$HB_G/envs/.uv-tool/bin/uv" "$HB_G/envs/.uv-tool-d4d/bin/uv"; do
  [ -x "$c" ] && { UV=$c; break; }
done
if [ -z "$UV" ]; then
  python3 -m pip install -q --target "$HB_G/envs/.uv-tool-d4d" -i "$MIRROR" --trusted-host "$MHOST" uv
  UV=$HB_G/envs/.uv-tool-d4d/bin/uv
fi
"$UV" --version   # standalone binary; no PYTHONPATH needed

# ---- 2. Python 3.11 ------------------------------------------------------------------------------------------------
PYBASE=""
for d in "$HB_G/uv-python" "$HB_G/envs/.uv-python"; do
  if PYBASE=$(UV_PYTHON_INSTALL_DIR=$d "$UV" python find --managed-python 3.11 2>/dev/null) || \
     PYBASE=$(UV_PYTHON_INSTALL_DIR=$d "$UV" python find 3.11 2>/dev/null); then
    echo "== reusing python $PYBASE"; break
  fi
  PYBASE=""
done
if [ -z "$PYBASE" ]; then
  export UV_PYTHON_INSTALL_DIR=$HB_G/uv-python
  if "$UV" python install 3.11 || UV_PYTHON_INSTALL_MIRROR=https://registry.npmmirror.com/-/binary/python-build-standalone "$UV" python install 3.11; then
    PYBASE=$("$UV" python find 3.11)
  else
    PYBASE=/usr/local/bin/python3.12
    echo "WARN: no Python 3.11 available, falling back to $PYBASE (static build; fine for this stack: no torchcodec)"
  fi
fi
echo "== base python: $PYBASE ($("$PYBASE" -V 2>&1))"
[ -x "$HB_ENV/bin/python" ] || "$UV" venv --seed --python "$PYBASE" "$HB_ENV"
PY=$HB_ENV/bin/python
UVP=("$UV" pip install --python "$PY")

# ---- 3. torch (exact upstream versions) ------------------------------------------------------------------------------
CONS=$HB_ENV/hb-constraints.txt
printf 'torch==2.7.0\ntorchvision==0.22.0\nnumpy==1.26.4\n' > "$CONS"
"${UVP[@]}" torch==2.7.0 torchvision==0.22.0

# ---- 4. upstream requirements.txt (filtered) -------------------------------------------------------------------------
REQ=$HB_ENV/dit4dit-requirements.txt
SKIP='^(torch|torchvision|triton|nvidia-[a-z0-9-]+|decord|eva-decord|pipablepytorch3d|retinaface-py|deepspeed|diffusers|opencv-python|style|update|websocket|pytest|iniconfig|pluggy|gevent|greenlet|zope\.event|zope\.interface)([=<> @;]|$)'
grep -v -E "$SKIP" "$HB_D4D_SRC/requirements.txt" | grep -v -E '^\s*(#|$)' > "$REQ"
# upstream pins sympy==1.13.1 (torch 2.6 era) but its torch==2.7.0 pin needs sympy>=1.13.3 -> unsatisfiable as-is
sed -i -E 's/^sympy==.*/sympy==1.13.3/' "$REQ"
echo "== installing $(wc -l < "$REQ") pinned requirements (skipped: $(grep -c -E "$SKIP" "$HB_D4D_SRC/requirements.txt"))"
if ! "${UVP[@]}" -c "$CONS" -r "$REQ"; then
  echo "WARN: full pinned set failed; installing the training subset with the same pins"
  grep -E '^(accelerate|transformers|tokenizers|huggingface-hub|hf-xet|safetensors|omegaconf|antlr4-python3-runtime|numpy|pandas|pyarrow|pillow|tqdm|wandb|PyYAML|av|einops|timm|pydantic|pydantic_core|hjson|ninja|py-cpuinfo|psutil|msgpack|scipy|sentencepiece|regex|requests|filelock|fsspec|packaging|typing_extensions|Jinja2|MarkupSafe|sympy|networkx|mpmath|opencv-python-headless|imageio|einx|tiktoken|protobuf)==' \
    "$HB_D4D_SRC/requirements.txt" > "$REQ.min"
  sed -i -E 's/^sympy==.*/sympy==1.13.3/' "$REQ.min"
  "${UVP[@]}" -c "$CONS" -r "$REQ.min"
fi

# ---- 5. deepspeed (pure python install; ZeRO-2 + torch AdamW needs no compiled op) -----------------------------------
"${UVP[@]}" setuptools wheel
DS_BUILD_OPS=0 DS_SKIP_CUDA_CHECK=1 "${UVP[@]}" -c "$CONS" --no-build-isolation deepspeed==0.18.4

# ---- 6. diffusers @ pinned commit ----------------------------------------------------------------------------------
DIFF_OK=0
TB=$HB_G/wheels/diffusers-$HB_DIFFUSERS_COMMIT.tar.gz
if [ ! -s "$TB" ]; then
  curl -fL --retry 3 --max-time 2400 -o "$TB.part" "https://codeload.github.com/huggingface/diffusers/tar.gz/$HB_DIFFUSERS_COMMIT" \
    && mv "$TB.part" "$TB" || rm -f "$TB.part"
fi
if [ -s "$TB" ] && "${UVP[@]}" -c "$CONS" "diffusers @ file://$TB"; then
  DIFF_OK=1
elif timeout 3600 "${UVP[@]}" -c "$CONS" "diffusers @ git+https://github.com/huggingface/diffusers.git@$HB_DIFFUSERS_COMMIT"; then
  DIFF_OK=1
else
  echo "WARN: diffusers@$HB_DIFFUSERS_COMMIT not reachable; installing diffusers==0.37.0 (first release containing it)"
  "${UVP[@]}" -c "$CONS" diffusers==0.37.0
fi
echo "$DIFF_OK" > "$HB_ENV/hb-diffusers-pinned"

# ---- 7. checks -----------------------------------------------------------------------------------------------------
"$PY" - <<'EOF'
import importlib, os, sys
import numpy, torch, torchvision, transformers, accelerate, diffusers, deepspeed, omegaconf, wandb, av, pyarrow
from diffusers import AutoencoderKLWan, CosmosTransformer3DModel, UniPCMultistepScheduler
from diffusers.pipelines.cosmos.pipeline_cosmos2_5_predict import Cosmos2_5_PredictBasePipeline
from transformers import Qwen2_5_VLForConditionalGeneration, get_scheduler
sys.path.insert(0, os.environ["HB_D4D_SRC"])
for m in ["DiT4DiT.model.framework", "DiT4DiT.model.framework.DiT4DiT", "DiT4DiT.model.modules.vlm.Cosmos25",
          "DiT4DiT.model.modules.action_model.ActionDiT", "DiT4DiT.training.trainer_utils.trainer_tools",
          "DiT4DiT.training.trainer_utils.config_tracker", "DiT4DiT.dataloader"]:
    importlib.import_module(m)
from DiT4DiT.model.tools import FRAMEWORK_REGISTRY
assert "DiT4DiT" in FRAMEWORK_REGISTRY._registry, FRAMEWORK_REGISTRY._registry
import hb_aug_v2, hb_d4d_data, hb_d4d_config, hb_patches
print("python", sys.version.split()[0], "| torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available(),
      "| torchvision", torchvision.__version__, "| transformers", transformers.__version__, "| accelerate", accelerate.__version__,
      "| diffusers", diffusers.__version__, "| deepspeed", deepspeed.__version__, "| numpy", numpy.__version__,
      "| av", av.__version__, "| pyarrow", pyarrow.__version__, "| hb_aug_v2", hb_aug_v2.VERSION)
assert torch.__version__.startswith("2.7.0") and numpy.__version__ == "1.26.4", (torch.__version__, numpy.__version__)
assert transformers.__version__ == "4.57.6" and accelerate.__version__ == "1.12.0", (transformers.__version__, accelerate.__version__)
EOF
[ "$(cat "$HB_ENV/hb-diffusers-pinned")" = 1 ] || echo "WARN: diffusers is the 0.37.0 fallback, not the pinned commit (see README)"
echo "BUILD_ENV_OK $HB_ENV"
