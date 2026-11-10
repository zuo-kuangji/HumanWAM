#!/bin/bash
# Download the inputs of Cosmos-Predict2.5-2B (base/post-trained): public parts anonymously, the gated transformer
# checkpoint from the OFFICIAL repo with the user's HF_TOKEN (exit 4 = NEED_TOKEN), then build
# the diffusers pipeline directory DiT4DiT loads ($HB_COSMOS_DIR). Run once on ONE node after build_env.sh:
#   bash $G/hb_dit4dit/fetch_models.sh 2>&1 | tee $G/models/fetch_dit4dit.log        # -> MODELS_OK
# Skips the build when $HB_COSMOS_DIR already holds a pipeline (ours: HB_PROVENANCE.json; or an official copy of
# nvidia/Cosmos-Predict2.5-2B@diffusers/base/post-trained placed there by hand: model_index.json + transformer/ + vae/).
# ~21 GB download (transformer .pt 4.1 GB, Cosmos-Reason1-7B 16.6 GB, Wan VAE 0.5 GB, tokenizer) + ~21 GB output.
set -euo pipefail
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
PY=$HB_ENV/bin/python
HF=$HB_ENV/bin/hf
[ -x "$HF" ] || HF=$HB_ENV/bin/huggingface-cli
export HF_HUB_DOWNLOAD_TIMEOUT=600 HF_HUB_ETAG_TIMEOUT=60 HF_HUB_DISABLE_TELEMETRY=1
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE

# Transformer weights ONLY from the official gated repo (license acceptance + the user's HF token in env HF_TOKEN;
# never written to disk/logs). Third-party re-uploads of the gated model are deliberately NOT used.
OFFICIAL_REPO=nvidia/Cosmos-Predict2.5-2B
OFFICIAL_REV=${HB_COSMOS_REV:-main}
CKPT=base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt
CKPT_SIZE=4118575006                                    # size listed for the official file
REASON_REPO=nvidia/Cosmos-Reason1-7B
REASON_REV=375e24000b24baed78f4618d3dd779e47cd96323
QWEN_REPO=Qwen/Qwen2.5-VL-7B-Instruct
QWEN_REV=cc594898137f460bfe9f0759e9844b3ce807cfb5
WAN_REPO=Wan-AI/Wan2.1-T2V-1.3B-Diffusers
WAN_REV=0fad780a534b6463e45facd96134c9f345acfa5b

if [ -f "$HB_COSMOS_DIR/model_index.json" ] && [ -d "$HB_COSMOS_DIR/transformer" ] && [ -d "$HB_COSMOS_DIR/vae" ]; then
  echo "== $HB_COSMOS_DIR already present ($(ls "$HB_COSMOS_DIR" | tr '\n' ' '))"
  [ -f "$HB_COSMOS_DIR/HB_PROVENANCE.json" ] || echo "NOTE: no HB_PROVENANCE.json -> assuming an official diffusers copy"
  echo "MODELS_OK $HB_COSMOS_DIR"
  exit 0
fi
mkdir -p "$HB_COSMOS_SRC"

dl() {  # dl <repo> <rev> <dir> [files/--include ...]; retried
  local repo=$1 rev=$2 dir=$3 i
  shift 3
  for i in 1 2 3 4 5; do
    if "$HF" download "$repo" "$@" --revision "$rev" --local-dir "$dir"; then return 0; fi
    echo "retry $i: $repo"; sleep 20
  done
  return 1
}

echo "== 2/4 text encoder ($REASON_REPO@$REASON_REV)"
dl "$REASON_REPO" "$REASON_REV" "$HB_COSMOS_SRC/cosmos-reason1-7b" --exclude "*.md" ".gitattributes"

echo "== 3/4 tokenizer ($QWEN_REPO@$QWEN_REV)"
dl "$QWEN_REPO" "$QWEN_REV" "$HB_COSMOS_SRC/qwen2.5-vl-7b-tokenizer" \
  config.json generation_config.json tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.json \
  preprocessor_config.json

echo "== 4/4 VAE ($WAN_REPO@$WAN_REV)"
dl "$WAN_REPO" "$WAN_REV" "$HB_COSMOS_SRC/wan2.1-t2v-1.3b-diffusers" vae/config.json vae/diffusion_pytorch_model.safetensors
echo "PUBLIC_PARTS_OK"

echo "== 1/4 transformer checkpoint ($OFFICIAL_REPO@$OFFICIAL_REV, gated)"
F=$HB_COSMOS_SRC/predict2.5-2b/$CKPT
if [ ! -s "$F" ]; then
  [ -n "${HF_TOKEN:-}" ] || { echo "NEED_TOKEN: $OFFICIAL_REPO is gated; accept its license on HF and rerun with HF_TOKEN set"; exit 4; }
  dl "$OFFICIAL_REPO" "$OFFICIAL_REV" "$HB_COSMOS_SRC/predict2.5-2b" "$CKPT"
fi
[ "$(stat -c %s "$F")" = "$CKPT_SIZE" ] || { echo "FATAL: $F size $(stat -c %s "$F") != $CKPT_SIZE"; exit 3; }
CKPT_SHA=$(sha256sum "$F" | cut -d' ' -f1)
echo "   size OK, sha256 $CKPT_SHA"

PROV=$(cat <<EOF
{"transformer_ckpt": {"repo": "$OFFICIAL_REPO", "revision": "$OFFICIAL_REV", "file": "$CKPT", "size": $CKPT_SIZE, "sha256": "$CKPT_SHA"},
 "text_encoder": {"repo": "$REASON_REPO", "revision": "$REASON_REV"},
 "tokenizer": {"repo": "$QWEN_REPO", "revision": "$QWEN_REV"},
 "vae": {"repo": "$WAN_REPO", "revision": "$WAN_REV", "subfolder": "vae"},
 "recipe": "diffusers@$HB_DIFFUSERS_COMMIT scripts/convert_cosmos_to_diffusers.py --transformer_type Cosmos-2.5-Predict-Base-2B --save_pipeline"}
EOF
)
echo "== converting -> $HB_COSMOS_DIR"
mkdir -p "$(dirname "$HB_COSMOS_DIR")"
"$PY" "$H/hb_build_cosmos.py" --transformer-ckpt "$F" --text-encoder "$HB_COSMOS_SRC/cosmos-reason1-7b" \
  --tokenizer "$HB_COSMOS_SRC/qwen2.5-vl-7b-tokenizer" --wan-root "$HB_COSMOS_SRC/wan2.1-t2v-1.3b-diffusers" \
  --out "$HB_COSMOS_DIR" --provenance "$PROV"
du -sh "$HB_COSMOS_DIR" || true
echo "MODELS_OK $HB_COSMOS_DIR"
