#!/bin/bash
# pi05_base JAX params -> PyTorch (repo converter, float32, 32-dim) -> 76-dim G1 init checkpoint.
# One node, CPU only (JAX_PLATFORMS=cpu), needs ~40 GB RAM.  Usage: bash convert_base.sh
# Inputs : $HB_PI05_BASE_JAX/{params,assets}   (gs://openpi-assets/checkpoints/pi05_base)
#          $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model (only needed later for training/verify)
# Outputs: $HB_PI05_BASE_PT/model.safetensors (+config.json, assets/, hb_convert_report.json)
#          $HB_PI05_INIT/model.safetensors    (+hb_init_report.json listing every freshly initialised key)
set -euo pipefail
H=$(dirname "$(realpath "$0")")
source "$H/hb_env.sh"
# GCS is ~100 KB/s from the DLC nodes; default source is HF lerobot/pi05_base (same weights, PyTorch, names remapped).
LEROBOT_BASE=${HB_PI05_BASE_LEROBOT:-$HB_G/models/lerobot-pi05_base}
if [ "${HB_PI05_BASE_FORMAT:-lerobot}" = lerobot ]; then
  [ -f "$LEROBOT_BASE/model.safetensors" ] || { echo "missing $LEROBOT_BASE/model.safetensors"; exit 2; }
  cd "$HB_OPENPI"
  python "$H/hb_init_weights.py" init76 --lerobot --base "$LEROBOT_BASE" --out "$HB_PI05_INIT"
  ls -la "$HB_PI05_INIT"
  exit 0
fi
[ -d "$HB_PI05_BASE_JAX/params" ] || { echo "missing $HB_PI05_BASE_JAX/params"; exit 2; }
cd "$HB_OPENPI"
if [ ! -f "$HB_PI05_BASE_PT/model.safetensors" ]; then
  # Equivalent to the README command
  #   python examples/convert_jax_model_to_pytorch.py --checkpoint_dir $HB_PI05_BASE_JAX \
  #     --config_name pi05_droid_finetune --output_path $HB_PI05_BASE_PT --precision float32
  # plus a check that the converter's non-strict load_state_dict skipped nothing but lm_head.
  python "$H/hb_init_weights.py" convert --jax-dir "$HB_PI05_BASE_JAX" --out "$HB_PI05_BASE_PT"
fi
python "$H/hb_init_weights.py" init76 --base "$HB_PI05_BASE_PT" --out "$HB_PI05_INIT"
ls -la "$HB_PI05_BASE_PT" "$HB_PI05_INIT"
