# Common environment for hb_pi05. Source it:  source /path/to/hb_pi05/hb_env.sh
# Every variable can be overridden by exporting it before sourcing.
HB_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
export HB_DIR
export HB_G=${HB_G:-$(dirname "$HB_DIR")}                      # = $P/policies-20261109 when deployed as $G/hb_pi05
export HB_OPENPI=${HB_OPENPI:-$HB_G/HumanWAM/third_party/openpi}
export HB_OPENPI_COMMIT=215abfb217dbac7d5f1273282331b9b1866c0479
export HB_ENV=${HB_ENV:-$HB_G/envs/openpi}
export HB_PI05_DATA=${HB_PI05_DATA:-$HB_G/pi05-data}            # decoded 224px caches + norm stats per task
export HB_PI05_RUNS=${HB_PI05_RUNS:-$HB_G/pi05-runs}            # probes, chains, checkpoints, logs
export HB_PI05_BASE_JAX=${HB_PI05_BASE_JAX:-$HB_G/models/openpi/pi05_base}               # gs://openpi-assets/checkpoints/pi05_base
export HB_PI05_BASE_PT=${HB_PI05_BASE_PT:-$HB_G/models/openpi/pi05_base_pytorch_fp32}     # repo converter output (32-dim)
export HB_PI05_INIT=${HB_PI05_INIT:-$HB_G/models/openpi/pi05_base_g1a76_pytorch}         # 76-dim init used by training
# openpi.shared.download maps gs://big_vision/paligemma_tokenizer.model -> $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-$HB_G/models/openpi}
# The PyTorch trainer imports JAX (tree utils, jax.process_count); keep JAX off the GPUs.
export JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HB_G/hf-home}
# hb_aug_v2.py is expected next to this file; $HB_G/hb_groot is a fallback location.
export PYTHONPATH="$HB_DIR:$HB_G/hb_groot${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$HB_ENV/bin" ]; then export PATH="$HB_ENV/bin:$PATH"; fi
