# Common environment for hb_dit4dit. Source it:  source /path/to/hb_dit4dit/hb_env.sh
# Every variable can be overridden by exporting it before sourcing.
HB_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
export HB_DIR
export HB_G=${HB_G:-$(dirname "$HB_DIR")}                                  # = $P/policies-20261109 when deployed as $G/hb_dit4dit
# DiT4DiT source: HumanWAM c76527b pins third_party/DiT4DiT -> Mondo-Robotics/DiT4DiT @ 66a6f3a (docs/policies/DIT4DIT.md)
export HB_D4D_SRC=${HB_D4D_SRC:-$HB_G/HumanWAM/third_party/DiT4DiT}
export HB_D4D_COMMIT=66a6f3a12e2c8740157c1e478795952c040d31dd
export HB_DIFFUSERS_COMMIT=3996788b602eaae4da41a1d45726b62e662b73cf      # DiT4DiT requirements.txt pin (Cosmos2_5_PredictBasePipeline)
export HB_ENV=${HB_ENV:-$HB_G/envs/dit4dit}
export HB_D4D_DATA=${HB_D4D_DATA:-$HB_G/dit4dit-data}                   # decoded 224px caches + stats per task
export HB_D4D_RUNS=${HB_D4D_RUNS:-$HB_G/dit4dit-runs}                   # chains, probes, checkpoints, logs
# Cosmos-Predict2.5-2B (diffusers layout, = the gated nvidia/Cosmos-Predict2.5-2B@diffusers/base/post-trained) assembled
# from public sources by fetch_models.sh (or an official copy placed here by hand).
export HB_COSMOS_DIR=${HB_COSMOS_DIR:-$HB_G/models/cosmos-predict2.5-2b-diffusers-post-trained}
export HB_COSMOS_SRC=${HB_COSMOS_SRC:-$HB_G/models/cosmos-predict2.5-src}   # raw downloads used to build HB_COSMOS_DIR
export HF_HOME=${HF_HOME:-$HB_G/hf-home}
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TQDM_MININTERVAL=${TQDM_MININTERVAL:-30}
# hb_aug_v2.py lives in $HB_G/hb_groot (a copy next to these scripts takes precedence).
export PYTHONPATH="$HB_DIR:$HB_D4D_SRC:$HB_G/hb_groot${PYTHONPATH:+:$PYTHONPATH}"
if [ -d "$HB_ENV/bin" ]; then export PATH="$HB_ENV/bin:$PATH"; fi
