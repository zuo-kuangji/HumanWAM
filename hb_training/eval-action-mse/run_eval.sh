#!/bin/bash
# One evaluation in the model's own environment (same env / code pins as its training).
#   run_eval.sh <model> <task> <split> <ckpt> <out.json> [eval_run.py extra args, e.g. --device cpu --max-queries 2]
set -euo pipefail
H=$(dirname "$(realpath "$0")")
export G=${G:-$(dirname "$H")}
P=$(dirname "$G")
MODEL=$1 TASK=$2 SPLIT=$3 CKPT=$4 OUT=$5
shift 5
export HF_HOME=$G/hf-home HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
case $MODEL in
  groot)
    ENV=$G/envs/groot-py312; GR=$G/HumanWAM/third_party/Isaac-GR00T
    export LD_LIBRARY_PATH="$ENV/lib/hbstub:${LD_LIBRARY_PATH:-}" PYTHONPATH="$GR:$G/hb_groot" GROOT_SRC="$GR"
    PY=$ENV/bin/python ;;
  pi05)
    # shellcheck disable=SC1091
    source "$G/hb_pi05/hb_env.sh"
    PY=$HB_ENV/bin/python ;;
  psi0|act)
    ENV=$G/envs/psi0; PSI=$G/HumanWAM/third_party/Psi0
    export PYTHONPATH="$PSI/src:$G/hb_psi0:$G/hb_act" PSI_SRC="$PSI" HB_AUG_DIR="$G/hb_groot" TORCH_HOME="$G/torch-home"
    PY=$ENV/bin/python ;;
  deltaflow)
    export PYTHONPATH="$G/deltaflow-runs/HumanWAM/src:$G/deltaflow-runs/HumanWAM"
    PY=$P/venv/bin/python ;;
  dit4dit)
    # shellcheck disable=SC1091
    source "$G/hb_dit4dit/hb_env.sh"
    PY=$G/envs/dit4dit/bin/python ;;
  fastwam)
    # 2026-11-16: FastWAM env; the DLC rank variables would put FastWAM / accelerate into distributed mode
    F=$G/fastwam
    unset RANK WORLD_SIZE LOCAL_RANK GROUP_RANK MASTER_ADDR MASTER_PORT
    export FASTWAM_SRC=$F/FastWAM DIFFSYNTH_MODEL_BASE_PATH=$F/models DIFFSYNTH_SKIP_DOWNLOAD=true PYTHONPATH="$F/FastWAM/src"
    export PATH=$G/envs/fastwam/bin:$PATH
    PY=$G/envs/fastwam/bin/python ;;
  *) echo "run_eval: unknown model $MODEL"; exit 64 ;;
esac
mkdir -p "$(dirname "$OUT")"
exec "$PY" "$H/eval_run.py" --model "$MODEL" --task "$TASK" --split "$SPLIT" --ckpt "$CKPT" --out "$OUT" "$@"
