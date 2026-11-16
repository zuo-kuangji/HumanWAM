#!/bin/bash
# Relay cmd item (2026-11-15 22:xx user): the GR00T A18 / A17 h32 variant on the CROPPED v2 data
#   ranks 0,1 -> A18 pour water cropped v2 (HF a18-pour-water-expert-cropped-v2 @44cdd57, 100 episodes, 66,850 frames)
#   ranks 2,3 -> A17 unscrew bottle cropped v2 (HF a17-unscrew-bottle-expert-cropped-v2 @3520b70, 100 episodes, 75,184 frames)
# Everything else unchanged: head + chest, 2 x 8 x 32 = 512, no accumulation, 4000 steps, full fine-tune from the GR00T
# N1.7 base, lr 1e-4 cosine warmup 5 %, wd 1e-5, aug v2 + A20 proprio; every dropout 0, action horizon 32, 16 flow
# sampling steps, layer-wise lr (language 1e-6, vision 1e-5, projector / rest 1e-4). Output groot-runs-h32/<task>.
# Each pair waits (up to 2 h) until its data view is verified; skips (exit 3) when the preparation failed.
G=$(dirname "$(dirname "$(realpath "$0")")")
L=$G/h2cr2-20261115
wait_ok() {
  for i in $(seq 720); do [ -f "$1" ] && return 0; [ -f "$2" ] && return 1; [ $((i % 60)) = 1 ] && echo "$(date -u) waiting for $1"; sleep 10; done
  return 1
}
case "${RANK:?DLC RANK missing}" in
  0|1) wait_ok "$L/groot-a18cr2.ok" "$L/groot-a18cr2.failed" || { echo "SKIP groot h32 a18cr2: data view not verified"; exit 3; } ;;
  2|3) wait_ok "$L/groot-a17cr2.ok" "$L/groot-a17cr2.failed" || { echo "SKIP groot h32 a17cr2: data view not verified"; exit 3; } ;;
esac
export HB_GROOT_MODEL_OVERRIDES='{"attn_dropout": 0.0, "diffusion_model_cfg.dropout": 0.0, "state_dropout_prob": 0.0, "num_inference_timesteps": 16}'
export HB_GROOT_LR_GROUPS='{"language": 1e-6, "vision": 1e-5, "projector": 1e-4, "rest": 1e-4}'
exec bash "$G/hb_groot/groot_pair2x32_h2v.sh" a18cr2 a17cr2 29730 29732
