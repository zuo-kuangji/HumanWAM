#!/bin/bash
# Relay cmd item (2026-11-15 user): GR00T A18 (nodes 0,1) + A17 (nodes 2,3) re-train variant, each head + chest, 2 x 8 x 32
# = 512, no accumulation, 4000 steps, full fine-tune from the GR00T N1.7 base, lr 1e-4 cosine warmup 5 %, wd 1e-5, aug v2
# + A20 proprio; changes: every dropout 0, action horizon 32, 16 flow sampling steps, layer-wise lr (Psi-Zero style).
export HB_GROOT_MODEL_OVERRIDES='{"attn_dropout": 0.0, "diffusion_model_cfg.dropout": 0.0, "state_dropout_prob": 0.0, "num_inference_timesteps": 16}'
export HB_GROOT_LR_GROUPS='{"language": 1e-6, "vision": 1e-5, "projector": 1e-4, "rest": 1e-4}'
exec bash "$(dirname "$(realpath "$0")")/groot_pair2x32_h2v.sh" a18c2 a17c2 29730 29732
