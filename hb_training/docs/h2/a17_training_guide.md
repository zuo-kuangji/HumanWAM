# A17 training guide: H2 unscrew bottle cap on two cameras (GR00T N1.7, Psi0, DeltaFlow)

Written 2026-11-14 for the colleagues who train the benchmark models. Owner decisions are marked **(owner)**. The
evaluation results behind them are in the [a17 debug handoff](debug_unscrew_bottle_rollout.md) (sections 2 and 4).
The a14 round of the same program is the [a14 training guide](a14_training_guide.md).

## 1. Why a new round

Every a17 checkpoint so far, evaluated on the same random layouts with the same evaluator (section 2.2):

| Model | HF folder | Cameras | Recipe | Success | Where it fails |
|---|---|---|---|---|---|
| GR00T N1.7 | `groot/a17-unscrew-bottle/epoch-020` | head, chest | VLM frozen (projector, action head, VLLN trained), batch 696, 3,315 steps (20 epochs) | 0/50 | left fist on the bottle in 26/50; the right hand stops 7-10 cm from the cap and never touches it |
| GR00T N1.7 | `groot/a17-unscrew-bottle/b256/step-004000` | head, chest, wrist | full fine-tune, batch 256, 4,000 steps (8.9 epochs) | 0/50 | 49/50 never move |
| Psi0 | `psi0/a17-unscrew-bottle/epoch-020` | head, chest | full fine-tune, batch 512, 4,506 steps (20 epochs) | **5/50** | right fingers on the cap in 44/50, cap off the thread in 9/50 |
| Psi0 | `psi0/a17-unscrew-bottle/b512/step-004000` | head, chest, wrist | as `epoch-020` plus the wrist camera, 4,000 steps, no proprio delay | 0/5 | the open right hand hovers 6-7 cm above the cap |
| DeltaFlow | none | | | | no a17 checkpoint was trained |

**This round (owner):** GR00T and DeltaFlow train on one protocol: head + chest cameras, global batch 512, 4,000
steps, the a17 state78 / action75 contract unchanged. Each starts from a recipe that already has a result:

- GR00T: its current a17 recipe (`b256`, full fine-tune), with two cameras and batch 512;
- DeltaFlow: its a18 pour configuration (`a18_e40`: head + chest, 31/100 on a18), on the a17 data.

**Psi0 is not retrained (owner):** `epoch-020` already succeeds on a17 (5/50) and stays its a17 checkpoint. Its
two-camera configuration is recorded in section 4.2 as the reference.

## 2. Dataset: `humanoid-benchmark/a17-unscrew-bottle-expert-v1`

Private. The same data and commit as every earlier a17 checkpoint; nothing changes.

| | |
|---|---|
| Data commit | `407e6e2` (`407e6e20db2debc9b77fc2813e3445f549d51a33`, 2026-11-09; still the latest on 2026-11-14) |
| Episodes / frames | 100 / 115,333 at 30 fps (median episode 38.3 s, range 37.9-43.9 s) |
| Cameras in the files | head, chest, wrist (right wrist), third_person; 480 x 640 H.264 |
| Task text (`task`) | "Facing the worktable, grasp the glaze bottle with the left hand in a fist and lift it, then unscrew the cap with the right hand's five fingers and lift it off." |
| Phases (`annotation.phase`) | idle, grasp_bottle, grip_cap, unscrew, lift_cap, hold |

### 2.1 Training inputs and targets (all three models)

| Role | Feature | Dims |
|---|---|---|
| Images **(owner: head and chest only)** | `observation.images.head`, `observation.images.chest` | 2 x 480 x 640 x 3 |
| State78 | `observation.state` (75: legs 0-11, waist 12-14, left arm 15-21, right arm 22-28, head 29-30, hands 31-74) + `observation.projected_gravity` (3) | 78 |
| Action75 | `action.body_target` (31) + `action.hands` (44), in this order | 75 |
| Prompt | the task text above | |

This is the contract of every earlier a17 and a18 checkpoint. Not inputs: `observation.images.wrist` (owner: two
cameras; the only non-zero a17 result comes from a two-camera checkpoint), `third_person` (not a robot camera) and the
privileged columns (`eef_state`, `object_state`, `task_state`, `tactile`, `tactile_force`, `base_velocity`). Not
targets: `action.arms`, `action.locomotion`.

Action dimension 39 (a hand target) is constant over the 100 episodes (q01 = q99); the evaluator executes and feeds
back its recorded value. The normalization must tolerate a zero range there; the a18 data had 22 such dimensions.

### 2.2 What the evaluator does with Action75

So that the targets you train on mean what the evaluator executes (`debug/newton/h2_sharpa_policy_eval.py --task
bottle`; full protocol in [the handoff, section 4](debug_unscrew_bottle_rollout.md#4-evaluation-protocol)):

- Rows are executed at 30 Hz, linearly interpolated; the policy is queried every **16 rows** (0.53 s). Predict at
  least 16 rows per query.
- Each query gets live 480 x 640 renders of head and chest, state78 and the 16 executed rows (the history, which
  only DeltaFlow reads).
- `body_target` 0-30: **all executed**, on PD with the expert's LowCmd gains. There is no SONIC; the elastic torso
  support stays on for the whole episode, as in the collection. Unlike a14, the leg and waist columns are executed.
- Arms: from 9.22 s (left) and 15.10 s (right), the expert's gravity feed-forward and the row velocity as dq (the
  median times at which the expert's skill server took each arm over).
- `hands` 44: Sharpa position targets.
- Episode: 60 s budget (1.5 x the 38.4 s mean demonstration), random layouts from the collection profile
  `h2_unscrew_bottle_cap_r1` with held-out layout seeds 1000 + episode. Success, held for 3 s: the cap off the
  thread and at least 2 cm above its seat with at least 2 right-hand digits on it, the bottle in the left fist and
  off the table (`cap_off_lifted_v4`), the robot standing, the table moved less than 5 mm.

The evaluator reproduces the data: the recorded rows of training episode 3, replayed through it, succeed at 36.3 s
with the joints within 0.5 deg of the recording (executor parity, 2026-11-14).

## 3. Common protocol for the new runs (owner)

For GR00T and DeltaFlow.

| Item | Value |
|---|---|
| Global batch | **512** |
| Steps | **4000** (17.8 epochs: 226 steps per epoch over 115,333 frames) |
| Checkpoints to publish | step 2000 and step 4000 |
| Cameras | **head + chest** |
| State / action | state78 / action75 (section 2.1) |
| Split | all 100 episodes for training |
| Initialization | the base weights of each model's reference recipe (section 4), not an earlier a17 checkpoint |
| HF path | `humanoid-benchmark/<model>/a17-unscrew-bottle/b512-2cam/step-002000` and `step-004000` |

`b512-2cam` keeps these runs apart from the existing 3-camera folders (such as `groot/.../b256/`).

Each published checkpoint should carry, besides the weights: `protocol.json` (dataset repo and commit, cameras,
state78 / action75 layout, global batch, steps, trainable modules, initialization), the normalization statistics it
was trained with, the training config and launch arguments, the training log and loss curve, and the inference code
the evaluation needs. Keep the model card and `protocol.json` consistent with what was trained: the 3-camera
re-trains published a README camera column that said head, chest, and the GR00T `b256` `protocol.json` field `model`
says "VLM frozen" although that run trained all modules.

## 4. Per-model recipes

### 4.1 GR00T N1.7: the current a17 recipe with two cameras and batch 512 (owner)

Start from `groot/a17-unscrew-bottle/b256/step-004000` (`protocol.json`, `experiment_cfg/conf.yaml`,
`code/hb_groot/`) and change only:

| Item | `b256/step-004000` | this round |
|---|---|---|
| Cameras (modality `video`) | head, chest, wrist | **head, chest** |
| Global batch | 256 (8 GPUs x 32) | **512** (e.g. 16 GPUs x 32) |
| Checkpoints | saved every 1000 steps, step 4000 published | publish step 2000 and step 4000 |

The cameras have to change in two places: the `video` modality keys in `conf.yaml`, and the data preparation. The
published `code/hb_groot/prepare_groot_data.py` sets `CAMS["a17"] = ("head", "chest", "wrist")` (since 2026-11-12
every H2 task trained with three cameras); set it to `("head", "chest")` for this run. The two-camera modality config
is `code/hb_groot/h2_body31_hand44_h16.py`; `h2w3_body31_hand44_h16.py` is the three-camera one.

Unchanged from `b256`:

```yaml
model:
  init: GR00T N1.7 base (groot-n17-base), VLM nvidia/Cosmos-Reason2-2B, embodiment tag new_embodiment
  trainable: all modules (tune_llm, tune_visual, tune_projector, tune_diffusion_model, tune_vlln)
  action: body_target (31), left_hand_target (22), right_hand_target (22); delta indices 0-15 (16 rows; model
          action horizon 40); ABSOLUTE, NON_EEF
  state: left_leg, right_leg, waist, left_arm, right_arm, head, left_hand, right_hand, projected_gravity (78)
  normalization: percentiles (use_percentiles true)
  state_dropout_prob: 0.2
augmentation:
  visual: augmentation spec v2 (15 visual effects, one per sample at 1.5x); model-native image augmentation off
  proprio: the GR00T A20 recipe: per-sample joint bias +-2 deg, noise 0.25 deg; no delay, no history drop
train:
  steps: 4000, lr 1e-4, cosine, warmup_ratio 0.05, weight_decay 1e-5, adamw_torch, max_grad_norm 1.0
  precision: bf16 + tf32, DeepSpeed stage 2
```

The `b256` run ended at a loss of about 0.017 (mean of the last 50 logged steps). The earlier two-camera GR00T
(`epoch-020`, 0/50) froze the VLM, so this round is not a repeat of it.

### 4.2 Psi0: no new run; the `epoch-020` configuration (owner)

Psi0 already succeeds on a17 with `psi0/a17-unscrew-bottle/epoch-020` (5/50 on random layouts; demos in
`debug/outputs/h2_policy_rollouts/a17_unscrew_bottle_cap/Psi0/succeed/`). It stays the a17 Psi0 checkpoint and is not
retrained. Its configuration, from `argv.txt` and `run_config.json` in that folder, is the two-camera reference:

```yaml
data:
  dataset: a17-unscrew-bottle-expert-v1 @ 407e6e2 (100 episodes, 115,333 frames, all for training)
  cameras: [observation.images.head, observation.images.chest]
  image_size: 270x480 (resize + center_crop)
  state: observation.hb78 (78), normalized
  action: action.hb75 (75), action_norm_type bounds, pad_action_dim 75
  action_chunk: 30, exec_horizon: 30, observation_horizon: 1   # the evaluator executes the first 16 rows per query
augmentation:
  hb_aug: true             # augmentation spec v2: 15 visual effects, one per sample at 1.5x
  hb_joint_perturb: true
  hb_delay_prob: 0.15      # proprioception delay, at most 2 frames
  img_aug / view_aug: false
  state_noise_std: 0.0
model:
  init: psi0 postpre.sonic1.1.unifolm.2609181726.40k
  noise_scheduler: flow (train steps 1000)
  num_blocks: 12, vlm_layer_indices: [3,5,8,10,12,14,17,19,21,23,26,28]
  tune_vlm: true, qk_norm: rms_norm, combined_temb, state_as_action_token, state_null_token, no_film, no_rtc
  pooled_text_encoder: clip-vit-large-patch14 (projection 768, frozen, cached)
train:
  gpus: 16 (2 nodes x 8), per_gpu_batch 32, global_batch 512
  steps: 4506 (20 epochs), warmup 113, checkpoints every 2253 steps (epoch-010 and epoch-020)
  lr: 1e-4 (action head / projector), vision tower 1e-5, language backbone 1e-6
  scheduler: cosine, AdamW betas (0.95, 0.999), weight_decay 1e-6, grad clip 1.0
  precision: bf16, DDP, gradient checkpointing, seed 292285
```

This is a full fine-tune: 2.80 of the 2.81 billion parameters train (the rest are a fixed positional encoding and an
unused final norm), with the layer-wise learning rates above; the CLIP text encoder is frozen and only its cached
features are used. `epoch-020` ended at a loss of about 0.49 (mean of the last 50 logged steps).

The 3-camera re-train `b512/step-004000` (0/5) differs from it only in: the wrist camera, 32 GPUs x 16, 4000 steps
(warm-up 100, checkpoints every 2000) and `hb-delay-prob=0.0`; it ended at 0.54. Anyone who re-runs the `epoch-020`
configuration: the published `code/hb_psi0/` now builds that 3-camera re-train by default, so start from
`epoch-020/argv.txt`, and use a new output directory, because `--train.resume_from_checkpoint=latest` would resume
an old run found there.

### 4.3 DeltaFlow: the a18 pour configuration on the a17 data (owner)

Use the configuration of `deltaflow/legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/epoch-040`
(`training_config.yaml`, `protocol.json`) and change only:

| Item | a18 `epoch-040` | a17 |
|---|---|---|
| Dataset | a18-pour-water-expert-v1 @ 8e956a10 (100 episodes, 72,752 frames) | a17-unscrew-bottle-expert-v1 @ 407e6e2 (100 episodes, 115,333 frames) |
| Dataset reader | `imagewam.datasets.h2_pour.H2PourDataset` on the a18 export | the same reader on the a17 export (same feature names and layout) |
| Normalization | `normalization.json` fit on a18 | refit on a17 (one constant action dimension, 39) |
| Text cache | the a18 instruction (Qwen3-4B, 128 tokens, dim 7680) | recompute for the a17 task text |
| Augmentation `policy_id` | `a18-h2-rgb-v1` | an a17 id; the effects unchanged |
| Global batch | 256 (32 GPUs x 8) | **512** (32 GPUs x 16, or 64 x 8) |
| Steps | 11,400 (40 of a planned 60 epochs), warm-up 855 | **4000**, warm-up 200, cosine to 0 at step 4000 |
| Checkpoints | every 2,850 steps | steps 2000 and 4000 |

Unchanged: FLUX.2-klein-base-4B video expert trained with full parameters (no LoRA); the action DiT (5 double-stream
+ 20 single-stream blocks, width 1024, 24 heads) initialized from `action-shared-flux2-4b.pt`; mixture-of-transformers
in `noise_idm` mode with the future tokens; knowledge insulation (no action gradient into the video expert); FLUX.2
AE and Qwen3-4B frozen (cached text features); per sample the head and chest frames at t (224 x 224 each, side by
side 224 x 448), state78, the 16 previous action rows (t-16..t-1), target the next 16 rows (t..t+15) and the two
frames at t+16; losses action flow matching 1.0 (from the previous chunk plus noise std 0.5), action one-step 0.5,
future frame 0.5, frame one-step 1.0, scheduler shift 5.0 with 1000 timesteps; AdamW lr 1e-4, weight decay 0.01, grad
clip 1.0; DeepSpeed ZeRO-1, bf16; visual augmentation `b6-visual-aug-v2` (consistent across the two views and the
future frame), no proprio augmentation.

The training code is published (`deltaflow/code/hb_deltaflow/`: `prepare_deltaflow_h2.py`, `train_deltaflow.py`,
`h2_hbaug.py`, `qwen_cache.py`). The a18 evaluation had to reconstruct the model code, AE and text cache locally
because only the weights were published at first; please publish the full inference package (model code,
normalization, text cache, a minimal inference entry) with the a17 checkpoint.

## 5. Checks before publishing

- Loss curves without spikes; report the final loss (mean of the last 50 logged steps). Losses are not comparable
  across models (DeltaFlow's total includes the video terms).
- `protocol.json` matches the run: cameras head + chest, dataset commit `407e6e2`, steps, global batch, trainable
  modules, initialization.
- An offline check on a few training episodes (open loop on the recorded inputs): the first 16 predicted rows beat the
  "hold the last action" baseline on the right arm (22-28) and the right-hand columns, including the `grip_cap` and
  `unscrew` phases. The evaluation side runs the same check (`debug/newton/h2_policy_offline_check.py --task bottle`)
  before any closed-loop run.

## 6. Known risks

- **The last centimetres at the cap.** The a17 failures are near-misses: GR00T `epoch-020`'s right fingertips stop
  7-10 cm from the cap origin (the expert grips with its distal links 5.2-5.4 cm from it), Psi0 `b512` hovers 6-7 cm
  above the cap, and even Psi0 `epoch-020` got the cap off the thread in only 9 of the 44 episodes where its fingers
  reached it (4 of those 9 dropped the cap). A low loss over the whole action does not guarantee contact; the right
  arm and hand decide the task.
- **GR00T `b256` never started** (49/50 episodes almost static: every chunk nearly equal to the first), and the cause
  is not known. This round changes the cameras and the batch at once, so its result does not isolate either.
- **DeltaFlow sees fewer samples than its a18 run**: 4000 x 512 = 2.05 M against 11,400 x 256 = 2.92 M (17.8 epochs
  against 40), on episodes 1.6 times longer (38 s against 24 s). Please keep the resume state of the last checkpoint
  so the run can be extended if the evaluation asks for it.
