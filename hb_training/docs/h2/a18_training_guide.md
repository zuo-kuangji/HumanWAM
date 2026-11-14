# A18 training guide: H2 pour water (DeltaFlow, GR00T N1.7, Psi0)

Written 2026-11-14 for the colleagues who train the benchmark models. Owner decisions are marked **(owner)**. The
analysis behind them is in the [pour-water debug handoff](debug_pour_water_rollout.md) (sections 3 and 5). The
structure follows the [a14 training guide](a14_training_guide.md); unlike a14, the a18 action contract does not
change: the robot does not walk, so there is no planner output.

## 1. What is needed, and why

| Model | Checkpoint now | Closed-loop result (success without the 2 cm put-back) | Needed **(owner)** |
|---|---|---|---|
| DeltaFlow | `deltaflow/legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/epoch-040` (head + chest) | **31/100** (fixed 15/50, random 16/50) | **Nothing: done.** Its configuration is section 4.3 |
| GR00T N1.7 | `groot/a18-pour-water/epoch-020` (head + chest, batch 696, 2091 steps, VLM frozen); `b256/step-004000` (head + chest + wrist, batch 256, full fine-tune) | 0/100; 0/6 | **Train**: the b256 recipe with head + chest, batch 512, 4000 steps (section 4.1) |
| Psi0 | `psi0/a18-pour-water/epoch-020` (head + chest, batch 1280, **only 1137 steps**); `b512/step-004000` (head + chest + wrist, batch 512, 4000 steps) | 0/100; 0/5 fixed, **3/10 random** | **Continue training to 4000 steps** with head + chest, on the 2-camera configuration that succeeded on a17 (a17_e20, 5/50 bottle cap; section 4.2) |

Why these choices:

- **Head + chest for every model (owner).** The non-zero results so far come from 2-camera DeltaFlow a18 (31/100)
  and Psi0 a17_e20 (5/50), and from Psi0 a18 b512 (3 cameras, 3/10). The 3-camera GR00T a18 never closes its hand.
  The camera effect is not isolated: 2- and 3-camera checkpoints also differ in recipe. Training all three
  models on the same two views makes them comparable.
- **Psi0's 2-camera a18 checkpoint is undertrained, not wrong.** `epoch-020` used global batch 1280, so its 20 epochs
  were only 1137 optimizer steps; its train loss stopped at 1.16. That is the value a17_e20 (also head + chest)
  had at the same step (1.12 at step 1137). a17_e20 went on to 4506 steps, ended at 0.47 and scored 5/50 on the
  bottle cap. Its imprecise grasp (palm 34 mm from the expert grasp point when closing) drags the pitcher off the
  table in 68/100 episodes.
- **GR00T needs a run that is comparable with the other two.** Both existing a18 GR00T checkpoints end their reach
  17-18 mm above the expert grasp point (section 6). The 2-camera b512 run removes the wrist camera and the batch
  difference as explanations.

The evaluator is not the bottleneck: the expert's own recorded actions, executed by the evaluator, succeed in 22/22
episodes, and the live renders do not change GR00T's or Psi0's predictions along the expert trajectory (handoff
section 5).

## 2. Dataset: `humanoid-benchmark/a18-pour-water-expert-v1`

Private. Unchanged since it was uploaded; every model so far trained on this commit.

| | |
|---|---|
| Data commit | `8e956a10` (`8e956a102c84f991d2745a8cc66155c9991b742d`) |
| Episodes / frames | 100 / 72,752 at 30 fps (episodes 23.9-24.6 s, median 24.3 s) |
| Cameras in the files | head, chest, wrist (right wrist), third_person; 480 x 640 H.264 |
| Task text (`task`) | "Grasp the pitcher's handle with the right hand, lift it, pour water into the glass tumbler, then put the pitcher back where it stood and let go." |
| Split | all 100 episodes for training |

### 2.1 Training inputs and targets (all three models)

Same features as every earlier a18 checkpoint; only the camera list is fixed by the owner.

| Role | Feature | Dims |
|---|---|---|
| Images **(owner: head and chest only)** | `observation.images.head`, `observation.images.chest` | 2 x 480 x 640 x 3 |
| State78 | `observation.state` (75: legs 0-11, waist 12-14, left arm 15-21, right arm 22-28, head 29-30, hands 31-74) + `observation.projected_gravity` (3) | 78 |
| **Action75** | `action.body_target` (31) + `action.hands` (44: left 22, right 22), in this order | 75 |
| Prompt | the task text above | |

Do not use `observation.images.wrist` (owner) or `third_person` (not a robot camera). The left hand does not move
in a18: its 22 action dims are constant over the whole dataset. Keep them in the action; the evaluator pins them
to the recorded value.

### 2.2 What the evaluator does with Action75

So that the targets you train on mean what the evaluator will execute (`debug/newton/h2_sharpa_policy_eval.py`):

- Rows are executed at 30 Hz; the policy is queried every **16 rows** (0.53 s). Predict at least 16 rows per query.
- The 31 body targets are tracked on PD with the expert's gains, the elastic torso support on (as in every a18
  demonstration). From 4.23 s, when the expert's skill server took the arm over, the right arm also gets the
  expert's gravity and payload feed-forward. The 44 hand targets are executed (Sharpa position targets).
- Episode: budget 36 s (1.5 x the demonstrations); fixed layout (the registered one) and random layouts (profile
  `h2_pour_water_r1`, held-out seeds 1000 + episode). Success **(owner 2026-11-11)**: cup >= 150 mL, spill <= 10 mL,
  pitcher upright and released, robot standing, cup and table moved < 5 mm. Putting the pitcher back within
  2 cm is not required.
- Precision this asks for: the expert grasps with the palm at a fixed point of the pitcher frame (std < 1 mm) and
  pours from 15.9 cm with the stream landing 2.3 cm from the cup centre. A pour position about 2 cm off already
  spills 50 mL or more (Psi0 b512, fixed layout).

## 3. Common protocol (owner)

| Item | Value |
|---|---|
| Global batch | **512** |
| Steps | **4000** (about 28 epochs: 142 steps per epoch over 72,752 frames) |
| Checkpoints to publish | step 2000 and step 4000 |
| Cameras | **head + chest** |
| State / action | state78 / **action75** (section 2.1), as every earlier a18 checkpoint |
| Split | all 100 episodes for training |
| HF path (suggested) | `humanoid-benchmark/<model>/a18-pour-water/b512-2cam/step-002000` and `step-004000` |

Each published checkpoint should carry, besides the weights: `protocol.json` (dataset repo and commit, cameras,
state78 / action75 layout, global batch, steps, recipe), the normalization statistics it was trained with, the
training config, the training log and loss curve, and the inference code the evaluation needs. Keep the model card
and `protocol.json` consistent with the training config: the 3-camera re-trains published a README camera column
that disagreed with their configs, and the GR00T `protocol.json` field `model` says "VLM frozen" while its recipe
trains all modules.

## 4. Per-model recipes

### 4.1 GR00T N1.7: the b256 recipe with two cameras and batch 512 (owner) — to train

Start from `groot/a18-pour-water/b256/step-004000` (`protocol.json`, `experiment_cfg/conf.yaml`, `code/hb_groot/`)
and change only:

| Item | a18 b256 | a18 b512-2cam |
|---|---|---|
| Cameras (modality `video`) | head, chest, wrist (`h2w3_body31_hand44_h16.py`) | **head, chest**: the modality config `h2_body31_hand44_h16.py` from the same `code/hb_groot/` (identical except the video keys) |
| Global batch | 256 (8 GPUs x 32) | **512** (e.g. 16 GPUs x 32) |
| Steps | 4000 (14 epochs) | 4000 (28 epochs) |
| Checkpoints | steps 2000, 4000 | steps 2000, 4000 |

Unchanged: dataset `a18-pour-water-expert-v1` @ `8e956a10`; action modalities `body_target` (31),
`left_hand_target` (22), `right_hand_target` (22), all `ABSOLUTE`, action horizon 16; state modalities as in
`modality.json` (legs, waist, arms, head, hands, projected gravity); Cosmos-Reason2-2B VLM, embodiment tag
`new_embodiment`; **all modules trained** (`tune_llm`, `tune_visual`, projector, diffusion action head, VLLN); lr
1e-4 cosine with warmup ratio 0.05, weight decay 1e-5; augmentation spec v2 (15 visual effects, one per sample at
1.5x) with the proprio augmentation of the GR00T A20 recipe (per-sample joint bias +-2 deg, noise 0.25 deg; no delay,
no history drop); model-native image augmentation off; the rest of `conf.yaml` (image 256 x 256 with crop 230,
`state_dropout_prob` 0.2, 4 inference timesteps).

### 4.2 Psi0: the a17_e20 configuration with two cameras, to 4000 steps (owner) — continue training

The existing 2-camera a18 checkpoint (`psi0/a18-pour-water/epoch-020`) stopped at 1137 steps (section 1). Train the
a18 head + chest model on the configuration of `psi0/a17-unscrew-bottle/epoch-020` (the 5/50 bottle-cap checkpoint),
the same one the [a14 guide](a14_training_guide.md#42-psi0-the-a17_e20-configuration-the-550-bottle-cap-checkpoint)
uses, and change only what a18 requires:

| Item | a17_e20 | a18 b512-2cam |
|---|---|---|
| Dataset | a17-unscrew-bottle-expert-v1 @ 407e6e2 (100 episodes, 115,333 frames) | **a18-pour-water-expert-v1 @ 8e956a10** (100 episodes, 72,752 frames) |
| Prompt | the a17 text | the a18 task text (section 2) |
| Cameras | head, chest | head, chest |
| State / action | `observation.hb78` (78) / `action.hb75`, `pad_action_dim` 75 | the same |
| Normalization | bounds, fit on a17 | bounds, **fit on a18** (as the a18 `epoch-020` / `b512` statistics) |
| Steps | 4506 (20 epochs), warmup 113, checkpoint every 2253 | **4000** (28 epochs), warmup 100, checkpoint every 2000 |
| Global batch | 512 (16 GPUs x 32) | **512** |

Start from the Psi0 initialization of a17_e20 and a18 b512 (`postpre.sonic1.1.unifolm.2609181726.40k`), so that the
run is comparable with both. Warm-starting from the a18 `epoch-020` weights saves about 1100 steps, but that run
used batch 1280 and its cosine schedule already ended at step 1137. If you do warm-start, record it in
`protocol.json`.

Unchanged from a17_e20:

```yaml
data:
  image_size: 270x480 (resize + center_crop)
  action_norm_type: bounds
  action_chunk: 30, exec_horizon: 30, observation_horizon: 1   # the evaluator executes the first 16 rows per query
augmentation:
  hb_aug: true             # augmentation spec v2: 15 visual effects, one per sample at 1.5x
  hb_joint_perturb: true
  hb_delay_prob: 0.15      # proprioception delay, at most 2 frames (the a18 b512 run used 0.0)
  img_aug / view_aug: false
  state_noise_std: 0.0
model:
  init: psi0 postpre.sonic1.1.unifolm.2609181726.40k
  noise_scheduler: flow (train steps 1000, eval steps 10)
  num_blocks: 12, vlm_layer_indices: [3,5,8,10,12,14,17,19,21,23,26,28]
  tune_vlm: true, qk_norm: rms_norm, combined_temb, state_as_action_token, state_null_token, no_film, no_rtc
  pooled_text_encoder: clip-vit-large-patch14 (projection 768)
train:
  lr: 1e-4 (action head / projector), vision tower 1e-5, language backbone 1e-6
  scheduler: cosine, AdamW betas (0.95, 0.999), weight_decay 1e-6, grad clip 1.0
  precision: bf16, DDP, gradient checkpointing, seed 292285
```

Reference losses (`hb_metrics.jsonl`, `train/loss`): a18 `epoch-020` ended at 1.16 (step 1137); a17_e20 ended at
0.47 (step 4506); a18 b512 (3 cameras) ended at 0.69 (step 4000). Expect the new run to reach 0.5-0.7 by step 4000.

### 4.3 DeltaFlow: the a18 epoch-040 configuration (owner) — done, no training

The checkpoint `deltaflow/legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/epoch-040` (31/100) stays the
DeltaFlow a18 model. Its configuration (`training_config.yaml`, `protocol.json`, `original_protocol.json` in that HF
folder) is the one the a14 guide starts from. Full description, in Chinese:
`debug/outputs/h2_policy_rollouts/a18_pour_water/deltaflow_a18_e40_training_config.md`.

| Item | Value |
|---|---|
| Dataset | a18-pour-water-expert-v1 @ `8e956a10`, all 100 episodes |
| Cameras / image | head + chest, each 224 x 224, side by side 224 x 448 |
| State / action | 78 / 75 (`h2_sharpa_body31_hand44`), normalization `normalization.json` |
| Sample | the frames at t, state78, the 16 previous action rows (t-16..t-1); targets the next 16 rows (t..t+15) and the two frames at t+16 |
| Trained | FLUX.2-klein-base-4B video expert (full parameters, no LoRA) + action DiT (5 double-stream + 20 single-stream blocks, width 1024, 24 heads; initialized from `action-shared-flux2-4b.pt`), a mixture of transformers in `noise_idm` mode with knowledge insulation (no action gradient into the video expert) |
| Frozen | FLUX.2 AE; Qwen3-4B text encoder (cached features, 128 tokens) |
| Losses | action flow matching 1.0 (from the previous chunk plus noise std 0.5), action one-step 0.5, future frame 0.5, frame one-step 1.0; scheduler shift 5.0, 1000 timesteps |
| Global batch | 256 (32 GPUs x 8) |
| Steps | 11,400 = 40 epochs of a planned 60 (cosine over 17,100 steps, warm-up 855; lr 2.8e-5 at the stop) |
| Optimizer | AdamW lr 1e-4, weight decay 0.01, grad clip 1.0; DeepSpeed ZeRO-1, bf16 |
| Augmentation | `b6-visual-aug-v2` (colour consistent across the two views and the future frame), no proprio augmentation |
| Final loss | 0.072 total (includes the frame terms; not comparable with GR00T or Psi0) |
| Inference | 1 sampling step from the executed history plus noise 0.5; 16 rows per query |

It does not follow the common protocol (batch 256, 11,400 steps, older augmentation); the owner keeps it because it
already succeeds. The evaluation reconstructed the normalization, AE and text cache locally (only the weights were
published at first). The official `normalization.json` and text cache are now on HF and have not yet been compared
with that reconstruction.

## 5. Checks before publishing

- Loss curves without spikes; report the final loss (mean of the last 50 logged steps).
- An offline check on a few training episodes (open loop on recorded inputs): the right arm beats the "hold the
  current pose" baseline, and the right hand starts closing near the expert's time (about 7.7 s). The evaluation side
  runs the same check (`debug/newton/h2_policy_offline_check.py`) before any closed-loop run, and then the shadow
  check along expert replays (`debug/newton/h2_policy_expert_replay.py --shadow`).
- The camera list, action layout and dataset commit in `protocol.json` match the training config.

## 6. Known risks

- **GR00T grasp height.** Both a18 GR00T checkpoints end their reach 17-18 mm above the expert grasp point,
  23-25 mm from it. The 2-camera `epoch-020` then closes loosely and pours beside the cup; the 3-camera `b256` never
  closes (it closes only within about 5 mm). On the same inputs along the expert trajectory, GR00T's right-arm
  predictions are 2-2.5x less precise than Psi0's (0.024-0.026 vs 0.009-0.012 rad), and the live renders are not
  the cause. If the b512-2cam run still ends high, the cause is in the model or inference (4 flow steps, all 16
  predicted rows executed with no overlap, 256 x 256 squashed images, state dropout 0.2), not in the camera set or
  the batch.
- **Pour precision.** Spilling more than 10 mL fails an episode, and a pour position about 2 cm off already spills
  50 mL. Psi0 b512 grasps 12 mm from the expert point with no height error, lifts in 14/15 episodes, and still
  succeeds only 3/10 on random layouts and 0/5 on the fixed one.
- **Irreversible failures.** A tipped or dragged pitcher spills its water and cannot be retried within the 36 s
  budget, unlike the bottle cap's 60 s budget with retries.
