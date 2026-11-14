# A14 training guide: H2 screw-bulb handover with planner commands (DeltaFlow, GR00T N1.7, Psi0)

Written 2026-11-14 for the colleagues who train the benchmark models. Owner decisions are marked **(owner)**. The
analysis behind them is in the [a14 debug handoff](debug_screw_bulb_rollout.md) (sections 5, 5.1, 5.2, 6). Which
cells of the 3 tasks x 3 models still need training: [H2 training matrix](h2_training_matrix.md); its a17 and a18
cells reuse this guide's section 4 recipes.

## 1. What changes from the a14 v1 checkpoints, and why

The v1 checkpoints (`groot` and `psi0` `a14-unscrew-bulb/step-004000`) predict joint targets only: body 31 + hands
44. In a14 the robot has to walk, and the leg and waist targets in the data are the outputs of SONIC's balance
controller while its planner walked the robot. They are not commands that make a robot walk:

- executed open loop on PD, even the expert's own recorded targets fall within about 1 s of walking;
- fed to SONIC as a joint reference, the predicted leg targets jump between chunks 10-20 times more than the expert's,
  and the robot turns and walks away.

Both v1 checkpoints therefore score 0 % before any manipulation is tested.

**v2 (owner): the policy also outputs SONIC planner commands, and the evaluation's SONIC executes them.** The
evaluator runs the same chain the expert ran during collection:

```
policy: action.planner (30 Hz) -> SONIC planner (re-plans every 0.2 s) -> G1 reference motion
                               -> SONIC encoder + decoder (every 20 ms, closed loop on the measured state)
                               -> leg and waist targets
policy: arm and head targets, hand targets -> executed directly (PD; arms with gravity feed-forward)
```

This keeps SONIC in the loop for balance and walking, as in the G1 tasks (b6). It is not the same contract as b6:
the G1 policies output SONIC's 64-d motion token (one level below the planner, whole body including the arms). The H2
data holds no tokens, and the H2 expert drove its arms directly (the skill server overrode SONIC's arms), so the
token contract cannot be built from this data. The v2 contract stays the H2 one (body 31 + hands 44, as a17 and a18)
plus the planner command.

Evidence that the planner interface works in the evaluation loop: replaying the expert's walks through a sim-time
planner + SONIC with the expert's route follower, 15 of 15 walks finished without a fall and stopped 1.7-17 cm from
their targets (the expert's own stops: 3.6-12.2 cm). The training data already spans such stop differences (the
bulb relative to the pelvis when the grasp starts varies by 6 cm median, 12 cm at the 90th percentile), so the policy
has to place its hands from what it sees, not from a fixed trajectory.

What is still hard: with the walking handed to them, the v1 checkpoints did not grasp the bulb (GR00T stopped 4-6 cm
short and never closed the hand; Psi0 reached and once closed on time but 1-2 cm off). Grasp precision is the next
bottleneck after walking; see section 6.

## 2. Dataset: `humanoid-benchmark/a14-unscrew-bulb-expert-v2`

Private, like v1. It is v1 (`a14-unscrew-bulb-expert-v1`, data commit `da301631`) with two added columns; every v1
column, video and episode is unchanged.

| | |
|---|---|
| Episodes / frames | 100 / 185,638 at 30 fps (median episode 60.9 s) |
| Cameras in the files | head, chest, wrist (right wrist), third_person; 480 x 640 H.264 |
| Task text (`task`) | "Walk to the table's short edge, pick up the bulb with the left hand, walk along the table to the socket, insert the bulb with the left hand, turn to face the table and screw it in with the right hand's fingers until it lights up." |
| New `action.planner` | float32 [7], below |
| New `annotation.locomotion_source` | int64 [1], below |
| Data commit | `8411a6da` (`8411a6dac45d0097d6068a60307de0c2e3b621a7`, uploaded 2026-11-14; 610 files, sizes and every LFS SHA256 checked against the local export) |

### 2.1 Training inputs and targets (all three models)

| Role | Feature | Dims |
|---|---|---|
| Images **(owner: head and chest only)** | `observation.images.head`, `observation.images.chest` | 2 x 480 x 640 x 3 |
| State78 | `observation.state` (75: legs 0-11, waist 12-14, left arm 15-21, right arm 22-28, head 29-30, hands 31-74) + `observation.projected_gravity` (3) | 78 |
| **Action82** | `action.body_target` (31) + `action.hands` (44) + `action.planner` (7), in this order | 82 |
| Prompt | the task text above | |

Do not use `observation.images.wrist`: the wrist camera is on the right wrist while the left hand picks, carries and
inserts, so for the first ~41 s of an episode it shows the floor and the robot's own leg; the head view shows the
working hand in every phase. Do not use `third_person` (not a robot camera).

### 2.2 `action.planner` (the walking command)

One row per 30 Hz frame: the request the expert's walker sent to the SONIC planner at that frame. Directions are
**relative to the robot's current pelvis yaw** (egocentric), as cosine and sine so that they do not wrap.

| Index | Name | Meaning | Outside a walk |
|---|---|---|---|
| 0 | `planner_active` | 1 while a planner session runs (the walk, its initial turn and its final stop), else 0 | 0 |
| 1 | `walk` | 1 when the request asks to move (planner mode 1), 0 for idle / turn / stop (mode 0) | 0 |
| 2, 3 | `move_cos`, `move_sin` | commanded movement direction minus the pelvis yaw | 1, 0 |
| 4, 5 | `face_cos`, `face_sin` | commanded facing direction minus the pelvis yaw (a turn in place changes only this) | 1, 0 |
| 6 | `speed_m_s` | commanded planner speed while `walk` = 1 (the expert's law: 0.25 m/s, 0.35 m/s for the square walk, a start ramp from 0.18 m/s, slowing to 0.15-0.2 m/s near the end), else 0 | 0 |

`annotation.locomotion_source` tells how the expert actually moved in that frame: 0 not moving, 1 planner walk (all
100 episodes, three walks each; 44,285 frames), 2 a short corrective step on the walking clip (12 episodes, 298
frames), 3 a turn in place through SONIC's joint reference (28 episodes, 2,504 frames). Sources 2 and 3 are encoded
in the same seven numbers as idle planner sessions (`planner_active` 1, `walk` 0): for 2 the facing holds the start
heading (those steps moved the pelvis only 1-4 cm), for 3 it follows the turn's reference yaw. The evaluator executes
all of them with the planner. Keep these frames for training; the flag is there for analysis or reweighting.

Over the dataset `planner_active` is 1 on 47,087 frames (25.4 %) and `walk` on 31,166 (16.8 %); the commanded speed
while walking is 0.15-0.35 m/s (median 0.25). Check of the encoding: on planner-walk frames moving faster than
0.1 m/s, the commanded movement direction is within 45 deg of the measured base velocity on 90.9 % of the frames
(median 20 deg; the gait sways the pelvis velocity within each step).

### 2.3 What the evaluator does with Action82

So that the targets you train on mean what the evaluator will execute (prototype runner in `debug/newton/`, built on
the sim-time planner and SONIC of `h2_sonic_planner.py`):

- Rows are executed at 30 Hz; the policy is queried every **16 rows** (0.53 s). Predict at least 16 rows per query.
- `body_target` 15-28 (arms): executed on PD, with the expert's gravity feed-forward from its arm takeover times
  (left 5.54 s, right 40.78 s). 29-30 (head): executed. **0-14 (legs, waist): not executed**, SONIC owns them; keep the
  columns in the action anyway (same layout as a17 / a18, and they are learnable), no special loss weight.
- `hands` 44: executed (Sharpa position targets).
- `planner`: thresholded at 0.5. A rising `planner_active` starts a planner session at the measured posture; a falling
  edge ends it (SONIC then holds the upper-body reference built from the policy's arm targets). Within a session every
  20 ms: mode = 1 if `walk` else 0; movement and facing = current pelvis yaw + atan2(sin, cos); speed clipped to
  0.15-0.5 m/s when walking. The planner re-plans every 0.2 s with the latest request.
- Episode: torso support for the first 3.08 s (as in the collection), budget 93 s, random layouts (layout seed
  1000 + episode). Success: the lamp lit, peak turns >= 0.30, robot standing, table / socket / power strip moved
  < 5 mm.

## 3. Common protocol (owner)

| Item | Value |
|---|---|
| Global batch | **512** |
| Steps | **4000** (about 11 epochs: 363 steps per epoch over 185,638 frames) |
| Checkpoints to publish | step 2000 and step 4000 |
| Cameras | **head + chest** |
| State / action | state78 / **action82** (section 2.1) |
| Planner output | **required**: the 7 `action.planner` dims, executed by the planner + SONIC |
| Split | all 100 episodes for training |
| HF path | `humanoid-benchmark/<model>/a14-unscrew-bulb/v2-b512/step-002000` and `step-004000` |

Each published checkpoint should carry, besides the weights: `protocol.json` (dataset repo and commit, cameras,
state78 / action82 layout with the planner names, global batch, steps, recipe), the normalization statistics it was
trained with, the training config, the training log and loss curve, and the inference code the evaluation needs. The
3-camera re-trains published a README camera column that disagreed with their configs; please keep the card and
`protocol.json` consistent.

## 4. Per-model recipes

### 4.1 GR00T N1.7: full fine-tune (owner)

Start from the current a14 recipe (`groot/a14-unscrew-bulb/step-004000`: `protocol.json`, `experiment_cfg/conf.yaml`,
`code/hb_groot/`) and change only:

| Item | v1 a14 recipe | v2 |
|---|---|---|
| Dataset | a14-unscrew-bulb-expert-v1 @ b4390fc | a14-unscrew-bulb-expert-v2 |
| Cameras (modality `video`) | head, chest, wrist | **head, chest** |
| Action modalities | `body_target` (31), `left_hand_target` (22), `right_hand_target` (22) | the same + **`planner`** (`action.planner`, 0-7) |
| Global batch | 256 (8 GPUs x 32) | **512** (e.g. 16 GPUs x 32) |
| Steps | 4000 | 4000 |
| Trainable | all modules | **all modules**: `tune_llm`, `tune_visual`, projector, diffusion action head, VLLN |

Unchanged: Cosmos-Reason2-2B VLM, embodiment tag `new_embodiment`, action horizon 16, lr 1e-4 cosine with warmup
ratio 0.05, weight decay 1e-5, augmentation spec v2 (15 visual effects, one per sample at 1.5x) with the proprio
augmentation of the GR00T A20 recipe (per-sample joint bias +-2 deg, noise 0.25 deg; no delay, no history drop),
model-native image augmentation off. Note: the v1 `protocol.json` field `model` still says "VLM frozen" while its
recipe trains all modules; write the v2 field to match what was trained.

### 4.2 Psi0: the a17_e20 configuration (the 5/50 bottle-cap checkpoint)

Use the configuration of `psi0/a17-unscrew-bottle/epoch-020` and change only what a14 v2 requires:

| Item | a17_e20 | a14 v2 |
|---|---|---|
| Dataset | a17-unscrew-bottle-expert-v1 @ 407e6e2 (100 episodes, 115,333 frames) | a14-unscrew-bulb-expert-v2 (100 episodes, 185,638 frames) |
| Prompt | the a17 text | the a14 task text (section 2) |
| Cameras | head, chest | head, chest |
| State | `observation.hb78` (78) | the same (state75 + gravity3) |
| Action | `action.hb75`, `pad_action_dim` 75 | **action82** = hb75 + `action.planner` (7), `pad_action_dim` 82; re-initialize the 6 input/output projection layers for 82 / 78 dims as was done for 75 / 78 |
| Steps | 4506 (20 epochs), warmup 113, checkpoint every 2253 | **4000**, warmup 100, checkpoint every 2000 |
| Global batch | 512 (16 GPUs x 32) | **512** |

Unchanged from a17_e20:

```yaml
data:
  image_size: 270x480 (resize + center_crop)
  action_norm_type: bounds
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
  pooled_text_encoder: clip-vit-large-patch14 (projection 768)
train:
  lr: 1e-4 (action head / projector), vision tower 1e-5, language backbone 1e-6
  scheduler: cosine, AdamW betas (0.95, 0.999), weight_decay 1e-6, grad clip 1.0
  precision: bf16, DDP, gradient checkpointing, seed 292285
```

With `bounds` normalization the binary planner dims map to -1 / 1 and the cos / sin dims stay in range; nothing else
is needed for them. For reference, a17_e20 ended at a loss of about 0.49.

### 4.3 DeltaFlow: the a18 configuration (the 31/100 pour checkpoint)

Use the configuration of `deltaflow/legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/epoch-040`
(`training_config.yaml`, `protocol.json`) and change only:

| Item | a18 epoch-040 | a14 v2 |
|---|---|---|
| Dataset | a18-pour-water-expert-v1 @ 8e956a10 (72,752 frames) | a14-unscrew-bulb-expert-v2 (185,638 frames) |
| Action / state | 75 / 78 (`h2_sharpa_body31_hand44`) | **82** / 78 (body31 + hand44 + planner7); new `normalization.json` refit on v2 |
| Text cache | the a18 instruction (Qwen3-4B, 128 tokens, dim 7680) | recompute for the a14 task text |
| Global batch | 256 (32 GPUs x 8) | **512** (32 GPUs x 16, or 64 x 8) |
| Steps | 11,400 (40 of a planned 60 epochs), warm-up 855 | **4000**, warm-up 200, cosine to 0 at step 4000 |
| Checkpoints | every 2,850 steps | steps 2000 and 4000 |

Unchanged: FLUX.2-klein-base-4B video expert trained with full parameters (no LoRA); the action DiT (5 double-stream
+ 20 single-stream blocks, width 1024, 24 heads) initialized from `action-shared-flux2-4b.pt`; mixture-of-transformers
in `noise_idm` mode; knowledge insulation (no action gradient into the video expert); FLUX.2 AE and Qwen3-4B frozen
(cached text features); per sample the head and chest frames at t (224 x 224 each, side by side 224 x 448), state78,
the 16 previous action rows (t-16..t-1), target the next 16 rows (t..t+15) and the two frames at t+16; losses action
flow matching 1.0 (from the previous chunk plus noise std 0.5), action one-step 0.5, future frame 0.5, frame one-step
1.0, scheduler shift 5.0 with 1000 timesteps; AdamW lr 1e-4, weight decay 0.01, grad clip 1.0; DeepSpeed ZeRO-1, bf16;
visual augmentation `b6-visual-aug-v2` (consistent across the two views and the future frame), no proprio
augmentation.

The training code is now published (`deltaflow/code/hb_deltaflow/`: `prepare_deltaflow_h2.py`, `train_deltaflow.py`,
`h2_hbaug.py`). The a18 evaluation had to reconstruct the model code, AE and text cache locally because only the
weights were published; please publish the full inference package (model code, normalization, text cache, a minimal
inference entry) with the a14 checkpoint.

## 5. Checks before publishing

- Loss curves without spikes; report the final loss (mean of the last 50 logged steps).
- An offline check on a few training episodes (open loop on recorded inputs): `planner_active` and `walk` agree with
  the labels on most frames and switch near the same times; the predicted movement / facing direction errors are
  small while walking; the arm columns beat the "hold the current pose" baseline. The evaluation side runs the same
  kind of check (`debug/newton/h2_policy_offline_check.py`) before any closed-loop run.
- The camera list, action layout and dataset commit in `protocol.json` match the training config.

## 6. Known risks

- **Grasp precision.** Most 3-camera re-trains stopped short of contact: GR00T a18 2-3 cm short of the handle (on
  that checkpoint, swapping only the wrist image at the stall turned 0/8 hand closures into 8/8), GR00T a17 never
  started the reach, Psi0 a17 hovered 6-7 cm above the cap, GR00T a14 stopped 4-6 cm short of the bulb. Psi0 a18
  (3 cameras) did grasp and pour but spilled. The only non-zero closed-loop results so far come from 2-camera
  checkpoints (DeltaFlow a18 31/100, Psi0 a17_e20 5/50). Head + chest follows that evidence, but the 2- and 3-camera
  checkpoints also differed in recipe, so the camera effect is not isolated. If budget allows, a 3-camera twin with
  the same recipe would settle it.
- **Long horizon.** Pick, carry, insert and screw must all succeed; the expert itself succeeded in 57 % of its
  attempts, and walk 2 (carrying the bulb) was its most common failure.
- **Planner sample differences.** The evaluator runs the planner ONNX on onnxruntime; the collection ran it through
  TensorRT. The same command can give a slightly different gait, most visibly when walk 2 starts with the bulb held out.
