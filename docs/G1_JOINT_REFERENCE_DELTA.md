# G1 joint47 ImageWAM Delta-flow v2

The existing [motion76 recipe](G1_WHOLEBODY_DELTA.md) predicts `SONIC motion64 + measured Inspire12`. This separate joint-reference recipe predicts `body29 + measured Inspire12 + root rotation6D = 47D`. It needs its own projections, statistics, source contract and checkpoint. Released motion76 weights are not joint predictions. Fast-WAM's native Gaussian FM interface is documented separately in [FASTWAM.md](policies/FASTWAM.md).

The v2 joint47 recipe predicts **40 source-time samples**, keeps **16 historical samples**, and executes **16 source-time samples** before replanning. The other24 predictions provide actual model lookahead for SONIC. This differs from the v1 H16 joint prototype and legacy H40 motion-token model.

## Declare the label source

`--body-target` must be chosen explicitly. All v2 preparations require a source-contract JSON; its SHA256 is preserved in statistics, generated configuration and checkpoint metadata.

| Choice | Body label | Scope |
|---|---|---|
| `direct_reference` | Verified original motion-reference qpos with matching root quaternion | Requires verified frame, pairing and synchronization |
| `measured` | Recorded `observation.state` body29 | Measured-trajectory prediction experiment |
| `wbc` | Recorded `action.wbc` body29 | WBC motor-qtarget prediction experiment |

The audited clothing, trash-bin and pillow collections have measured joints and WBC motor targets. They do not have a verified full original29D reference field. The pose sender's `joint_pos` is initialized to zeros and only its wrist entries are assigned; it is not a full-body original reference. See [the recorder audit](policies/REFERENCE_LABEL_AUDIT_20261110.md) for actual source paths, excerpts and SHA256 fingerprints.

For an experiment on these collections, [g1_existing_measured_v1.json](../configs/wholebody/source_contracts/g1_existing_measured_v1.json) pairs measured joints with `observation.root_orientation`. It records same-stage pairing and a verified IMU frame, while leaving hardware synchronization unverified. It does not certify deployment readiness or original-reference equivalence. The separate [WBC experiment example](../configs/wholebody/source_contracts/g1_existing_wbc_experiment_v1.json) explicitly records `same_stage_pairing=false`: it combines a previous motor target with a measured root, not a matched original q/root reference. A `direct_reference` dataset needs newly verified source fields and provenance; no such contract is supplied for these recordings.

The source contract declares the original qpos key,29 explicit indices/names, radian units, root key/order/frame, validity policy, pairing and provenance. Its `root_frame=world` describes the recorded quaternion, not the frame of the model's6D target.

## State, output and heading frame

```text
action47 = body29 | left Inspire6 | right Inspire6 | root rotation6D
state44  = current body29 | current Inspire12 | projected_gravity3
```

Body joint names follow the explicit dataset canonical order and are reordered by name at the SONIC boundary. Hands use the same target row's `observation.inspire_left_hand_state` and `observation.inspire_right_hand_state`, ordered `[pinky, ring, middle, index, thumb_bend, thumb_rotation]`, in `[0,1]`. These are continuous measured states, not binary switches or Dex3's7D commands.

Root means the floating base/pelvis orientation, not torso-link orientation or root xyz. The audited `observation.root_orientation` is raw base-IMU wxyz, without an additional recorder heading transform. Here world means the IMU estimator frame; geographic north and a shared frame across IMU restarts are not established. Raw `teleop.body_quat_w` is a different SMPL/pelvis signal and cannot silently replace it.

Gravity does not identify global yaw. The v2 target uses one heading frame anchored at the current observation:

```text
R_relative[k] = Rz(-yaw(current_root_quaternion)) @ R_world[k]
R_world[k]    = Rz(yaw_anchor) @ R_relative[k]   # inference restoration
```

Past and future rotations in each sample share the same anchor. Rotation6D concatenates matrix columns `[R00,R10,R20,R01,R11,R21]`; its normalization bounds are fixed `[-1,1]`. At inference the current measured root quaternion supplies only the coordinate anchor. No future quaternion is an inference input.

Current numerical state enters only the action expert's projection. It is not appended to image or text tokens. Visual FM/IC updates the visual branch; action FM/IC uses stop-gradient on visual conditioning. The quaternion anchor does not become a visual/text condition. Body velocity is calculated at deployment, not predicted by an extra29D head.

## Time sampling and actual lookahead

Data is30 FPS with stride2, so adjacent action samples are separated by `2/30 s`, not0.05s. For anchor `t`:

| Content | Original-frame offsets |
|---|---|
| Historical actions | `-32,-30,...,-2` (16) |
| Predicted/supervised actions | `0,2,...,78` (40) |
| Execution segment | first16 source samples, offsets `0..30` |
| Additional predicted lookahead | remaining24, offsets `32..78` |
| Current/future image | `0 / 32` |
| Proprioception | current anchor only |

The predictions span2.6s. Executing16 source intervals gives a nominal replan deadline of `16*2/30 = 1.0667s`; this is not16 low-level50Hz ticks. The visual target remains offset32, at the execution boundary. Windows never cross episodes; target padding and its loss mask remain explicit.

The source prior uses the16 normalized historical samples followed by24 repetitions of the last historical sample, then adds noise. At episode start with no history it uses normalized-zero history. This repetition only constructs the source prior. All40 clean targets are real recorded samples; all40 deployment references are model predictions. No extra future coverage is created by holding the last predicted output.

SONIC mode0 needs10 references at `0,0.1,...,0.9s`. Coverage must remain available through the full execution interval, including inference latency and declared jitter. A trajectory that covers the first window can still run out during execution. The bridge checks actual timestamps and execution interval and rejects missing lookahead; it does not pad or extrapolate references. See [SONIC_JOINT_REFERENCE.md](policies/SONIC_JOINT_REFERENCE.md).

Current/future ego and chest images are independently resized to224×224 and joined ego-left/chest-right into224×448. Visual source is `0.5*z_current + sqrt(0.75)*epsilon`. Action source adds `0.5*epsilon` to all47 normalized dimensions without extra variance normalization. Visual FM/IC has total weight0.5 and IC weight1; action FM has weight1 and IC weight0.5. Both IC losses are one-step L1.

## Prepare data, base initialization and training

Run from the repository root, using the independent environment and local FLUX.2/AE/Qwen assets in [the environment guide](G1_WHOLEBODY_DELTA.md#4-环境与外部模型) and [MODEL_INITIALIZATION.md](MODEL_INITIALIZATION.md). This example explicitly uses measured demonstrations, not an original-reference dataset.

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)/packages/humanwam/src${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_DIR=/path/to/lerobot_dataset
export WORK_DIR=/path/to/new_joint47_v2_run
export FLUX2_SRC=/path/to/flux2
export FLUX2_MODEL_PATH=/path/to/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/path/to/ae.safetensors
export QWEN3_MODEL_DIR=/path/to/Qwen3-4B
export SOURCE_CONTRACT="$(pwd)/configs/wholebody/source_contracts/g1_existing_measured_v1.json"

python -m imagewam.datasets.wholebody_joint_reference \
  --dataset-dir "$DATASET_DIR" --output-dir "$WORK_DIR/inputs" \
  --recipe-version v2 --body-target measured \
  --orientation-target predicted_rotation6d --source-contract "$SOURCE_CONTRACT" \
  --action-horizon 40 --past-action-size 16 --execution-horizon 16 \
  --action-stride 2 --future-image-offset 32

python scripts/wholebody/cache_text.py \
  --dataset-dir "$DATASET_DIR" --cache-dir "$WORK_DIR/textcache" \
  --qwen-model "$QWEN3_MODEL_DIR" --flux2-src "$FLUX2_SRC"

python scripts/flux2/preprocess_action_dit_flux2.py \
  --model-config configs/wholebody/deltaflow_joint_reference_p25.yaml \
  --flux2-src-path "$FLUX2_SRC" --flux2-model-path "$FLUX2_MODEL_PATH" \
  --variant klein-base-4b --action-dim 47 --apply-alpha-scaling true \
  --device cpu --output "$WORK_DIR/action_shared_flux2_4b.pt"

python scripts/wholebody/build_config.py \
  --representation joint --horizon 40 --execution-horizon 16 \
  --dataset-dir "$DATASET_DIR" --stats-path "$WORK_DIR/inputs/dataset_stats.json" \
  --text-cache-dir "$WORK_DIR/textcache" --output-dir "$WORK_DIR/configs" \
  --flux2-src "$FLUX2_SRC" --flux2-model "$FLUX2_MODEL_PATH" \
  --ae-model "$FLUX2_AE_MODEL_PATH" --qwen-model "$QWEN3_MODEL_DIR" \
  --action-initializer "$WORK_DIR/action_shared_flux2_4b.pt" \
  --epochs 60 --batch-size 8 --accumulation 2 --num-processes 8

accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 scripts/wholebody/train.py \
  --config "$WORK_DIR/configs/smoke.yaml" --smoke

accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 scripts/wholebody/train.py \
  --config "$WORK_DIR/configs/production.yaml"
```

Preparation checks metadata, Parquet fields and timing, fits task-specific statistics on all episodes, and saves source hashes without changing original data or creating a validation split. Contract or dataset changes require fresh statistics and a fresh run directory. The builder constructs the real CPU loader to determine updates and schedule.

New configs use augmentation v2. A 50% sample-level gate enables paired RGB transforms for both cameras; transform probabilities are conditional on it, and each camera shares its parameters between current and future frames. Current-state noise has probability 0.25, and one state-conditioning dropout gate has probability 0.1. The state gate zeros the whole state-encoder output including bias and is shared by FM/IC, without changing raw measurements, labels, action history or the root heading anchor. Use `--no-augmentation` for the ablation or `--augmentation-config PATH` for a custom v2 profile. The full profile and SHA are recorded. Enabled v1 training is rejected while its historical provenance remains readable; inference applies neither noise nor the conditioning gate. See [augmentation settings](DATA_AUGMENTATION.md).

The initializer reuses shared FLUX.2 backbone tensors. Joint47 action input/output and state44 projections are newly initialized. Old motion76 task checkpoints and optimizers are not silently partially loaded. VAE and Qwen stay frozen; visual DiT, action expert and current-state projection train. Contract SHA, statistics, text cache, temporal recipe and orientation frame are checked on checkpoint loading.

The example's8 GPUs × batch8 × accumulation2 gives nominal global batch128. It is not a memory measurement for this new40-sample joint recipe. Default LR is1e-4,5% warmup, cosine to1e-6, WD0.01, BF16, seed42 and ZeRO-1; FlashAttention is not required. Complete five actual optimizer updates and rank checks before production. Smoke saves no model/optimizer; production starts again from the base initializer. Use a new output directory. Default retention is four recent weights and one latest training state. After batch/epoch changes, regenerate schedule and repeat the check.

## Inference and verification boundary

The separate `imagewam-joint47` config uses `IMAGEWAM_PYTHON`, `IMAGEWAM_JOINT_MODEL_CONFIG`, `IMAGEWAM_JOINT_CHECKPOINT` and `IMAGEWAM_JOINT_TEXT_CACHE`. Supply current uint8 HWC `ego_rgb`/`chest_rgb`, physical `current_state[44]`, and at most16 physical `past_actions[N,47]` with `past_action_frame_offsets=[-2*N,...,-2]`. Past rotations at this API boundary are **world-frame**6D; the backend transforms them to the current heading before normalization. Optional `past_root_orientation_valid[N]` identifies invalid recorded orientation rows. Do not pass past rotations left in an older anchor frame or pre-normalized actions.

Also supply `current_root_quaternion_wxyz[4]` as the current measured IMU-world heading anchor, actual `observation_time_s` in the policy host's monotonic clock, and `timestamp_clock="monotonic"`. The timestamp is the observation capture time, not a fresh timestamp assigned after inference. Empty history requires `at_episode_start=true`. No future image/state is accepted. The backend returns restored-world rotations and both observation/inference-ready timestamps for the runtime checker.

Inference returns40 physical samples, with16 designated for execution. It does not send motor commands. It does not predict root xyz, duration,29 additional physical velocities, or the feedback controller's output. The legacy joint41/external-root path remains an offline compatibility route and requires an actual future root trajectory; it cannot silently use current IMU or identity rotations. A v1 H16 or joint41 checkpoint cannot replace v2 joint47.

CPU checks establish software/data/protocol contracts only. They do not establish a complete new FLUX.2 GPU training run, SONIC encoder/decoder closed-loop equivalence or physical task success. Measured and WBC recipes remain explicitly experimental until their control chains are evaluated.
