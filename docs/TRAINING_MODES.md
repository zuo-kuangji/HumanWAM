# G1 training modes

HumanWAM maintains two ImageWAM Delta-flow training modes for G1 with Inspire hands.
Run commands from the repository root, using the dedicated ImageWAM environment.
The lightweight `humanwam` package does not install FLUX.2 or the training dependencies.
For checkpoint files, persistent inference services and SONIC integration, see the
[deployment guide](G1_DEPLOYMENT.md).
New generated training configs apply paired RGB augmentation with a 50% sample-level gate
shared by both cameras. Transform probabilities are conditional on that gate; each camera
otherwise samples its own parameters, shared between its current and future frames.
Current-state noise has probability 0.25, and a single state-conditioning dropout gate has
probability 0.1. The state gate zeros the complete projected feature, including its bias;
FM and IC share it, and raw measurements, labels, history and the root anchor stay unchanged. Use
`--no-augmentation` for a clean-input ablation or `--augmentation-config PATH` for a custom
v2 profile. Legacy configs without this field remain disabled; enabled v1 training is rejected
rather than reinterpreted, while its checkpoint provenance remains readable. See the
[augmentation settings](DATA_AUGMENTATION.md) for exact probabilities and separate
Delta-flow source noise settings. Inference does not apply training augmentation.

| Mode | Predicted physical action after denormalization | History / prediction / execution | SONIC v1.1 interface |
|---|---|---|---|
| `motion_token` | 64D motion token + 12D measured Inspire hand feedback | New training defaults to 16 samples; legacy 40-step checkpoints retain their own protocol | Existing external-token interface; hands remain separate |
| `joint` | 29D joint position + 12D measured Inspire feedback + 6D root rotation | 16 history samples, 40 predictions, 16 source execution intervals, 24 predicted lookahead samples | World-frame joint references, derived joint velocities and root quaternion through the joint-reference encoder |

Joint mode's additional predictions provide the controller's future window. They are not a
40-step execution chunk. Recorded data are 30 FPS and this recipe samples every second frame,
so adjacent joint predictions are `2/30` seconds apart. The 40 predictions span 2.6 seconds.
SONIC tracks at 50 Hz; this does not make the model's prediction interval 0.02 seconds.

## Shared method

Both modes use current body/hand/gravity state44 in the action expert only. State is not
appended to image or text tokens. Action losses do not backpropagate into the visual expert.
The visual source is `0.5 * z_current + sqrt(0.75) * noise`. The action source adds
Gaussian noise with standard deviation 0.5 to normalized historical actions, including the
continuous Inspire feedback coordinates. Both branches use flow matching and endpoint
consistency. AE and Qwen are frozen; the visual DiT, action expert and state projection train.

Joint mode extends its 16-point historical **source prior** by holding its last point for
24 additional positions before adding noise. All 40 **targets** are genuinely future
recorded samples, with episode-end padding masked from loss. Holding a source prior does
not create usable future predictions and is never used to fill a missing deployment window.

## Prepare data

Use a new `WORK_DIR` for each mode and dataset. Source datasets are read-only. All episodes
are used for training; these recipes do not create a held-out validation split.

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/humanwam/src${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_DIR=/absolute/path/to/lerobot_dataset
export WORK_DIR=/absolute/path/to/new_run
```

For motion-token mode:

```bash
python -m imagewam.datasets.wholebody \
  --dataset-dir "$DATASET_DIR" --output-dir "$WORK_DIR/inputs"
```

For joint mode on the existing G1 recordings:

```bash
python -m imagewam.datasets.wholebody_joint_reference \
  --dataset-dir "$DATASET_DIR" --output-dir "$WORK_DIR/inputs" \
  --recipe-version v2 --body-target measured \
  --orientation-target predicted_rotation6d \
  --action-horizon 40 --past-action-size 16 --execution-horizon 16 \
  --action-stride 2 --future-image-offset 32 \
  --source-contract configs/wholebody/source_contracts/g1_existing_measured_v1.json
```

Existing recordings contain measured body joints and raw base-IMU quaternions from the
same telemetry snapshot. They do not contain a verified complete upstream 29D SONIC
reference. The example therefore declares `reference_from_demonstration`, records its
coordinate/provenance audit, and leaves hardware synchronization unverified. It does not
claim equivalence to the original teleoperation reference or a validated robot controller.
See the [source audit](policies/REFERENCE_LABEL_AUDIT_20261110.md).

The body source is always explicit. `wbc` means recorded decoder motor targets and is a
separate experiment. `direct_reference` requires actual reference fields, matching body/root
provenance and a validated source contract; it never falls back to `action.wbc`, measured
positions, or zero-filled teleoperation wrist packets.

## Initialize and train

Prepare the FLUX.2 base, AE, pure-instruction text cache and audited 114-tensor shared action
initializer following [model initialization](MODEL_INITIALIZATION.md). A 76D task checkpoint
is not a 47D joint checkpoint. Each mode fits and hashes its own normalization statistics.

```bash
python scripts/wholebody/cache_text.py \
  --dataset-dir "$DATASET_DIR" --cache-dir "$WORK_DIR/textcache" \
  --qwen-model "$QWEN3_MODEL_DIR" --flux2-src "$FLUX2_SRC"

# Choose exactly one: motion_token or joint.
export MODE=joint
python scripts/wholebody/build_config.py \
  --representation "$MODE" \
  --dataset-dir "$DATASET_DIR" --stats-path "$WORK_DIR/inputs/dataset_stats.json" \
  --text-cache-dir "$WORK_DIR/textcache" --output-dir "$WORK_DIR/configs" \
  --flux2-src "$FLUX2_SRC" --flux2-model "$FLUX2_MODEL_PATH" \
  --ae-model "$FLUX2_AE_MODEL_PATH" --qwen-model "$QWEN3_MODEL_DIR" \
  --action-initializer "$ACTION_INITIALIZER" \
  --epochs 60 --batch-size 8 --accumulation 2 --num-processes 8

accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 scripts/wholebody/train.py \
  --config "$WORK_DIR/configs/smoke.yaml" --smoke

accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 scripts/wholebody/train.py \
  --config "$WORK_DIR/configs/production.yaml"
```

The builder measures the native distributed loader to derive optimizer steps, warmup and
save intervals. Production requires matching five-update smoke receipts from every rank,
then starts independently from the base. Batch 8 on eight GPUs is a requested configuration,
not a measured memory guarantee for the new joint recipe. Change the requested batch before
building and repeat the smoke if the hardware requires a different setting.

## Joint inference boundary

The network predicts root rotation relative to the **current** measured IMU heading. The
backend uses that current heading to restore world-frame quaternions before SONIC encoding.
The anchor is current state, not a future observation or a visual input. Historical root
actions supplied to the backend remain world-frame rotations; the backend transforms them
to the same current-heading frame used during training.

Use `configs/humanwam/imagewam-joint47.json` with a matching v2 checkpoint, model config,
statistics and text cache. Supply the real observation time in the policy host's monotonic
clock. Warm the model before collecting the first observation intended for execution;
cold-start time is not removed from the latency check.

The [SONIC bridge](policies/SONIC_JOINT_REFERENCE.md) checks the full execution interval plus
0.9 seconds of lookahead, accounts for inference delay and the 50 Hz grid, derives physical
joint velocity from the same position trajectory, and rejects stale or incomplete plans.
It exports native body v1 bytes and Inspire12 separately. It does not reinterpret Inspire12
as the official Dex3 14D hand command, open a socket, or actuate the robot.

## Compatibility and checks

The original `motion76`, `joint47` and `joint41` representation names remain accepted aliases.
`joint41` and `imagewam-joint47-legacy-v1.json` retain the earlier explicit legacy protocol.
V1, v2 and motion-token weights, statistics, source contracts and temporal schemas are not
interchangeable. No existing task checkpoint is overwritten or silently partially loaded.

Run the complete CPU checks from a native training environment:

```bash
python scripts/wholebody/check_two_routes.py --output-dir /absolute/path/to/new_cpu_report
```

Native loss fixtures run in separate processes to avoid contaminating real dataset imports.
These checks cover data decoding, Hydra construction, FM/endpoint losses, gradient isolation,
heading conversion, checkpoint separation, temporal plans and protocol bytes. They do not
load a full FLUX model, establish GPU throughput, train a joint policy, or establish robot
task success. Those stages require the actual checkpoint and deployment environment.
