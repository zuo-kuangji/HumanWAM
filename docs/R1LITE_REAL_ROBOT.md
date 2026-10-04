# R1Lite real-robot I2I + A2A + IC

This is the real-robot adaptation of the ImageWAM/RoboTwin training workflow.
The cups run completed 20 epochs; flower uses the same protocol with a different
dataset and an instruction override. No physical rollout success rate is claimed.

## Validated protocol

| Setting | Value |
|---|---|
| Cameras | head RGB + right-wrist RGB |
| Image layout | Resize each to 224 x 224; concatenate horizontally to 224 x 448 |
| Action/state | Six right-arm joints + right gripper, 7D absolute targets |
| History / prediction | Past 16 logged actions / future 16 actions; preserve boundary padding |
| Normalization | Training-set q01/q99 for actions and current proprio |
| Visual source | `(current latent + Gaussian noise) / sqrt(2)` |
| Action source | Normalized past actions + 0.5 Gaussian noise on all 7 dimensions |
| Objectives | Visual/action flow matching + visual/action endpoint consistency |
| Gradient routing | Action losses do not backpropagate through visual K/V conditioning |
| Fine-tuning | Full visual/action experts and proprio encoder; not LoRA |
| Frozen components | Autoencoder and Qwen3-4B; cached text context length 128 |
| Hardware / batch | 8 x A100 80GB, microbatch 10, global 80, accumulation 1 |
| Optimizer | AdamW, LR 1e-4 to 1e-6, cosine, 5% warmup, weight decay 0.01 |
| Training | 20 epochs, bf16, ZeRO-1, activation checkpointing |
| Retention | Save every epoch; retain latest 4 weights and latest complete optimizer state |

At inference, use one visual step and one action step, **fresh noise per call**,
and the actual executed action history. The example never commands a robot.
Do not substitute measured states for the preceding logged/executed action targets.

| Dataset | Pinned revision | Episodes / frames | Steps per epoch / total (batch 80) |
|---|---|---:|---:|
| `at237299966/r1lite_place_multi_cups` | `935cd0d7f397573616435d245f558ab160da27c5` | 110 / 61,068 | 764 / 15,280 |
| `at237299966/r1lite_insert_flower` | `5c4b884303cf7be101b0b8611ad6362b5441db4a` | 98 / 45,684 | 572 / 11,440 |

## Installation and assets

Follow the repository's FLUX.2 setup, including its pinned `third_party/flux2`
checkout. The validated server used Python 3.11, PyTorch 2.7.1+cu128,
transformers 4.56.1, accelerate 1.12.0, deepspeed 0.18.5 and huggingface-hub 0.36.2.
Install the shared extras and `pytest` for CPU contract checks.

After the repository setup, use the validated FLUX dependencies rather than the
OmniGen2 transformer version. Do not change a shared environment while it is
running another training job.

```bash
uv pip install "transformers==4.56.1" "huggingface-hub==0.36.2" \
  "accelerate==1.12.0" "deepspeed==0.18.5" pytest
CUDA_VISIBLE_DEVICES= PYTHONPATH=src python -m pytest -q tests/test_r1lite_training_contract.py
```

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export FLUX2_SRC="$PWD/third_party/flux2"
export FLUX2_MODEL_PATH="$PWD/checkpoints/flux2/FLUX.2-klein-base-4B/flux-2-klein-base-4b.safetensors"
export FLUX2_AE_MODEL_PATH="$PWD/checkpoints/flux2/FLUX.2-dev/ae.safetensors"
export FLUX2_QWEN3_MODEL_SPEC=Qwen/Qwen3-4B
export ACTION_INIT="$PWD/checkpoints/action_dit_flux2_4b_r1lite_joint7_init.pt"
```

## Prepare selected views

Only the two selected cameras and matching data/metadata are downloaded.
The downloader pins the revision and checks file counts, byte lengths and LFS hashes.
The view builder changes camera metadata, not the original videos/actions.

```bash
python scripts/r1lite/download_selected_views.py \
  --destination "$PWD/data/r1lite_place_multi_cups"
python scripts/r1lite/prepare_data.py \
  --source "$PWD/data/r1lite_place_multi_cups" \
  --view "$PWD/data/r1lite_place_multi_cups_head_right" --repo "$PWD"

python scripts/flux2/preprocess_action_dit_flux2.py \
  --model-config configs/model/imagewam_flux2_klein_4b_base.yaml \
  --flux2-src-path "$FLUX2_SRC" --flux2-model-path "$FLUX2_MODEL_PATH" \
  --variant klein-base-4b --action-dim 7 --output "$ACTION_INIT"

python scripts/flux2/precompute_flux2_qwen3_embeds.py \
  task=r1lite_cups_a2ai2i_20ep model.flux2_src_path="$FLUX2_SRC" \
  model.qwen3_model_spec="$FLUX2_QWEN3_MODEL_SPEC" \
  data.train.dataset_dirs="[$PWD/data/r1lite_place_multi_cups_head_right]" \
  data.train.qwen_text_cache_dir="$PWD/data/r1lite_place_multi_cups_head_right/imagewam_training_meta/qwen3_flux2_len128"
```

The action initializer is derived from base FLUX.2 weights, **not a trained cups
or RoboTwin checkpoint**. The visual source is variance-normalized Mid-Anchor;
the action head stays at scale 1.0, without Soft-Zero.

For flower, use `--repo at237299966/r1lite_insert_flower`, its pinned revision,
`--expected-files 298`, separate raw/view paths, and
`--task r1lite_insert_flower_20ep` in `prepare_data.py`. Use the same task in
text precomputation. The override applies to every training sample:

> Use the right arm to pick up the flower from the transparent white vase on the left and insert it into the purple vase on the right.

The original dataset task metadata is unchanged. This means move the **flower**,
not pick up the vase.

## Train / resume

```bash
python scripts/r1lite/train.py --task cups \
  --dataset "$PWD/data/r1lite_place_multi_cups_head_right" \
  --output "$PWD/runs/r1lite_cups_20ep" --gpus 8 --batch-size 10
```

Use `--task flower` and separate dataset/output paths for flower. The launcher
computes the epoch milestones and warmup from the actual frame count; it refuses
to overwrite a nonempty output without explicit `--resume`. `--print-command`
checks preparation and prints the launch command without allocating GPUs.
Do not run concurrent jobs on occupied GPUs. Batch 12 OOM'd on the validated setup;
batch 10 passed five complete optimizer updates before production.

To resume a run, pass its original output and
`--resume /path/to/checkpoints/state/step_XXXXXX`, retaining the original schedule.
Inference `.pt` weights alone do **not** contain optimizer/scheduler/RNG state.
Store outputs on a data volume: a full eight-rank optimizer state is about 60 GiB,
each retained inference weight about 8.4 GiB, plus temporary checkpoint-write space.

## Final checkpoints and offline inference

[Download the final cups inference release](https://huggingface.co/ZUO66/imagewam-r1lite-place-multi-cups).
It contains only `step_015280.pt`, portable `config.yaml`, training
`dataset_stats.json` and SHA256 manifest. No optimizer, earlier checkpoint,
dataset video or credential is uploaded.

```bash
hf download ZUO66/imagewam-r1lite-place-multi-cups --local-dir checkpoints/r1lite_cups
python scripts/r1lite/infer.py --release checkpoints/r1lite_cups \
  --head head.png --right-wrist right_wrist.png --inputs raw_inputs.npz \
  --output prediction.npz --flux2-src "$FLUX2_SRC" \
  --flux2-weights "$FLUX2_MODEL_PATH" --ae-weights "$FLUX2_AE_MODEL_PATH"
```

`raw_inputs.npz` must contain `past_action` (16 x 7) and current `proprio` (7),
in the **same physical joint/gripper units and order as the dataset**. The example
uses the release's stats, verifies a strict full-checkpoint load and saves 16 x 7
absolute targets. It does not send predictions to hardware. By default it uses
fresh noise; `--seed` is only for explicit reproducibility tests.

[Download the final flower inference release](https://huggingface.co/ZUO66/imagewam-r1lite-insert-flower).
Its final checkpoint is `step_011440.pt` (20 epochs). It was trained from base
initialization, not continued from cups. Use the flower release's own config,
stats and instruction; do not mix them with cups.

```bash
hf download ZUO66/imagewam-r1lite-insert-flower --local-dir checkpoints/r1lite_flower
python scripts/r1lite/infer.py --release checkpoints/r1lite_flower \
  --head head.png --right-wrist right_wrist.png --inputs raw_inputs.npz \
  --output flower_prediction.npz --flux2-src "$FLUX2_SRC" \
  --flux2-weights "$FLUX2_MODEL_PATH" --ae-weights "$FLUX2_AE_MODEL_PATH"
```

The adapter selects the final checkpoint from `manifest.json` and defaults to
the instruction override in `config.yaml`. Flower therefore automatically uses
the right-arm, left-transparent-white-vase to right-purple-vase instruction.
An explicit `--instruction` overrides this only when intentionally requested.

Qwen can be loaded online, or supply `--text-cache` with the matching prompt's
precomputed `*.qwen3_flux2_len128.pt`. Keep camera order, normalization and prompt
identical to training. Validate joint limits, control conventions and emergency
stop behavior separately before any physical deployment.
