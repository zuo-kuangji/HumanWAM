# Augmentation v3 validation — 2026-11-10

The default new-training profile is `configs/wholebody/augmentation_v3.yaml`.
It adds Gaussian pixel noise with sigma sampled uniformly from 4/255 to 8/255,
JPEG quality sampled uniformly from integers 50 through 80, and affine jitter
with translation up to 4%, rotation up to 3 degrees and scale 0.96–1.04.
The 50% sample-level visual gate remains shared by both cameras. Pixel noise
and JPEG each have conditional probability 25% (overall 12.5% per camera);
jitter retains conditional probability 40% (overall 20%).

The current/future pair shares parameters, including the pixel-noise seed and
noise map. JPEG encodes each frame separately with the same quality and 4:4:4
sampling. Tests verify that changing a future image does not change the
current-image transformation, explicit parameters do not consume global RNG,
and cached input tensors remain unchanged. Both state44 and numerical action
labels retain their existing definitions.

Native CPU validation used Python from `/mnt/data1/zuo/AAA/.venv/bin/python`,
Torch 2.7.1, Pillow 12.0.0 and `CUDA_VISIBLE_DEVICES=""`:

| Check group | Passed |
|---|---:|
| Canonical configuration and v1/v2 compatibility | 86 |
| Paired visual transforms, noise and JPEG | 52 |
| Dataset augmentation, immutable labels and state gates | 28 |
| Augmentation CLI and v2/v3 smoke contracts | 19 |
| Whole-body training CLI | 23 |
| Joint v2 training CLI | 4 |
| Native FM/IC, state isolation and checkpoint behavior | 71 |
| Motion-token dataset | 22 |
| Joint-reference dataset | 22 |
| Policy/runtime contracts | 161 |
| Total | 588 |

All groups passed without skips. One initial policy/runtime run lacked two
unchanged CLI helper files in the isolated QA snapshot; adding those helper
files resolved both failures, and the complete 161-test group passed.
Configuration hashing preserves existing v1 and v2 canonical profiles and
their recorded SHA256 digests. Explicit v2 training remains supported and
does not acquire the new operations or larger jitter ranges. Inference
applies none of these training augmentations.

The isolated logs are under
`/mnt/data0/aaa/humanwam_augmentation_qa_20261110/augmentation_v3_native/report_v3`.
Local configuration/CLI checks, Python compilation, targeted lint and
`git diff --check` also passed. These are software checks, not training or
real-robot performance results. Existing training jobs were not modified.
