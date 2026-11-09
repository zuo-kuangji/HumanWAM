# Probability-gated augmentation v2 verification, 2026-11-10

The v2 recipe implements the requested three sample-level events:

| Event | Probability | Application |
|---|---:|---|
| Visual pipeline | 50% | One shared on/off gate for both cameras; paired current/future transforms |
| Small current-state noise | 25% | Body29 std 0.01 and Inspire12 std 0.005 in normalized units |
| State conditioning dropout | 10% | Zero the entire encoded state condition, including projection bias |

Each event is sampled once per training sample. FM and endpoint consistency share the same
state and condition gate. Numeric action labels, action history and raw heading anchors
remain unchanged. Visual sub-operation probabilities are conditional on the 50% pipeline
gate; blur remains a mild Gaussian filter, without image erasing or complete obscuration.
See [the settings and upstream references](../DATA_AUGMENTATION.md).

The final v2 snapshot passed **456 CPU tests with zero skips** in the native ImageWAM
environment. 132 Python/YAML files matched the released source snapshot by SHA256.

| Check group | Passed |
|---|---:|
| Policy runtime and HTTP contracts | 161 |
| Joint source/rotation contracts | 14 |
| Motion / joint datasets | 22 / 22 |
| Native FM/IC/checkpoint routing | 71 |
| Training / joint-v2 CLI | 23 / 4 |
| Augmentation configuration | 64 |
| Paired visual augmentation | 31 |
| Probabilistic state noise / dataset gates | 27 |
| Augmentation CLI / smoke contracts | 17 |

The native checks verify that dropped samples contribute no state-encoder bias or weight
gradient, a changed state cannot affect dropped FM/IC predictions, and inference retains
real state even if a training profile requests 100% dropout. Dataset checks cover one shared
visual gate, probabilistic noise, scalar-to-batch keep flags, clean labels/history, camera
pairing, cache protection and worker/epoch RNG. Smoke validation accepts a legitimate
all-dropped microbatch with zero local state-encoder gradient.

Old v1 canonical configuration and digests remain readable as checkpoint provenance.
Enabled v1 training is rejected instead of silently reinterpreting its dimension masking.
The separate lightweight run passed 279 tests, with four Torch cases skipped; the native
CPU run above covered those cases. Package lint, syntax and whitespace checks passed.

A private two-camera montage shows original images and individually triggered exposure,
overexposure, color, small affine jitter, Gaussian blur and combined examples. Individual
operations are forced only in that comparison; training uses the configured event rates.
No source image was added to the repository.

Reproduce in the native ImageWAM environment:

```bash
python scripts/wholebody/check_two_routes.py --output-dir /absolute/path/to/new-report
```

Tests use CPU fixtures for backbone/native-method routing and do not load the full FLUX.2
weights, measure GPU performance, perform optimizer updates or actuate a robot. No task
success improvement is claimed, and existing production runs and datasets were not changed.
