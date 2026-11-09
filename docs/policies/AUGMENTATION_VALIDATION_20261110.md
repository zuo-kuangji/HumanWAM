# Training augmentation verification, 2026-11-10

The motion76 and joint47 v2 dataset paths now support the shared training-only profile in
[`augmentation_v1.yaml`](../../configs/wholebody/augmentation_v1.yaml).
Defaults, units, switches and inference behavior are documented in
[DATA_AUGMENTATION.md](../DATA_AUGMENTATION.md).

The final source snapshot passed **431 CPU tests with zero skips** in the native ImageWAM
environment. Tests ran as eleven isolated processes with `CUDA_VISIBLE_DEVICES=""`.
The 131 Python/YAML files used for this check matched the local release snapshot by SHA256.

| Check group | Passed |
|---|---:|
| Policy runtime and HTTP contracts | 161 |
| Joint label/rotation contracts | 14 |
| Motion dataset | 22 |
| Joint dataset | 22 |
| Native FM/IC and checkpoint routing | 66 |
| Training CLI | 23 |
| Joint v2 CLI | 4 |
| Pure augmentation config | 58 |
| Paired RGB augmentation | 29 |
| Current-state augmentation and dataset RNG | 18 |
| Augmentation CLI/provenance/epoch hook | 14 |

Coverage includes paired current/future transform parameters, independence of the current
transform from future image contents, cache protection, small state-mask limits, unchanged
action labels/history/root anchors, worker/rank/epoch RNG separation, and persistent-worker
epoch propagation before prefetch. The raw joint smoke schema accepts target40/history16.
Config, dataset, build receipt, smoke receipt and checkpoint provenance bind the full
canonical profile and its SHA256. Legacy checkpoints lacking augmentation metadata remain
loadable, and inference does not construct the training augmenter.

A separate lightweight run passed 270 tests, with four native-Torch cases skipped because
that environment has no Torch; all four ran in the native CPU checks above. Package lint
and Python syntax checks passed. A private real-camera montage was inspected for exposure,
color, small affine jitter, blur and current/future pairing; no source image was committed.

Reproduce the full CPU check in the ImageWAM environment:

```bash
python scripts/wholebody/check_two_routes.py --output-dir /absolute/path/to/new-report
```

Native method tests use small CPU backbone/scheduler fixtures. These checks do not load the
full FLUX.2 weights, measure GPU performance, perform optimizer updates or operate a robot.
The augmentation defaults are initial engineering settings, with no task-success improvement
claim. Existing production runs and original datasets were not changed.
