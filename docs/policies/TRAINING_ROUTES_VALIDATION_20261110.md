# Training-route validation, 2026-11-10

The motion-token and joint-reference routes passed **295 CPU tests, with no skips**, in
the native ImageWAM environment on NTU-ZUO. CUDA was explicitly hidden. Each test group
ran in a separate process so native-loss fixture stubs could not replace real dataset modules.
The [machine-readable receipt](training_routes_cpu_20261110.json) records counts and hashes
of the changed executable source and tests.

| Group | Passed | What was exercised |
|---|---:|---|
| Policy and runtime | 146 | Backend inference handoff, world-heading restoration, timed buffer, stale/overrun rejection, native SONIC bytes, pinned source compatibility and existing policy adapters |
| Source and rotation | 14 | Provenance, canonical hashes, Hydra mapping conversion, heading transforms and invalid contracts |
| Motion-token dataset | 22 | Existing action76 mapping, current state44, causal windows, normalization and video decoding |
| Joint dataset | 22 | Explicit measured/direct/WBC fields, real Hydra dataset construction, H40 targets/history16, root masks, heading anchors and video decoding |
| Native losses and protocols | 64 | Actual FM/endpoint loss routines in CPU fixtures, state isolation, action-to-visual stop-gradient, new projections, factory construction and incompatible checkpoint rejection |
| Training CLI | 23 | Existing modes, measured native loader lengths, schedules, smoke receipts and fresh-run guards |
| Joint v2 configuration | 4 | Generated prediction40/history16/execution16 configs and rejection of source, frame or timing drift |

A separate lightweight run passed 185 tests and 145 subtests. Its two Torch-dependent
checks were skipped locally and ran successfully in the native CPU run above. Ruff and
`git diff --check` also passed.

End-to-end testing identified and corrected three integration defects before publication:

- Hydra passes nested `DictConfig`/`ListConfig`; source contracts and recipes now convert
  these explicitly to plain values without changing unrelated model configuration types.
- Episode-start inference now accepts empty JSON history and produces normalized zero
  history without inventing a measured root pose.
- Large monotonic clock values caused cancellation when deriving each relative time from
  absolute times. The runtime now constructs the relative 50 Hz grid from small offsets.
  Regressions cover 1e7- and 1e8-second clocks, non-grid observations, stable derivatives
  and native packet construction; the existing grid-precision check remains enabled.

These checks do not construct the full FLUX model or load its multi-billion-parameter
weights. The factory test mocks only the bottom-level base-model constructor; the factory
and its contracts execute. The backend handoff test substitutes a small deterministic
native-model output and executes the real normalization boundary, rotation conversion,
runtime and serializer. GPU memory, five-update distributed smoke, trained joint-policy
quality and real-robot execution remain separate validation stages. No training or robot
service was started or interrupted by these checks.

Reproduce from the native training environment:

```bash
python scripts/wholebody/check_two_routes.py \
  --output-dir /absolute/path/to/new_cpu_report
```

See [training modes](../TRAINING_MODES.md), the [reference-label audit](REFERENCE_LABEL_AUDIT_20261110.md)
and the [SONIC boundary](SONIC_JOINT_REFERENCE.md) for the corresponding data and deployment contracts.
