# Integration validation, 2026-11-09

The lightweight package suite passed **40 tests and 23 subtests** on Python3.12.
It exercises the four lazy adapters, native command construction, no-target/
no-future-state input boundaries, action/horizon checks, OpenPI history and
normalization ownership, DiT4DiT single-current-camera semantics, safe NPZ
loading, configuration expansion, and dataset metadata validation.

The native ImageWAM whole-body CPU suite passed **65 tests, zero skipped** in
the existing NTU ImageWAM runtime with `CUDA_VISIBLE_DEVICES` empty. It exercises
native FM/IC method bodies and gradients, action-only current-state routing,
stop-gradient, H16 inference, explicit legacyH40, cross-horizon checkpoint
rejection, dataset windows, smoke/production contracts, and actual native
sharded-loader length calculation. Existing training jobs were not modified.

`humanwam doctor` verified all four upstream commits and exact allowed patches.
Bootstrap was run twice to verify that applied patches are recognized without
duplicating them. Package Ruff checks and Git whitespace checks passed.

These are code and CPU contract checks. This integration did not load actual
policy checkpoints for GPU inference and did not run a robot. Published
ImageWAM release wrappers retain the original release checks; existing custom
GR00T FM/ODE and R1Lite Torch formats keep their original runtimes rather than
being relabeled as native HF/OpenPI checkpoints.

Reproduce the lightweight checks:

```bash
python -m pip install -e 'packages/humanwam[test]'
python -m pytest packages/humanwam/tests -q
python -m ruff check packages/humanwam
python scripts/humanwam/bootstrap.py
humanwam doctor
```

In the separate ImageWAM environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src" python -m unittest discover \
  -s tests -p 'test_wholebody*.py' -v
```
