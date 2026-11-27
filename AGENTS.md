# Agent instructions for this repository

## Humanoid Benchmark training: read before scheduling, launching, continuing or uploading any training

Any task that trains, fine-tunes, continues or uploads a model on Humanoid Benchmark data must follow these files. This applies to every model (GR00T, Psi-Zero, FastWAM, π0.5, DiT4DiT, Delta Flow, ACT, …) and to every task, new or old.

1. **[`hb_training/REPRODUCIBILITY.md`](hb_training/REPRODUCIBILITY.md)** is the authoritative protocol. It covers:
   - seeds, batch and node layout, steps, learning-rate schedule and checkpoints for each model
   - data versions and episode rules
   - augmentation and proprioception
   - in-training MSE
   - what an upload must contain
   - the registry of exceptions
2. **[`hb_training/protocol.yaml`](hb_training/protocol.yaml)** holds the same values in machine-readable form.
3. **[`hb_training/check_protocol.py`](hb_training/check_protocol.py)** checks a run against the protocol. Run it on every finished run before you report or upload it.

Non-negotiable rules. Each one exists so that other researchers can reproduce our released checkpoints and results.

- **Align with the protocol in every respect it covers.** Same seed per model, global batch, gradient accumulation = 1, steps, learning-rate curve, augmentation spec v2, proprioception interface (G1 44 / H2 78, no joint velocity), cameras and data version.
  - Only an explicit user decision may change anything.
  - Record every deviation in REPRODUCIBILITY.md section 7, in `protocol.yaml` `exceptions`, in the run directory and in the uploaded `protocol.json`.
- **One training seed per model, never changed** (user 2026-11-27: the seed is uniform inside each model, kept on every task, rerun, resume and continuation; the goal is that every training and every evaluation of the benchmark is strictly reproducible).
  - GR00T 42, FastWAM 42, π0.5 42, DiT4DiT 42, Delta Flow 42, FLASH 42, ODE-MLP / ODE-DiT 42, **FM-DiT v1 and v2 42** (FM-DiT v1 from 2026-11-27; its older seed-0 folders stay as recorded), Psi-Zero 292285 (its upstream's official value); ACT and DP (the collaborator's hb_baselines code) 0.
  - Runs already trained on another seed are not retrained: they stay, with their seed recorded (REPRODUCIBILITY.md section 2 and 7).
  - The augmentation random stream is seeded from the training seed (`HB_AUG_SEED`) in every training entry; hb_baselines does it for every model from Humanoid_benchmark bai `b4b94d29` (REPRODUCIBILITY.md section 3).
  - A new model uses 42 (or its upstream's official seed when upstream fixes one) and keeps it on every training.
- **ACT trains only with the hb_baselines code** of `github.com/zuo-kuangji/Humanoid_benchmark` (`evaluation/baselines/train.py --policy act`; REPRODUCIBILITY.md section 4.5). `hb_training/hb_act/` in this repository is retired (user, 2026-11-24): its entry exits unless `HB_ALLOW_RETIRED_ACT=1` is set to reproduce the old `epoch-NNN` releases.
- **Register in-training MSE before the run starts**: `eval-action-mse/hb_eval_common.py` `TASKS` plus the `train10` cache. Do not substitute post-hoc MSE runs.
- **Upload to a new HF sub-directory.** Never overwrite a published directory, never upload resume state, and include the full config, normalization stats, logs, in-loop MSE and the code snapshot.
- **Keep the node / GPU / per-GPU-batch layout and `num_workers` recorded in the protocol.** They change data sharding and random streams.
- **When a new decision changes the protocol**, update REPRODUCIBILITY.md and protocol.yaml in the same commit.

Training-code details live in [`hb_training/README.md`](hb_training/README.md), augmentation in [`hb_training/docs/augmentation_v2.md`](hb_training/docs/augmentation_v2.md), and H2 task guides in [`hb_training/docs/h2/`](hb_training/docs/h2/).
