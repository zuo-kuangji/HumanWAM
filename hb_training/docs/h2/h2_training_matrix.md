# H2 training matrix: 3 tasks x 3 models

Written 2026-11-14 for the colleagues who train the H2 benchmark models. Goal **(owner)**: each of DeltaFlow, GR00T
N1.7 and Psi0 reaches a non-zero closed-loop success rate (at least one successful demo) on each of the three H2
tasks. This page says which of the nine cells still need training. The data, the action contract, how the evaluator
executes it and the exact recipe of every cell are in the three task guides:

- [a14 training guide](a14_training_guide.md): screw the bulb (walk, pick, insert, screw)
- [a17 training guide](a17_training_guide.md): unscrew the bottle cap
- [a18 training guide](a18_training_guide.md): pour water

## 1. The matrix

| Task | DeltaFlow | GR00T N1.7 | Psi0 |
|---|---|---|---|
| **a14** screw the bulb | **Train** on the a14 v2 data ([a14 guide](a14_training_guide.md) 4.3) | **Train** on the a14 v2 data ([a14 guide](a14_training_guide.md) 4.1) | **Train** on the a14 v2 data ([a14 guide](a14_training_guide.md) 4.2) |
| **a17** unscrew the bottle cap | **Train** ([a17 guide](a17_training_guide.md) 4.3) | **Train** ([a17 guide](a17_training_guide.md) 4.1) | **Done**: `epoch-020`, **5/50** random. No training ([a17 guide](a17_training_guide.md) 4.2) |
| **a18** pour water | **Done**: legacy `epoch-040`, **31/100**. No training ([a18 guide](a18_training_guide.md) 4.3) | **Train** ([a18 guide](a18_training_guide.md) 4.1) | **Continue training to 4000 steps** ([a18 guide](a18_training_guide.md) 4.2) |

Six cells to train, three done. Common protocol **(owner)** for every cell that trains:

| Item | Value |
|---|---|
| Global batch | **512** |
| Steps | **4000** (publish the checkpoints at 2000 and 4000) |
| Cameras | **head + chest** only (`observation.images.head`, `observation.images.chest`); no wrist, no third_person |
| State | 78 = `observation.state` (75) + `observation.projected_gravity` (3) |
| Action | a14: **82** = `action.body_target` (31) + `action.hands` (44) + `action.planner` (7, the walking command executed by the SONIC planner); a17 and a18: **75** = body 31 + hands 44 |
| GR00T | full fine-tune (all modules) |
| Psi0 | the `a17-unscrew-bottle/epoch-020` configuration |
| DeltaFlow | the `legacy/a18-pour-water/from-scratch-pre-v2-aug-20261108/epoch-040` configuration |
| HF path | a14: `humanoid-benchmark/<model>/a14-unscrew-bulb/v2-b512/step-00{2,4}000`; a17 / a18: `humanoid-benchmark/<model>/<task folder>/b512-2cam/step-00{2,4}000` (`b512-2cam` keeps them apart from the existing 3-camera `b256` / `b512` folders) |

## 2. Data per task

| | a14 | a17 | a18 |
|---|---|---|---|
| HF dataset (private) | `a14-unscrew-bulb-expert-v2` @ `8411a6da` | `a17-unscrew-bottle-expert-v1` @ `407e6e20` | `a18-pour-water-expert-v1` @ `8e956a10` |
| Episodes / frames (30 fps) | 100 / 185,638 | 100 / 115,333 | 100 / 72,752 |
| Epochs at 4000 steps x 512 | about 11 | about 18 | about 28 |
| Action | 82 (with `action.planner`) | 75 | 75 |
| Robot | walks: the planner command, executed by the planner + SONIC | stands (torso support) | stands (torso support) |

The prompts are the dataset task strings; each guide quotes its own.

## 3. Per-cell notes

**a14, all three models: train.** The v1 a14 checkpoints (GR00T and Psi0 `a14-unscrew-bulb/step-004000`, 3 cameras,
joint targets only) scored 0 %: they cannot walk. DeltaFlow has no a14 checkpoint. The v2 data adds the planner
command; with the walking given, the v1 checkpoints still failed the grasp, so precision matters (a14 guide section
6).

**a17 DeltaFlow: train.** No a17 checkpoint exists. The a18 pour configuration on the a17 data (a17 guide 4.3).

**a17 GR00T: train.** Existing: `epoch-020` (2 cameras, 0/50) and `b256/step-004000` (3 cameras, 0/50; 49 of 50
episodes never move). The current a17 recipe with two cameras and batch 512 (a17 guide 4.1).

**a17 Psi0: done.** `epoch-020` (2 cameras, global batch 512, 4506 steps) succeeded in 5 of 50 random layouts and stays
the a17 Psi0 checkpoint; no new run (a17 guide 4.2). Its configuration is the Psi0 recipe of the other cells.

**a18 DeltaFlow: done.** `epoch-040` succeeded in 31 of 100 episodes (15/50 fixed, 16/50 random; the "put back within
2 cm" rule not required). A legacy run (before augmentation spec v2, global batch 256, 40 epochs) that the owner counts
as done; its configuration is the DeltaFlow recipe of the other cells (a18 guide 4.3).

**a18 GR00T: train.** Existing: `epoch-020` (2 cameras, 0/100) and `b256/step-004000` (3 cameras, 0/6: reaches the
handle, then freezes without closing the hand). The b256 recipe with two cameras and batch 512 (a18 guide 4.1).

**a18 Psi0: continue training to 4000 steps.** The 2-camera `epoch-020` is undertrained rather than wrong: global batch
1280 made its 20 epochs only 1137 optimizer steps (final loss 1.16, about a17 `epoch-020`'s 1.12 at the same step;
a17 went on to 4506 steps, 0.47 and 5/50). The a18 guide 4.2 trains the a18 head + chest model on the a17 `epoch-020`
configuration to step 4000, starting from the same Psi0 initialization as a17 for comparability; warm-starting from
the a18 `epoch-020` weights is allowed if recorded in `protocol.json`. For reference, the 3-camera
`b512/step-004000` scored 0/5 fixed and 3/10 random in our evaluation (2026-11-14).

## 4. How the results are measured

The evaluation (repository side, `debug/newton/h2_sharpa_policy_eval.py`) renders the checkpoint's cameras live in
simulation, feeds state78, and executes 16 action rows per query at 30 Hz. Each guide's section 2.2 (a14: 2.3)
describes the execution of the action columns.

| Task | Episodes | Budget | Success |
|---|---|---|---|
| a14 | random layouts (seed 1000 + episode) | 93 s | lamp lit, peak turns >= 0.30, robot standing, table / socket / power strip moved < 5 mm |
| a17 | 50 random layouts | 60 s | cap released, >= 2 cm above its unscrewed seat with >= 2 right digits on it, bottle in the left fist off the table, robot standing, table moved < 5 mm |
| a18 | 50 fixed + 50 random layouts | 36 s | cup >= 150 mL, spill <= 10 mL, pitcher upright and released, robot standing, props unmoved |

Per-task debugging and results: [a14](debug_screw_bulb_rollout.md), [a17](debug_unscrew_bottle_rollout.md),
[a18](debug_pour_water_rollout.md).
