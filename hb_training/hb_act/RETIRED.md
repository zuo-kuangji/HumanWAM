# hb_act is retired (user decision, 2026-11-24)

本目录（Psi0 仓库内的 LeRobot ACT）已退役，本项目不再用它训练任何 ACT。

- Every ACT training in the Humanoid Benchmark project (new tasks, re-training, continuation) uses the collaborator's
  hb_baselines code: `github.com/zuo-kuangji/Humanoid_benchmark`,
  `evaluation/baselines/train.py --policy act --contract <contract>`, augmentation
  `evaluation/augmentation/hb_aug_v2.py`, environment `evaluation/baselines/requirements.txt`.
- Recipe and seed (0): `hb_training/REPRODUCIBILITY.md` section 4.5 and `hb_training/protocol.yaml` `models.hb_baselines`.
- This directory (seed 292285) is kept only to document and reproduce the old
  `humanoid-benchmark/act/<task>/epoch-NNN` releases. `hb_act_train.py` exits unless `HB_ALLOW_RETIRED_ACT=1` is set.
