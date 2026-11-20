# 早期 Delta Flow 工作树里没有提交过的改动（2026-11-04/05）

A800 集群 NAS 上 4 个早期 HumanWAM 工作树都在分支 `wholebody-deltaflow-training-20261027` 的 9cb38e1 上，各自有未提交的修改
（`configs/wholebody/deltaflow76_p25.yaml`、`scripts/wholebody/build_config.py`、`scripts/wholebody/train.py`、
`src/imagewam/datasets/wholebody.py`、`src/imagewam/wholebody_delta.py`）。这里原样保存为补丁，内容相同的只存一份；
在 9cb38e1 上 `git apply <patch>` 即可还原。它们的未跟踪新文件（augmented_train.py、continuation_checks.py、
wholebody_augmented.py）已在 `hb_training/hb_deltaflow/code/g1/` 里。正式 benchmark 训练用的是 `hb_deltaflow/code/` 下的快照。

| 工作树 | NAS 路径（相对 preflight-20261104T1648Z） | 补丁 |
|---|---|---|
| HumanWAM | HumanWAM | HumanWAM.patch |
| all60-20261105 | all60-20261105/HumanWAM | all60-20261105.patch |
| augmentation-review-20261105 | augmentation-review-20261105/HumanWAM | augmentation-review-20261105.patch |
| continuation-20261105-30d | continuation-20261105-30d/HumanWAM | augmentation-review-20261105.patch |
