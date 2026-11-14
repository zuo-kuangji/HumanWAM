# H2 训练任务清单（7 个）


## 需要训练的 7 项

| # | 任务 | 模型 | 训练参考 | 要点 |
|---|---|---|---|---|
| 1 | a14 拧灯泡 | DeltaFlow | [a14_training_guide.md](a14_training_guide.md) 4.3 | v2 数据；动作 82 维（含 planner 指令） |
| 2 | a14 拧灯泡 | GR00T N1.7 | [a14_training_guide.md](a14_training_guide.md) 4.1 | v2 数据；动作 82 维（含 planner 指令）；全量微调 |
| 3 | a14 拧灯泡 | Psi0 | [a14_training_guide.md](a14_training_guide.md) 4.2 | v2 数据；动作 82 维（含 planner 指令）；a17 epoch-020 的配置 |
| 4 | a17 拧瓶盖 | DeltaFlow | [a17_training_guide.md](a17_training_guide.md) 4.3 | 把 a18 倒水的配置用到 a17 数据上；动作 75 维 |
| 5 | a17 拧瓶盖 | GR00T N1.7 | [a17_training_guide.md](a17_training_guide.md) 4.1 | 现有 a17 配方改为两路相机、batch 512；全量微调；动作 75 维 |
| 6 | a18 倒水 | GR00T N1.7 | [a18_training_guide.md](a18_training_guide.md) 4.1 | b256 配方改为两路相机、batch 512；全量微调；动作 75 维 |
| 7 | a18 倒水 | Psi0 | [a18_training_guide.md](a18_training_guide.md) 4.2 | a17 epoch-020 的配置，训到 4000 步；动作 75 维。 |




## 用到的数据集（均为私有）

| 任务 | HF 数据集 | 回合 / 帧数（30 fps） |
|---|---|---|
| a14 | `humanoid-benchmark/a14-unscrew-bulb-expert-v2` @ `8411a6da` | 100 / 185,638 |
| a17 | `humanoid-benchmark/a17-unscrew-bottle-expert-v1` @ `407e6e20` | 100 / 115,333 |
| a18 | `humanoid-benchmark/a18-pour-water-expert-v1` @ `8e956a10` | 100 / 72,752 |


