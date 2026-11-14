# H2 机器人任务的训练规范（2026-11-14）

这里存放评估方 2026-11-14 交给训练方的四份文档，原文未改。H2 任务（A14 拧灯泡、A17 拧瓶盖、A18 倒水）按这里的规定训练，和 G1 任务的默认训练方式不同。

| 文件 | 内容 |
|---|---|
| [h2_training_matrix.md](h2_training_matrix.md) | 3 个任务 × 3 个模型（DeltaFlow、GR00T N1.7、Psi0）哪些要训、哪些已完成，以及共同规定 |
| [a14_training_guide.md](a14_training_guide.md) | A14：v2 数据（加了 \`action.planner\` 行走指令），动作 82 维 |
| [a17_training_guide.md](a17_training_guide.md) | A17：GR00T、DeltaFlow 重训，Psi0 不重训 |
| [a18_training_guide.md](a18_training_guide.md) | A18：GR00T 重训，Psi0 训到 4000 步，DeltaFlow 不重训 |

- 文中链接的 \`debug_*_rollout.md\`、\`debug/newton/...\`、\`debug/outputs/...\` 在评估方的仿真仓库里，不在本仓库。
- 数据版本（2026-11-14 核对 HF）：A14 用 \`a14-unscrew-bulb-expert-v2\` @ \`8411a6da\`（11-14 新上传）；A17 \`a17-unscrew-bottle-expert-v1\` @ \`407e6e20\`、A18 \`a18-pour-water-expert-v1\` @ \`8e956a10\` 没有更新。

