# 全身 ImageWAM Δ-flow 的训练增强

本页对应main维护的motion76与joint47 v2。新训练配置生成器默认使用
[augmentation_v1.yaml](../configs/wholebody/augmentation_v1.yaml)，包含配对视觉增强、
当前state轻噪声，以及偶发的少量state维度屏蔽。增强只用于训练输入，
不修改原始数据、归一化统计、数值动作标签、历史动作或joint的root heading锚。

## 开关与旧配置

在[全身训练指南](G1_WHOLEBODY_DELTA.md)或[关节训练指南](G1_JOINT_REFERENCE_DELTA.md)
的`build_config.py`完整命令上选择：

| 选项 | 行为 |
|---|---|
| 不传增强选项 | 新builder使用默认`augmentation_v1.yaml` |
| `--no-augmentation` | 显式关闭本页额外的视觉与state增强 |
| `--augmentation-config /path/to/profile.yaml` | 使用自己的增强profile |

缺少augmentation配置的legacy数据集/训练路径保持关闭，不会自动为旧权重补上增强。
推理的直接policy调用和可选HTTP服务均不应用这些训练增强。
推理仍使用固定resize、对应任务的归一化与模型原有source构造。
`--no-augmentation`也不会关闭Δ-flow自身的source噪声。

## 视觉增强

每路当前/未来RGB先固定resize至224×224，在`[0,1]`空间增强，
再按ego左、chest右拼为224×448，最后映射到`[-1,1]`。
同一相机的当前/未来图像共享一次采样得到的参数；默认两台相机各自采样。

| 增强 | 触发概率 | 默认幅度 |
|---|---:|---|
| 曝光 | 35% | gain 0.85–1.25，gamma 0.85–1.15 |
| 过曝光 | 12% | gain 1.25–1.8，并限制至`[0,1]` |
| 颜色 | 70% | 对比度0.9–1.1，饱和度0.85–1.15，hue偏移±0.01 |
| 小幅仿射抖动 | 40% | 水平/竖直平移各±2%，旋转±1.5°，scale 0.98–1.02 |
| 高斯模糊 | 12% | 5×5核，sigma 0.2–0.8 |

各项独立决定是否触发，可在一个样本中组合。实现顺序是曝光、过曝光、颜色、仿射、模糊。
仿射使用双线性采样与border边界延拓；它不跨两相机拼接缝裁剪，也不交换左右相机。
没有随机翻转、随机时间偏移、背景替换或独立RGB像素Gaussian噪声。

共享的是随机变换参数，不是两帧的图像内容或统计量。对比度中心按每一帧自己的灰度均值计算，
当前图像不会读取未来图像均值；仿射和模糊也逐帧应用相同参数。
因此改变未来监督图像，不会反过来改变当前帧的增强结果。
动作分支仍只接收当前图像、文本和原有噪声视觉上下文，不获得真实未来图像。
增强操作在缓存副本上进行，不能原地污染解码/LRU缓存。

## 当前state的轻噪声与少量维度屏蔽

state44仍按`body29 + measured Inspire12 + projected_gravity3`排列，
只进入动作专家。增强在本任务统计归一化后进行，所有标准差均为**归一化空间单位**。

| 分组 | 标准差 | bias标准差 | 默认处理 |
|---|---:|---:|---|
| body29，索引`0:29` | 0.01 | 0 | 加Gaussian，保留原归一化`[-5,5]`范围 |
| Inspire12，索引`29:41` | 0.005 | 0 | 加Gaussian，限制至`[-1,1]` |
| gravity3，索引`41:44` | 0 | — | 不加噪、不屏蔽 |

这些数值不能解释为0.01 rad或0.005的原始手状态噪声。
手部从物理`[0,1]`映射到`[-1,1]`，因此0.005归一化标准差对应0.0025物理开合量。
body的物理幅度由该任务各关节的归一化范围决定。

另有10%的样本触发state维度屏蔽。触发后，在body＋hands的41维中，
每维以5%概率进入候选；候选为空时补选1维，超过3维时随机保留3维。
因此一次只屏蔽1–3维，不会置空整条state。屏蔽值是归一化0，
不做inverted-dropout幅度放大，也不代表原始关节角为0。
自定义profile中将`probability`或`dimension_probability`设为0，可关闭这项屏蔽。

每个样本只增强一次当前state，再沿proprio时间轴重复。
同一样本的FM与IC使用相同的增强观测，不为40个重复位置独立制造噪声。
真实未来state仍不是模型输入；原始state、动作监督、历史source值、手部标签和root锚保持不变。
这里的输入维度屏蔽不是`nn.Dropout`，正式训练对模型内部`nn.Dropout(p>0)`的限制不变。

## 与Δ-flow source噪声区分

以下是方法本身的source定义，独立于上述训练增强：

```text
visual source = 0.5 * z_current + sqrt(0.75) * epsilon
action source = normalized_past_actions + 0.5 * epsilon
```

视觉“25%”表示当前latent系数平方为0.25，不是屏蔽25%的state或图像。
视觉source由本次增强后的当前图像编码得到，再按公式加噪；未来视觉目标使用同参数增强的未来图像。
Motion76的76维、joint47的47维动作source均加0.5 Gaussian，包括手部与joint root旋转。
source不额外做方差归一化，不在加噪后裁剪；数值动作监督保持干净。

Joint47先用16个历史点构造“原16点＋最后历史点重复24次”的40点source先验，再加噪。
这不修改历史记录，不生成伪造的未来监督，也不替代模型在部署时预测的24点lookahead。
state噪声、state屏蔽和这组动作source噪声是不同操作，不得共用一份被修改的标签张量。

训练还会分别采样视觉/动作FM时间；推理不采训练时间，但仍按模型配置采source噪声。
这与图像增强、物理控制频率和state维度屏蔽均不同。

## 实现与验证边界

- [增强profile与无Torch配置校验](../src/imagewam/augmentation_config.py)
- [配对视觉增强](../src/imagewam/datasets/wholebody_augmentation.py)
- [当前state增强、worker/epoch随机数管理](../src/imagewam/datasets/proprio_augmentation.py)
- [Motion76数据集与固定resize](../src/imagewam/datasets/wholebody.py)
- [Joint47数据集](../src/imagewam/datasets/wholebody_joint_reference.py)
- [Motion76模板](../configs/wholebody/deltaflow76_p25.yaml)
- [Joint47 v2模板](../configs/wholebody/deltaflow_joint_reference_v2_p25.yaml)
- [配置生成与训练检查](../scripts/wholebody/build_config.py)
- [训练协议和dropout检查](../scripts/wholebody/train.py)
- [动作/视觉source构造](../src/imagewam/models/backbones/imagewam.py)
- [FM连续时间scheduler](../src/imagewam/models/backbones/schedulers/scheduler_continuous.py)

检查应覆盖当前/未来配对参数、未来内容不影响当前增强、缓存不变、state只影响条件输入、
重复proprio一致、少量维度屏蔽范围，以及legacy和推理关闭增强。
这些软件检查不代表上述参数已改善任务成功率，也不代表完成了新GPU训练或真机效果验证。
