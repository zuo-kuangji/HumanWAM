# 全身 ImageWAM Δ-flow 的训练增强

本页对应main维护的motion76与joint47 v2。新训练配置生成器默认使用
[augmentation_v3.yaml](../configs/wholebody/augmentation_v3.yaml)，包含配对视觉增强、
按概率触发的当前state轻噪声，以及整份state conditioning的单次dropout。
增强只用于训练，
不修改原始数据、归一化统计、数值动作标签、历史动作或joint的root heading锚。

## 开关与旧配置

在[全身训练指南](G1_WHOLEBODY_DELTA.md)或[关节训练指南](G1_JOINT_REFERENCE_DELTA.md)
的`build_config.py`完整命令上选择：

| 选项 | 行为 |
|---|---|
| 不传增强选项 | 新builder使用默认`augmentation_v3.yaml` |
| `--no-augmentation` | 显式关闭本页额外的视觉与state增强 |
| `--augmentation-config /path/to/profile.yaml` | 使用自己的增强profile |

缺少augmentation配置的legacy数据集/训练路径保持关闭，不会自动为旧权重补上增强。
旧v1、v2的canonical配置和SHA规则保留，用于读取历史checkpoint的provenance。
显式指定v2仍可训练，保持其原有抖动范围，不会自动加入像素噪声、JPEG或v3的较强抖动。
新训练拒绝启用v1并要求使用v2或v3，不会把旧的维度屏蔽静默解释成整份条件dropout。
推理的直接policy调用和可选HTTP服务均不应用这些训练增强。
推理仍使用固定resize、对应任务的归一化与模型原有source构造。
`--no-augmentation`也不会关闭Δ-flow自身的source噪声。

## 视觉增强

每路当前/未来RGB先固定resize至224×224，在`[0,1]`空间增强，
再按ego左、chest右拼为224×448，最后映射到`[-1,1]`。
每个样本先以50%概率开启视觉增强，两台相机共用这一个总开关。
未开启时，两台相机均只做固定resize。开启后，默认每台相机独立采样下列子项与参数，
同一相机的当前/未来图像共享它们。总开关开启后仍可能没有子项触发，
所以50%表示进入增强采样的概率，不表示恰好50%的样本一定改变像素。

| 增强 | 总开关开启后的条件概率 | 每台相机的总体触发概率 | 默认幅度 |
|---|---:|---:|---|
| 曝光 | 35% | 17.5% | gain 0.85–1.25，gamma 0.85–1.15 |
| 过曝光 | 12% | 6% | gain 1.25–1.8，并限制至`[0,1]` |
| 颜色 | 70% | 35% | 对比度0.85–1.15，饱和度0.8–1.2，hue偏移最多±5° |
| 仿射抖动 | 40% | 20% | 水平/竖直平移各±4%，旋转±3°，scale 0.96–1.04 |
| 高斯模糊 | 12% | 6% | 9×9核，sigma 0.4–1.2 |
| Gaussian像素噪声 | 25% | 12.5% | std在`4/255`至`8/255`之间连续均匀采样 |
| JPEG压缩 | 25% | 12.5% | quality从50至80的整数中均匀采样，4:4:4编码 |

表中各项概率以视觉增强总开关开启为条件，不能直接当作全部训练样本的触发比例。
总体触发概率是总开关50%与子项条件概率的乘积，不等于最终可见改变的比例。
颜色在较小范围内改变对比度、饱和度和色相，避免主动改变红、绿等任务相关颜色。
配置中的hue以整圈为单位，5°对应5/360，约0.013888889；原幅度0.01对应3.6°。
纯红到纯绿的中心色相差约120°，这组±5°扰动不会执行这样的颜色转换。
曝光裁剪和模糊仍会改变可见细节，因此这些设置不构成任务识别结果不变的保证。
模糊使用9×9核容纳sigma 0.4–1.2的Gaussian。
v3的仿射幅度由v2的±2%平移、±1.5°旋转和0.98–1.02 scale，
调整为±4%、±3°和0.96–1.04；平移、旋转与scale均在各自范围内连续均匀采样。
各子项独立决定是否触发，可在一个样本中组合。
实现顺序是曝光、过曝光、颜色、仿射、模糊、像素噪声、JPEG。
仿射使用双线性采样与border边界延拓；它不跨两相机拼接缝裁剪，也不交换左右相机。
像素噪声在`[0,1]`空间加入零均值Gaussian，再限制至`[0,1]`。
同一相机clip共享noise seed、std及同一张逐像素噪声图，避免当前/未来监督各自叠加无关噪声。
JPEG对每一帧独立编码、解码，同一相机clip共用一个quality，不把当前和未来合为一幅压缩。
没有随机翻转、随机时间偏移或背景替换。

共享的是随机变换参数，不是两帧的图像内容或统计量。对比度中心按每一帧自己的灰度均值计算，
当前图像不会读取未来图像均值；仿射、模糊、像素噪声和JPEG也分别处理各帧。
配对共享的是变换参数与噪声图，不从未来图像内容生成当前图像的变换。
因此改变未来监督图像，不会反过来改变当前帧的增强结果。
动作分支仍只接收当前图像、文本和原有噪声视觉上下文，不获得真实未来图像。
增强操作在缓存副本上进行，不能原地污染解码/LRU缓存。

## 当前state的概率噪声

state44仍按`body29 + measured Inspire12 + projected_gravity3`排列，
只进入动作专家。每个样本以25%概率触发一次state噪声；未触发的样本保持干净。
噪声在本任务统计归一化后添加，所有标准差均为**归一化空间单位**。

| 分组 | 标准差 | bias标准差 | 默认处理 |
|---|---:|---:|---|
| body29，索引`0:29` | 0.01 | 0 | 加Gaussian，保留原归一化`[-5,5]`范围 |
| Inspire12，索引`29:41` | 0.005 | 0 | 加Gaussian，限制至`[-1,1]` |
| gravity3，索引`41:44` | 0 | — | 不加数值噪声 |

这些数值不能解释为0.01 rad或0.005的原始手状态噪声。
手部从物理`[0,1]`映射到`[-1,1]`，因此0.005归一化标准差对应0.0025物理开合量。
body的物理幅度由该任务各关节的归一化范围决定。

每个样本只对当前state采一次噪声，再沿proprio时间轴重复。
FM与IC使用同一份增强观测，不为16/40个重复位置独立制造噪声。
动作监督、历史source、原始测量、手部标签及joint root坐标锚不被改写。

## 整份state conditioning的单次dropout

每个训练样本另以10%概率关闭其state条件。操作发生在投影之后：

```text
state_features = state_encoder(current_state)
state_features = state_features * keep_sample
keep_sample = 0 with probability 0.1, otherwise 1
```

被选中的样本，其整份state特征都变成0，包括投影层的bias贡献。
这不是随机屏蔽1–3个输入维度，也不是把真实state44改成0后仍保留encoder bias。
保持样本不做`1/(1-p)`放大。每个样本只采一次gate，FM与IC以及重复使用当前state的动作计算共用它。
state噪声和条件dropout各自按概率触发；不丢动作历史、不改动作标签，也不改变root heading锚。

gravity原始数值不加噪；整份条件关闭时，它通过state encoder提供的条件贡献也随之关闭。
推理始终保留state条件，不执行这组训练gate。此处使用显式条件gate，
没有添加`nn.Dropout`层，正式训练对模型内部`nn.Dropout(p>0)`的限制不变。

这一位置参考了固定版本GR00T N1.7的[action-head state-feature dropout](https://github.com/NVIDIA/Isaac-GR00T/blob/7aa0bca50f1097f8b17f5c274c98264b72a1419f/gr00t/model/gr00t_n1d7/gr00t_n1d7.py#L249-L263)，
其[微调配置默认概率为0.2](https://github.com/NVIDIA/Isaac-GR00T/blob/7aa0bca50f1097f8b17f5c274c98264b72a1419f/gr00t/configs/finetune_config.py#L61)，
[base模型配置则为0.8](https://github.com/NVIDIA/Isaac-GR00T/blob/7aa0bca50f1097f8b17f5c274c98264b72a1419f/gr00t/configs/model/gr00t_n1d7.py#L128)。
官方[processor还包含另一处独立Bernoulli gate](https://github.com/NVIDIA/Isaac-GR00T/blob/7aa0bca50f1097f8b17f5c274c98264b72a1419f/gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py#L678-L694)。
HumanWAM采用一次10%的特征条件gate；官方微调默认20%仅作为参考，
这里不宣称复刻官方双重gate或其整体有效dropout率。

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
概率state噪声、整份state条件dropout和动作source噪声是不同操作，不得共用被修改的标签张量。

训练还会分别采样视觉/动作FM时间；推理不采训练时间，但仍按模型配置采source噪声。
这与图像增强、物理控制频率和state条件dropout均不同。

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

检查应覆盖当前/未来配对参数与噪声图、JPEG的逐帧处理、未来内容不影响当前增强、缓存不变、
state噪声概率及重复proprio一致、整份encoder特征含bias归零、FM/IC共享gate，
以及旧v1/v2 provenance、v2继续训练和推理关闭增强。
这些软件检查不代表上述参数已改善任务成功率，也不代表完成了新GPU训练或真机效果验证。
