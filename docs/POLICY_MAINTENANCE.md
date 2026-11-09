# HumanWAM policy 维护

按6个独立模型家族、7个注册入口维护。Gaussian、A2A及任务权重是家族内配置；
ImageWAM的motion76与joint-reference因动作语义和checkpoint不同，分别注册。

| Backend | 原生源码 | 训练和推理入口 | 当前边界 |
|---|---|---|---|
| `psi0` | `third_party/Psi0`及其PSI SONIC fork | 原生训练、RTC WebSocket与离线chunk | checkpoint拥有归一化和horizon，不自动当作feedback76 |
| `groot_n17` | `third_party/Isaac-GR00T` + `GR00T-WholeBodyControl` | 原生finetune + ZeroMQ PolicyServer | 原生SONIC78；独立NEW_EMBODIMENT配方可训feedback76/H16 |
| `pi05` | `third_party/openpi` + 版本锁定A2A patch | OpenPI train + WebSocket服务 | 当前适配LIBERO7/H10，G1/76D需要单独模型架构与训练适配 |
| `imagewam_delta` | `src/imagewam` + `scripts/wholebody` | 原生FM+IC训练 + 固定release/nativewholebody推理 | 新H16、旧H40显式选择，state不进视觉 |
| `imagewam_joint_reference` | `src/imagewam` + 独立joint-reference配方 | 新joint41/47训练与离线参考导出 | G1 body29＋Inspire12，可预测root6D；不是motion64，不兼容旧76完整checkpoint |
| `fastwam` | `third_party/FastWAM` | 原生Gaussian FM训练 + 原生infer_action及通用HTTP入口 | 官方LIBERO7/RoboTwin14、H32；未添加A2A/IC，未做GPU推理 |
| `dit4dit` | `third_party/DiT4DiT` | 原生Cosmos train + WebSocket服务 | native单当前摄像头和checkpoint动作空间，不把第二相机当未来帧 |

## 目录

```text
packages/humanwam/       轻量registry/CLI、适配器、CPU接口测试
configs/humanwam/        统一启动配置、显式动作schema
src/imagewam/           原有ImageWAM核心及全身Delta实现
scripts/wholebody/      ImageWAM数据缓存、配置生成和训练
scripts/humanwam/       源码准备、离线joint-reference文件导出
third_party/           固定gitlink的独立上游项目
patches/               OpenPI A2A与DiT4DiT的最小补丁
backends.lock.json      上游URL、精确commit和patch SHA256
docs/policies/          每类policy的原生依赖、数据、部署约定
```

初始化权重与恢复方式见[模型初始化说明](MODEL_INITIALIZATION.md)，实际检查边界见[运行验证](policies/RUNTIME_VALIDATION_20261109.md)。
新入口的具体边界见[Fast-WAM](policies/FASTWAM.md)、[G1关节参考训练](G1_JOINT_REFERENCE_DELTA.md)
和[SONIC离线参考桥接](policies/SONIC_JOINT_REFERENCE.md)。新增joint路径尚无已训练模型或真机验证。
47D姿态目标还需按数据审计处理录制器fallback，不能将所有保存的单位四元数都当作有效未来姿态；具体监督有效性规则以关节训练文档为准。

## 安装和检查

```bash
git clone --branch main https://github.com/zuo-kuangji/HumanWAM.git
cd HumanWAM
export HUMANWAM_ROOT="$PWD"
python3 scripts/humanwam/bootstrap.py
python3 -m venv .venv-tools
.venv-tools/bin/pip install -e 'packages/humanwam[test]'
.venv-tools/bin/humanwam list
.venv-tools/bin/humanwam doctor
.venv-tools/bin/python -m pytest packages/humanwam/tests
```

bootstrap只clone版本锁定的代码并应用明确补丁，不下载权重，不启动训练或机器人。
第二次执行会识别已应用补丁。doctor检查commit、patch和额外本地改动，不能替代GPU/真机测试。
如果安装wheel脱离checkout，必须设置`HUMANWAM_ROOT`。

## 独立环境

轻量CLI环境只用来查看和生成命令。每个模型按上游各建自己的环境，
在该环境也执行`pip install -e /path/to/HumanWAM/packages/humanwam`。
GR00T、OpenPI、ImageWAM、Fast-WAM、DiT4DiT与Psi0对Torch、JAX、transformers等版本要求不同。
不要把各套依赖强塞进同一个requirements，也不要覆盖服务器现有训练环境。

配置顶层`python`指定对应解释器。原生GR00T训练shell使用其PATH和项目环境，
请先激活GR00T环境。所有`source_dir`相对路径以HumanWAM根目录解析；
权重、数据、输出路径建议使用环境变量对应的绝对路径。
未设置的`${VARIABLE}`会报错，不会误用另一个任务的目录。

```bash
humanwam train --config configs/humanwam/pi05-train.json --dry-run
humanwam serve --config configs/humanwam/groot-sonic.json --dry-run
humanwam audit-data /path/to/lerobot-dataset
```

去掉`--dry-run`才运行原生入口。数据audit仅检查metadata的计数/时长，
不代表视频全量解码或数值映射已验证；开训前运行对应backend的数据检查。
本次整合不修改现有服务器实验，也不自动上传模型和数据。

## 推理与协议

七个注册入口都提供`build_policy(config)`、`predict(observation)`、`reset()`，
但输入保持各自的原生字段、归一化与动作含义。输出附带或保留原生动作空间。
不能把GR00T78D command hand、ImageWAM motion76 measured hand、joint41/47和LIBERO7D控制量混用。
joint路径必须保留`body_target_source`，区分WBC电机qtarget、实测关节与原始运动参考。

CLI离线预测使用NPZ数组和可选JSON metadata。GR00T数组可使用`video.<camera>`、
`state.<key>`形式，加载时恢复原生嵌套结构；语言放在JSON的`language`中。
OpenPI的`observation/image`等带斜杠键保留原样。禁止pickle/object数组。

原生GR00T服务继续使用SONIC支持的ZeroMQ/msgpack接口；
OpenPI、DiT4DiT使用各自WebSocket服务，Psi0保留原生RTC WebSocket；
ImageWAM与Fast-WAM提供通用JSON/NPZ HTTP预测入口。
统一的是配置和生命周期，不自动把不同控制器的输出接到机器人。

joint-reference导出只生成50Hz NPZ及可选SONIC v1 body bytes，不发布到网络。
SONIC v1.1模式0要求offset0..45、至少46帧，16步连续30Hz不够；
新H16 stride2目标可覆盖1秒，但不能把旧连续预测的timestamp乘2充当新训练。
未来root姿态缺失时拒绝，不复制当前IMU或单位旋转，Inspire12独立于官方Dex3手字段。

## 已有权重

HF模型仓库、checkpoint、归一化统计、配置和推理代码组成一个版本单元。
原版GR00T HF目录与我们旧的完整FM/ODE-head `.pt`导出不是同一格式；后者继续使用原来已发布的runtime。
π0.5 LIBERO JAX A2A与R1Lite Torch14D/7D权重也分别保留对应runtime。
本次没有修改旧权重维度、horizon或格式。
新joint41/47配方需要新建动作投影与checkpoint统计，复用FLUX.2基础派生114个共享张量
不等于加载旧76D完整任务权重。Fast-WAM发布`.pt`与其VAE/T5/tokenizer也需分别准备。
