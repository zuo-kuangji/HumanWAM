# GR00T ODE Policy / G1

`groot_ode` 的 ODE 实现直接维护在本仓库的 `packages/ode_g1`，可通过 `ode_g1` 模块导入。
代码最初来自 Jindou Jia 等人的 [ODE Policy](https://github.com/NTUMARS/ODE-POLICY)
公开 `real-g1` 组件，来源与归属记录在 [UPSTREAM.md](../../packages/ode_g1/UPSTREAM.md)。
它替换 GR00T N1.7 的动作分支，保留冻结视觉/语言骨干和可训练 context reader。
身体64维使用 FiLM 速度场，以 Euler 每个物理控制步积分一次，预测 H16；双手12维由独立连续解码器输出。
这里的积分时间是机器人控制步，不是 flow-matching 去噪时间。

## 本地源码与运行环境

| 组件 | 路径 | 版本关系 |
|---|---|---|
| HumanWAM ODE 实现 | `packages/ode_g1` | 随 HumanWAM 提交维护；首次引入来源为 ODE Policy `fd8bec470faf7793702533358867db02819e8f27` |
| ODE 使用的 Isaac-GR00T 框架 | `third_party/Isaac-GR00T-ODE` | 外部框架依赖，固定 `23ace64f17aa5015259b8609d371eb61a357c776` |

ODE 的数据准备、特征缓存、模型头、训练、推理和原生部署代码均在本地包中。
修改这些代码直接提交到 HumanWAM。ODE Policy 的原始 commit 用于出处记录和引用，
运行时不需要克隆 ODE-POLICY 仓库，也不要求本地包保持为外部 Git 子模块。

ODE 的 GR00T 源码与原 `groot_n17` 的 `7aa0bca...` 不同，使用独立环境。
固定 ODE runtime 的平台 pyproject 要求 Python3.10、Torch2.7.1、torchvision0.22.1、
Transformers4.57.3 和 NumPy1.26.4。按该 commit 的上游说明准备自己的 CUDA、FlashAttention、
FFmpeg/TorchCodec 环境，不把标准 GR00T 的 Python3.12 环境覆盖成这个版本。

```bash
cd "$HUMANWAM_ROOT"
python3 scripts/humanwam/bootstrap.py --only groot_ode
# 先在独立 Python3.10 环境按 Isaac-GR00T-ODE 的说明完成平台依赖安装。
"$GROOT_ODE_PYTHON" -m pip install --no-deps -e third_party/Isaac-GR00T-ODE
"$GROOT_ODE_PYTHON" -m pip install --no-deps -e './packages/ode_g1[data,vision]'
"$GROOT_ODE_PYTHON" -m pip install -e './packages/humanwam[sonic]'
```

`--no-deps` 假定平台环境已准备上游要求的依赖，并不会安装缺少的包。
源码准备不下载模型。所有启动器默认离线，并拒绝 `base_model.allow_download=true`。
需要提前准备以下本地资产，其 revision 与代码 commit 是不同概念：

| 本地资产 | 原生固定 revision |
|---|---|
| `nvidia/GR00T-N1.7-3B` | `2fc962b973bccdd5d8ce4f67cc63b264d6886495` |
| Cosmos backbone/processor | `9ce19a195e423419c349abfc86fd07178b230561` |

## 数据、特征与训练

启动配置分成 HumanWAM 的 `configs/humanwam/groot-ode.json` 和原生
`configs/humanwam/groot-ode.yaml`。先设置自己的本地位置：

```bash
export HUMANWAM_ROOT=/absolute/path/to/HumanWAM
export GROOT_ODE_PYTHON=/absolute/path/to/ode-env/bin/python
export GROOT_ODE_DATA_ROOT=/data/original-g1-task
export GROOT_ODE_RUN_ROOT=/runs/new-groot-ode-task
export GROOT_ODE_BASE_MODEL=/models/GR00T-N1.7-3B
export GROOT_ODE_COSMOS=/models/Cosmos-Reason2-2B
```

数据使用单任务 LeRobot v2.1 原始记录，两个配置相机为 ego/chest。
原始 state43 投影为 body29，追加左右实测 Inspire6+6 和 gravity3 得到 state44。
动作标签为同一行的 motion64＋左右实测反馈6+6，手部顺序为
`pinky, ring, middle, index, thumb_bend, thumb_rotation`，范围 `[0,1]`。
不能直接使用 stock GR00T 每手7维命令或 FastWAM 的 compact state44 数据副本替代原始记录。

```bash
"$GROOT_ODE_PYTHON" -m humanwam.backends.groot_ode_commands inspect \
  --config configs/humanwam/groot-ode.json
"$GROOT_ODE_PYTHON" -m humanwam.backends.groot_ode_commands prepare \
  --config configs/humanwam/groot-ode.json --dry-run
"$GROOT_ODE_PYTHON" -m humanwam.backends.groot_ode_commands prepare \
  --config configs/humanwam/groot-ode.json
"$GROOT_ODE_PYTHON" -m humanwam.backends.groot_ode_commands features \
  --config configs/humanwam/groot-ode.json
humanwam train --config configs/humanwam/groot-ode.json --dry-run
humanwam train --config configs/humanwam/groot-ode.json
```

这些是分阶段执行命令；本次集成没有自动执行它们。
`prepare` 写入独立目录，计算任务 q01/q99 统计并记录数据身份；原数据保持原样。
`features` 用固定视觉/语言处理流程产生 FP16 缓存，训练默认要求完整命中后将冻结 VLM 移到 CPU。
原生训练是单 optimizer 进程，默认90 epochs、batch112，没有多卡 DDP 实现。
配置中的预算是上游默认值，应按任务和显存调整；不要套用其他 GR00T 8卡启动命令。

原生训练输出完整 bundle，包括 `ode_config.json`、`normalization.json`、
`initial_pose.json`、`final_head.safetensors` 和 `policy_manifest.json`。
Manifest 绑定 head、统计、架构、base 权重和相机顺序；适配器保留完整哈希和严格 key/shape 检查。
历史实验的 `final_head.safetensors` 或 `.pt` 不能仅凭76维/H16就作为当前 bundle 使用，
需要根据当时源码、统计及 base 身份单独核对和迁移。本次没有改写已有实验权重或补造 manifest。

## 高层预测接口

```bash
humanwam serve --config configs/humanwam/groot-ode.json --dry-run
humanwam serve --config configs/humanwam/groot-ode.json
```

默认端口5558，与标准 GR00T5555、FastWAM5555 分开。服务采用 GR00T PolicyClient 的
ZeroMQ/msgpack 协议，针对固定旧版客户端发送 NPY 数组封装及旧版模态标记；
原有新版本服务仍保留 msgpack-numpy 格式。`ping`、`get_modality_config` 和配置检查不加载模型。
首次有效预测才通过本地 `ode_g1` 实现加载模型及固定版本的 GR00T 框架。输入布局与自有 GR00T feedback76 一致：

| 输入 | 契约 |
|---|---|
| `video.ego_view`、`video.chest` | 两路当前 uint8 RGB `[1,1,H,W,3]`，不在调用端 resize/拼接 |
| `state` | left_leg6、right_leg6、waist3、left_arm7、right_arm7、left_hand6、right_hand6、projected_gravity3，各为原值 float32 `[1,1,D]` |
| `language` | `{"annotation.human.task_description": [["任务指令"]]}` |
| `options` | 每次明确提供 `at_episode_start=true` 或最后实际发出的 `previous_body64[64]`，二者互斥 |

输出动作字段是 float32 `motion_token[1,16,64]`、`left_hand_joints[1,16,6]` 和
`right_hand_joints[1,16,6]`，已经反归一化；`predict` 另提供 dense `action[16,76]`。
相机预处理和聊天模板使用本地 `ode_g1.runtime.RGBEncoder` 与匹配的 GR00T 处理器。
训练和推理也可直接引用本仓库实现：

```python
from ode_g1.loader import load_policy
from ode_g1.model import SonicODEConfig, install_sonic_ode
from ode_g1.runtime import ODEPolicy, RGBEncoder
```

```python
from gr00t.policy.server_client import PolicyClient

policy = PolicyClient(host="127.0.0.1", port=5558)
actions, info = policy.get_action(observation, options={"at_episode_start": True})
# 执行端实际提交一部分动作后，使用最后一次实际提交的物理 token：
actions, info = policy.get_action(
    fresh_observation,
    options={"previous_body64": last_actually_sent_body64},
)
```

ODE 的 carry 是控制语义的一部分。服务只预览预测，不推进执行历史；不能把尚未执行的
整段预测末点当成下一段起点，不能把实测关节位置当作 body latent，也不能让丢弃/过期预测改变 carry。
`at_episode_start` 使用学习坐标中的 normalized-zero BOS，它不是要发给机器人的物理零命令。
每次都要重新声明起点，`reset` 不允许后续请求省略 carry，也不重置机器人或 SONIC 会话。
因此 stock 客户端虽然能解析输出，现有执行桥仍须传入实际发出前缀的末 token。

离线 `humanwam predict` 可读取 `video.*`、`state.*` NPZ 和 JSON language；
`at_episode_start` 放 JSON metadata，续段的 `previous_body64` 可放 NPZ。
本地 `ode-g1` CLI 另支持 cached feature NPZ，使用原生格式的 state44、当前 backbone_features/masks：

```bash
"$GROOT_ODE_PYTHON" -m humanwam.backends.groot_ode_commands infer \
  --config configs/humanwam/groot-ode.json --input /data/cached_observation.npz \
  --output /data/new_prediction.npz --episode-start --dry-run
```

去掉 `--dry-run` 才运行原生离线推理。该路径不连接执行器，续段时输入必须包含 `previous_body64`。

## SONIC 与原生部署边界

这个高层 policy RPC 不发送 SONIC `pose` 包，也不启动手部 DDS 驱动。
本地包保留的原生部署循环实现了 Protocol-v4 身体 token 发布、独立 Inspire DDS、
30Hz 时钟、sensor/prediction expiry、暂停/初始化/STOP，并只提交本地实际发送的前缀。
具体控制器、相机裁剪/旋转、网络接口和初始姿态必须按
[固定上游部署说明](https://github.com/NTUMARS/ODE-POLICY/blob/fd8bec470faf7793702533358867db02819e8f27/docs/deploy_groot/README.md#native-g1-deployment)
在对应机器人配置；本地提交不等于机器人执行确认。

准备好部署专用 YAML 后，可运行原生 `ode-g1 deploy --check` 做无连接配置检查。
HumanWAM 命令构建器只提供 `deploy-check`，不会将 `serve` 变成机器人部署。
本次没有启动 SONIC 控制器、相机、DDS、机器人或物理动作。

## 验证

2026-11-20 本地完整 CPU 检查525项及208个 subtest 通过，无跳过，
包含仓库内 ODE G1 实现的105项及6个 subtest。独立 wheel 和源码包构建通过，
实现文件、作者引用、Apache 许可证及第三方声明均包含在相应分发材料中。
使用 Python3.11.16、CPU Torch2.8.0、NumPy1.26.4、Safetensors0.8.0；
该轻量模型检查环境与完整 GPU GR00T Python3.10/Torch2.7.1 环境分开。
CI 另对 Python3.10/3.11/3.12 跑轻量接口检查，并以 Python3.10/CPU Torch2.7.1 跑原生 ODE 检查。

轻量检查覆盖源码/配置、模态布局、H16/76/44约束、缓存输入、原生协议及显式 carry。
本地包另有 ODE G1 的 CPU 原生测试，使用 toy reader/head、fake encoder 和合成数据，
验证真实 optimizer 更新、严格权重重载、精确恢复及已发送前缀语义。
它们不代表完整 GR00T/Cosmos 权重加载、CUDA 推理、任务成功率或真机测试通过。

```bash
python -m pytest packages/humanwam/tests/test_groot_ode*.py -q
PYTHONPATH=packages/ode_g1/src \
  python -m pytest packages/ode_g1/tests -q
```

第二条需要兼容的 CPU Torch、NumPy<2、Safetensors 和 PyArrow 测试环境。
本地包保留 [Apache-2.0 许可](../../packages/ode_g1/LICENSE-APACHE-2.0)与
[第三方声明](../../packages/ode_g1/THIRD_PARTY_NOTICES.md)，
引用和再分发时同时保留 [ODE Policy 来源记录](../../packages/ode_g1/UPSTREAM.md)。权重另遵守各模型许可。
