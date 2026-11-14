# G1 部署：模型服务与 SONIC v1.1

FastWAM 的 GR00T feedback76/H16 policy RPC、数据映射和独立训练入口见
[FastWAM SONIC 接入说明](policies/FASTWAM_SONIC.md)。它复用既有 feedback76 执行链路，
需要自己的匹配权重；下文仍专门描述两条 ImageWAM 路线。

本文对应 main 上的两条 ImageWAM Δ-flow 路线。模型训练见[训练入口](TRAINING_MODES.md)，
关节标签来源见[数据审计](policies/REFERENCE_LABEL_AUDIT_20261110.md)。

HumanWAM 已实现同进程直接调用、可选的常驻 HTTP 预测与客户端，以及关节参考的时钟检查和协议导出。
SONIC v1.1 接收端由固定版本的上游实现提供。本仓库目前没有完整的在线 ZMQ 发布循环或
Inspire 硬件驱动桥，下面明确标出这些接线边界；启动模型服务不等于已经启动机器人控制。
新 joint47 v2 需要先训练自己的模型，目前没有可直接下载的已训练 joint47 v2 checkpoint。

| 路线 | 模型输出 | 身体执行链路 | 双手 |
|---|---|---|---|
| Motion token | motion64＋Inspire 左6右6，合计76D | token64 → SONIC v4 external-token 接口 → decoder | Inspire12 单独映射到现有手部驱动 |
| Joint47 v2 | body29＋Inspire12＋root rotation6D，合计47D | 位置轨迹 → 速度计算＋world root quaternion → v1 body reference → SONIC v1.1 encoder → decoder | 同上，不能装入 body packet |

表中的 **v1/v4 是消息协议版本，控制器均为 SONIC v1.1**。76D、47D和官方含 Dex3 手部的78D不互换。
根据要测试的路线启动对应服务。示例默认使用`cuda:0`；并行启动时需按实际显存配置各自的device或可见GPU。

## 1. 准备环境与模型文件

在 GPU 推理主机上获取本仓库。ImageWAM 与 SONIC 使用各自的环境，不将 GR00T 的 Python 依赖覆盖到 ImageWAM 环境。

```bash
git clone --branch main https://github.com/zuo-kuangji/HumanWAM.git
cd HumanWAM
git submodule update --init third_party/GR00T-WholeBodyControl

export HUMANWAM_ROOT="$PWD"
export IMAGEWAM_PYTHON=/absolute/path/to/imagewam-env/bin/python
export PYTHONPATH="$HUMANWAM_ROOT/src:$HUMANWAM_ROOT/packages/humanwam/src${PYTHONPATH:+:$PYTHONPATH}"
"$IMAGEWAM_PYTHON" -m pip install -e packages/humanwam
```

ImageWAM 原生环境的安装命令和 CUDA 版本见[环境说明](G1_WHOLEBODY_DELTA.md#4-环境与外部模型)。
不要用根目录其他 backbone 的旧依赖覆盖该环境。

对于本仓库新训练得到的 native checkpoint，保留同一次实验的以下文件：

| 文件 | 用途 |
|---|---|
| 模型 checkpoint | 与训练模式、horizon和协议匹配的完整任务权重 |
| resolved `production.yaml` | 模型 factory、动作空间、时序和基础模型路径 |
| `dataset_stats.json` | 本任务的归一化、instruction及关节 source contract |
| `<prompt-hash>.qwen3_flux2_len128...pt` | 同一任务的纯 instruction 缓存 |
| 训练使用的 shared action initializer | native factory 初始化所需；保留原文件及 SHA |
| FLUX.2 基座、AE和外部 FLUX.2 源码 | 按 YAML 中实际使用的路径提供 |

不需要上传或加载 optimizer/scheduler state。使用已有文本缓存时，推理不重新运行 Qwen。
迁移机器后，只将 YAML 中使用的文件路径指向新位置，保留结构、horizon、统计值、instruction和各 SHA。
不要编辑 `dataset_stats.json` 来“适配”另一组权重，也不要将 motion76 checkpoint 改名后交给 joint factory。

## 同进程直接调用，不经过 HTTP

直接调用和HTTP都保留。已有采集／执行程序能使用ImageWAM环境时，可直接复用policy对象；
希望将模型与调用方放在不同进程或环境中时，可使用下文的HTTP服务。两种方式返回相同的动作语义，
训练和SONIC本身不要求HTTP。

下面代码在实际ImageWAM解释器中运行，所需环境变量与对应路线的服务配置相同：

```python
import json
from pathlib import Path

from humanwam.config import load_config
from humanwam.observations import observation_from_npz
from humanwam.registry import build_policy

# Motion token；Joint47改为configs/humanwam/imagewam-joint47.json。
policy = build_policy(load_config("configs/humanwam/imagewam-h16.json"))

def predict_capture(npz_path, metadata_path):
    metadata = json.loads(Path(metadata_path).read_text())
    observation = observation_from_npz(npz_path, metadata)
    return policy.predict(observation)

prediction = predict_capture("actual_observation.npz", "actual_metadata.json")
# 后续对新观测继续调用predict_capture，复用同一policy和已加载模型。
```

首次`predict`加载native模型，`build_policy`本身不切换Python解释器。
直接调用同样需要实际观测和严格过去的history；joint结果直接交给第7节的时钟runtime。
它不会绕开归一化、source contract、heading恢复或过期检查。下面的HTTP章节是可选部署方式。

## 2. 启动 Motion token 模型服务

以下命令对应新 native H16 模型：

```bash
export IMAGEWAM_MODEL_CONFIG=/absolute/path/to/motion-run/configs/production.yaml
export IMAGEWAM_CHECKPOINT=/absolute/path/to/motion-checkpoint.pt
export IMAGEWAM_TEXT_CACHE=/absolute/path/to/task-hash.qwen3_flux2_len128.pt

"$IMAGEWAM_PYTHON" -m humanwam serve \
  --config configs/humanwam/imagewam-h16.json --dry-run

"$IMAGEWAM_PYTHON" -m humanwam serve \
  --config configs/humanwam/imagewam-h16.json
```

默认服务地址为 `http://127.0.0.1:8090`。`--dry-run`只展示启动命令，不加载权重。

已发布的历史模型按自己的 release 入口加载：

| 模型 | 配置 | 额外环境变量 |
|---|---|---|
| 靠垫 release | `configs/humanwam/imagewam-pillow-release.json` | `IMAGEWAM_RELEASE`、`FLUX2_MODEL_PATH`、`FLUX2_AE_MODEL_PATH` |
| 垃圾桶 release | `configs/humanwam/imagewam-trash-release.json` | 同上 |
| 衣服 final25 legacy H40 | `configs/humanwam/imagewam-clothes-legacy40.json` | 同上，必须是完整的对应 release |

`IMAGEWAM_RELEASE`应指向含该 release 的 manifest／配置／推理脚本的实际目录，未必是 HF 下载的最外层目录。
保留 release 原目录结构和自己的统计。衣服的 H40 与新 joint47 的40点预测不是同一动作空间或权重格式。
启动时仅将上述 `--config` 换成对应 release 配置。

## 3. 启动 Joint47 v2 模型服务

```bash
export IMAGEWAM_JOINT_MODEL_CONFIG=/absolute/path/to/joint-run/configs/production.yaml
export IMAGEWAM_JOINT_CHECKPOINT=/absolute/path/to/joint-checkpoint.pt
export IMAGEWAM_JOINT_TEXT_CACHE=/absolute/path/to/task-hash.qwen3_flux2_len128.pt

"$IMAGEWAM_PYTHON" -m humanwam serve \
  --config configs/humanwam/imagewam-joint47.json --dry-run

"$IMAGEWAM_PYTHON" -m humanwam serve \
  --config configs/humanwam/imagewam-joint47.json
```

默认服务地址为 `http://127.0.0.1:8092`。该配置要求 v2 模型和统计，不能加载 legacy joint v1。
模型历史16点、总预测40点；执行段为16个源采样间隔，其余预测用于维持 SONIC 前瞻。
预测点间隔是训练数据的 `2/30` 秒，低层跟踪频率为50Hz。

两个native入口均延迟加载模型，历史release按自己的脚本实现加载。下面只检查 HTTP 是否存活：

```bash
curl --fail http://127.0.0.1:8090/health
curl --fail http://127.0.0.1:8092/health
```

`/health`成功不代表权重已加载或 GPU 推理已通过。首次真实 `/predict` 会触发加载。
先用一次实际观测预热并丢弃该结果，然后重新采集新的图像、state、history和时间用于在线预测。
不得更新时间戳，把冷启动期间已经过期的图像或结果伪装成新观测。

## 4. 输入 NPZ 与 JSON metadata

采集器应分别保存数组和标量声明。输入是物理原值，由模型 backend 使用对应统计归一化。

| 字段 | 保存位置 | Motion token | Joint47 v2 |
|---|---|---|---|
| `ego_rgb`, `chest_rgb` | NPZ | 当前 uint8 RGB `[H,W,3]` | 相同 |
| `current_state` | NPZ | `[44]` | 相同 |
| `past_actions` | NPZ | `[N,76]`，只含严格过去的数据 | `[N,47]`，`N≤16`；姿态为world rotation6D |
| `past_action_frame_offsets` | NPZ | 不需要 | 整数 `[-2*N,...,-2]` |
| `current_root_quaternion_wxyz` | NPZ | 不需要 | 当前实测 base IMU 单位四元数 `[4]` |
| `past_root_orientation_valid` | NPZ，可选 | 不需要 | 与 N 对应的 bool mask |
| `observation_time_s` | JSON | 不需要 | 实际采集时刻，policy主机的monotonic秒数 |
| `timestamp_clock` | JSON | 不需要 | 字符串 `"monotonic"` |
| `at_episode_start` | JSON | 不需要 | 空历史时显式 `true` |
| `seed` | JSON，可选 | 整数 | 整数 |

当前 NPZ 读取器不会将 scalar ndarray 自动转换为 Python 标量，因此时间、字符串和布尔声明放在 JSON 中。
同一 key 不能同时写进 NPZ 和 metadata。

当前 state44 的顺序为：

```text
body29 + left_inspire_measured6 + right_inspire_measured6 + projected_gravity3
```

现有43D记录中，body29取 `observation.state` 的 `[0:22] + [29:36]`，保持训练时的关节顺序。
每手6维顺序为 `pinky, ring, middle, index, thumb_bend, thumb_rotation`，值已是 `[0,1]` 时不再除1000。
图像保持 RGB；backend将每路resize224×224，再按ego左、chest右横拼，调用方不用预先拼图。

Motion history 每行是 `motion64 + left_feedback6 + right_feedback6`。
Joint history 必须与训练 source contract 同源：measured实验用过去实测关节及IMU旋转，
WBC实验用声明的过去电机目标，direct-reference用对应参考。不能直接复用另一模式的history。
Joint47的world6D为旋转矩阵两列依次拼接 `[R00,R10,R20,R01,R11,R21]`，backend会相对当前heading转换。

由实际采集器提供变量后，保存方式如下；示例不生成假图像、假state或新的采集时间：

```python
import json
import numpy as np
from pathlib import Path

# ego_rgb/chest_rgb/state44/history47/root_wxyz/observed_at_policy_monotonic
# 均由本次实际采集提供；observed_at_policy_monotonic必须是采集时刻。
np.savez_compressed(
    "joint_observation.npz",
    ego_rgb=ego_rgb,
    chest_rgb=chest_rgb,
    current_state=state44.astype(np.float32),
    past_actions=history47.astype(np.float32),
    past_action_frame_offsets=np.arange(-2 * len(history47), 0, 2, dtype=np.int64),
    current_root_quaternion_wxyz=root_wxyz.astype(np.float32),
)
Path("joint_metadata.json").write_text(json.dumps({
    "observation_time_s": float(observed_at_policy_monotonic),
    "timestamp_clock": "monotonic",
    "at_episode_start": bool(at_episode_start),
}))
```

episode起点用形状明确的空历史 `[0,76]` 或 `[0,47]`，joint同时声明 `at_episode_start=true`。
调用方负责维护历史及其采样时间；HTTP服务不自动维护已执行动作历史。
不能把整段尚未执行的未来预测放入history，也不能把最近16个50Hz tick当作joint训练的stride2历史。

## 5. 发起预测请求

调用服务的客户端只需要轻量 HumanWAM 包，不需要加载 PyTorch 模型：

```bash
python -m humanwam.http_client \
  --url http://127.0.0.1:8090/predict \
  --observation motion_observation.npz --metadata motion_metadata.json \
  --output new_motion_prediction.json --timeout 180

python -m humanwam.http_client \
  --url http://127.0.0.1:8092/predict \
  --observation joint_observation.npz --metadata joint_metadata.json \
  --output new_joint_prediction.json --timeout 180
```

输出路径必须是新文件。客户端原样转发 NPZ 和 metadata，不补时间、不归一化、不构造history，
也不会连接执行器。服务成功时直接返回 prediction dict，没有额外的 `result` 包裹。
`--timeout`仅限制HTTP等待时间，不放宽joint计划的延迟与过期检查。

客户端在其他主机时，可将本地端口转发到GPU主机的服务，例如：

```bash
ssh -N -L 18090:127.0.0.1:8090 gpu-host
```

随后客户端请求 `http://127.0.0.1:18090/predict`。Joint可相应转发8092。
端口转发只解决网络访问，不同步两台机器的monotonic clock；joint的时间要求仍按下文执行。
如果保留了HF下载代理环境，本地服务地址应包含在 `NO_PROXY` 中。

单次离线检查也可直接加载模型：

```bash
"$IMAGEWAM_PYTHON" -m humanwam predict \
  --config configs/humanwam/imagewam-h16.json \
  --observation motion_observation.npz --metadata motion_metadata.json \
  --output new_motion_prediction.json
```

Joint改用 `imagewam-joint47.json` 及其输入文件。`predict`在当前解释器内运行，不按配置里的
`python`另起进程，因此这里必须显式使用 ImageWAM 解释器。单次CLI不适合每次控制重启加载模型。

服务使用 checkpoint 对应 instruction 的文本缓存，不通过 metadata 随意替换成其他任务。
`POST /reset`需JSON body `{}`，当前两backend返回 `null`；它不清除调用方history，也不重置机器人或SONIC会话。

## 6. Motion token 版接 SONIC

新native模型的 `actions[:, :64]` 是motion token，后12维是左右Inspire反馈目标。
Release wrapper也提供标准 `actions` 字段，保留自身horizon。

```text
当前RGB/state/历史 → 常驻ImageWAM服务 → action76
                                         ├─ motion64 → v4 token publisher → SONIC v1.1 decoder
                                         └─ Inspire12 → 已验证的手部映射/驱动
```

身体沿用现有 G1 的 external-token 发布器与已验证的token处理设置，不将76维整体塞入 `token_state`。
官方 VLA 启动脚本默认的 Dex3 7＋7 手命令与这里的 Inspire6＋6不同，不能原样套用。
HTTP服务本身不是SONIC ZMQ服务；需要执行端读取prediction并按自己的控制时钟发布。

## 7. Joint47 版接 SONIC

Backend已将模型预测的相对heading旋转恢复为world四元数，结果包含：

```text
body_joint_position29[40,29]   # absolute rad，MuJoCo顺序
inspire_hand_state12[40,12]
root_quaternions_wxyz[40,4]   # 已恢复world
relative_timestamps[40]
observation_time_s / prediction_ready_time_s / source contract
```

在线接线用 `TimedJointPlan.from_policy_prediction` 和 `JointReferenceBuffer`，完整代码接口见
[关节时钟与协议说明](policies/SONIC_JOINT_REFERENCE.md#带时钟的计划检查)。
它会检查整个执行段、0.9秒前瞻、推理延迟、50Hz网格及过期状态；measured/WBC来源需显式实验选择。
不要通过低层数组接口绕开 checkpoint/source contract 检查。

```text
joint prediction → 时钟/来源检查 → 50Hz q、qvel、root quaternion
                                    ├─ body v1 packet → SONIC joint encoder → decoder
                                    └─ Inspire12 → 独立手部执行端
```

导出器已经将身体重排为IsaacLab顺序。packet中的q是绝对rad，不再减默认站姿，也不重复做关节重排。
速度从同一位置轨迹求导，不是FM噪声时间的velocity，也不是复制当前实测速度到整个未来。
原生SONIC还会应用初始化heading对齐和operator delta；backend恢复world不等于绕过这层会话对齐。
切换到joint输入时，要核对初始robot/reference heading及现有会话状态，不能默认该对齐恒为identity。

`build_sonic_joint_window.py`可将一次完整prediction JSON变成body bytes、独立Inspire NPZ和manifest，
用于协议核验。它不启动网络或执行器；过期JSON应被拒绝，不是将时间改成现在再导出。

Joint示例建议采集客户端、模型服务和控制桥位于同一GPU主机，共用该主机的monotonic clock。
G1、笔记本与NTU各自的 `time.monotonic()`不能直接互换。现有HTTP接口不提供跨主机时钟映射。
SONIC native会按frame index处理参考，但不会自动执行本runtime的expiry/deadline规则；
真实发布循环必须调用窗口检查并处理stale/overrun，不能将落盘JSON当成永久可执行的计划。

常见错误的处理方式：

| 错误 | 核对内容 |
|---|---|
| `/health`成功但`/predict`失败 | 首次请求才加载native模型，检查返回的权重/依赖错误及服务日志 |
| checkpoint/schema不匹配 | 检查native、release、legacy和joint v2入口，不能只改horizon |
| joint采集时间类型错误 | 将Python标量时间和clock声明放metadata JSON，避免NPZ scalar ndarray |
| 空历史被拒绝 | 保留正确空数组形状，joint显式声明episode起点 |
| stale、overrun或lookahead不足 | 核对预热、新观测、真实推理耗时和时钟映射，不能刷新旧时间来通过检查 |

## 8. SONIC v1.1接收端与仿真

已经运行G1的用户保留自己已验证的接口、驱动和会话设置。本节只列固定上游的仿真接收端命令，
不启动本仓库尚未实现的live sender或Inspire驱动。

在SONIC自己的环境准备匹配的v1.1文件：

```bash
cd "$HUMANWAM_ROOT/third_party/GR00T-WholeBodyControl"
python download_from_hf.py --sonic-v1-1
```

使用同目录的 `policy/sonic_v1_1/model_encoder.onnx`、`model_decoder.onnx` 和
`observation_config.yaml`，不能混入release/low-latency版本。
SONIC编译、运行依赖以及MuJoCo环境安装与启动见[固定上游quickstart](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/docs/source/getting_started/quickstart.md)。
在另一终端启动原生仿真接收端：

```bash
cd "$HUMANWAM_ROOT/third_party/GR00T-WholeBodyControl/gear_sonic_deploy"
./deploy.sh \
  --cp policy/sonic_v1_1/model \
  --obs-config policy/sonic_v1_1/observation_config.yaml \
  --input-type zmq --zmq-host 127.0.0.1 \
  --motor-kp-scale 4,10=1.5 --motor-kd-scale 4,10=1.5 \
  sim
```

增益参数来自固定上游的v1.1仿真/硬件说明，不是HumanWAM新推导的设置。
原生ZMQ链路是publisher `PUB bind`、控制器 `SUB connect`，默认topic `pose`、端口5556；
它与HTTP8090/8092是不同端点。`deploy.sh`不转发自定义 `--zmq-port/--zmq-topic`，
需要改端口时按[原生参数说明](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/docs/source/references/deployment_code.md)直接调用C++入口。
仅启动接收端不会自动消费HumanWAM HTTP输出。

## 9. 当前验证范围

- 模型路由、FM/endpoint loss、数据解码、heading转换、时钟窗口和协议一致性已通过
  [295项native CPU检查](policies/TRAINING_ROUTES_VALIDATION_20261110.md)。
- HTTP客户端另有真实loopback测试，验证原样传输metadata、错误响应和输出保护。
- 本文的服务启动参数与CLI已核对；本次没有启动GPU模型、SONIC控制器或机器人。
- Joint47 v2尚无训练权重、完整encoder/decoder闭环或真机任务结果；手部硬件桥和在线publisher仍需在实际G1部署端接入。
