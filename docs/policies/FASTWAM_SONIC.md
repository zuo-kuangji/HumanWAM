# FastWAM 与 GR00T feedback76 的 SONIC 接口对齐

`fastwam_sonic` 是独立的 Gaussian FM G1 入口，使用与
[`groot/feedback76_h16.py`](../../configs/humanwam/groot/feedback76_h16.py)
相同的观测字段、动作字段及 H16。原 `fastwam` 的 LIBERO/RoboTwin 入口保持原动作空间。

| 契约 | FastWAM SONIC / GR00T feedback76 |
|---|---|
| 当前图像 | `video.ego_view`、`video.chest`，uint8 RGB `[1,1,H,W,3]` |
| 当前 state | `state` 下按 left_leg6、right_leg6、waist3、left_arm7、right_arm7、left_hand6、right_hand6、projected_gravity3 分组；各为 raw float32 `[1,1,D]` |
| 指令 | `language["annotation.human.task_description"] = [["任务指令"]]` |
| 动作 | `motion_token[1,16,64]`、`left_hand_joints[1,16,6]`、`right_hand_joints[1,16,6]`，float32、已反归一化 |
| 时序 | 当前帧起连续16个原始采样点；频率从数据保留到 checkpoint manifest，由客户端按该频率执行 |
| 协议 | GR00T PolicyClient 的 ZeroMQ REQ/REP、msgpack-numpy，默认 `127.0.0.1:5555` |

state44 的身体顺序来自原 `observation.state[0:22] + [29:36]`。
手部是每手6维实测 Inspire 值 `[0,1]`，按原数据顺序保留，不替换成7维命令。
动作标签按 GR00T 配方使用同一行 motion64 和左右手实测反馈。
模型使用原生当前 state conditioning，Gaussian 噪声源，不接受过去动作或未来观测输入。
两个当前视角各 resize224²，ego 左、chest 右拼接；不做 LIBERO gripper 翻转或二值化。

## 环境和数据

在 FastWAM 独立环境安装，不覆盖 GR00T 的环境。基础模型和依赖沿用
[原生维护说明](FASTWAM.md)。额外安装轻量协议依赖及数据准备依赖：

```bash
cd "$HUMANWAM_ROOT"
"$FASTWAM_PYTHON" -m pip install -e './packages/humanwam[sonic,data]'
python scripts/humanwam/bootstrap.py --only fastwam
"$FASTWAM_PYTHON" -m pip install -e third_party/FastWAM
```

使用已有 LeRobot v2.1 记录创建独立副本；原数据不被修改，视频通过 symlink 引用：

```bash
"$FASTWAM_PYTHON" -m humanwam.backends.fastwam_sonic_data \
  --source /data/original-g1 --output /data/new-fastwam-feedback76
```

转换器要求原 state43、motion64、实测 Inspire6+6、gravity3 和 ego/chest 相机。
它按上述映射写 action76 和 state44，保留原 FPS、帧序、任务及图像数据，更新向量统计与 metadata，
记录 `meta/humanwam_feedback76.json`。本入口不把缺少反馈手状态的数据补零，也不重采样。
视频链接要求原数据一直可用。该转换不代表记录质量、相机同步或真机可执行性已核验。

## 新训练

`fastwam-sonic-train.json` 构造真实 Accelerate 原生训练进程。
专用 composition 固定 action76、state44、17个原视频采样点/16动作、video stride4、
Gaussian FM shift1、min/max 统计和绝对标签；只允许修改列出的训练超参数。
原生 transformer/backbone 结构及训练损失不被修改。

先按原生维护说明准备本地 Wan VAE、T5、tokenizer，以及新训练所需 Wan DiT 分片。
导出解析完成的专用 model config，避免 initializer 脚本遇到未解析维度后回退到7：

```bash
"$FASTWAM_PYTHON" - <<'PY'
import os
from pathlib import Path
from omegaconf import OmegaConf
from humanwam.backends.fastwam_sonic_training import compose_sonic_config
cfg = compose_sonic_config(Path(os.environ["HUMANWAM_ROOT"]) / "third_party/FastWAM")
with open("/configs/new-fastwam-feedback76-model.yaml", "x") as stream:
    stream.write(OmegaConf.to_yaml(cfg.model, resolve=True))
PY

cd "$HUMANWAM_ROOT/third_party/FastWAM"
export DIFFSYNTH_MODEL_BASE_PATH="$FASTWAM_MODEL_CACHE"
export DIFFSYNTH_SKIP_DOWNLOAD=true
"$FASTWAM_PYTHON" scripts/preprocess_action_dit_backbone.py \
  --model-config /configs/new-fastwam-feedback76-model.yaml \
  --output "$FASTWAM_SONIC_ACTION_INIT" --device cpu --dtype bfloat16

# 这里只复用上游扫描 tasks.jsonl 并生成文本缓存的脚本，不使用其 LIBERO 数据或动作头。
"$FASTWAM_PYTHON" scripts/precompute_text_embeds.py task=libero_uncond_2cam224_1e-4 \
  model.redirect_common_files=false \
  "data.train.dataset_dirs=[\"$FASTWAM_SONIC_DATA_ROOT\"]" \
  "data.train.text_embedding_cache_dir=$FASTWAM_SONIC_TEXT_CACHE"

cd "$HUMANWAM_ROOT"
humanwam train --config configs/humanwam/fastwam-sonic-train.json --dry-run
humanwam train --config configs/humanwam/fastwam-sonic-train.json
```

运行前设置 `FASTWAM_PYTHON`、`FASTWAM_MODEL_CACHE`、`FASTWAM_SONIC_DATA_ROOT`、
`FASTWAM_SONIC_ACTION_INIT`、`FASTWAM_SONIC_TEXT_CACHE` 和 `FASTWAM_SONIC_OUTPUT`。
ActionDiT initializer 是基础 backbone 初始化，不是训练好的 G1 策略。
原 LIBERO/RoboTwin 发布权重、GR00T 权重不能直接用于此入口。

## 权重登记与服务

完成匹配训练后，使用该次输出的完整 `.pt` 与 `dataset_stats.json` 创建 manifest。
`--action-frequency-hz` 必须与转换收据中的真实 FPS 相同：

```bash
"$FASTWAM_PYTHON" -m humanwam.backends.fastwam_sonic \
  --checkpoint "$FASTWAM_SONIC_CHECKPOINT" --stats "$FASTWAM_SONIC_STATS" \
  --action-frequency-hz 30 --output "$FASTWAM_SONIC_MANIFEST"
humanwam serve --config configs/humanwam/fastwam-sonic.json --dry-run
humanwam serve --config configs/humanwam/fastwam-sonic.json
```

Manifest 绑定 schema、频率、原生版本及权重/统计 SHA256。登记是操作者声明，不证明训练已成功。
实际加载还会严格检查完整 MoT、proprio encoder 的所有 key 和 shape，拒绝不完整或7/14维权重。
首次有效 `get_action` 才加载模型，加载前重新检查资产哈希；`ping` 不证明模型可推理。

现有 GR00T 客户端可调用同样的接口：

```python
from gr00t.policy.server_client import PolicyClient

with PolicyClient(host="127.0.0.1", port=5555) as policy:
    modality_config = policy.get_modality_config()
    action_dict, info = policy.get_action(observation)
    policy.reset({})
```

这里的 `observation` 必须是本次采集的原始嵌套观测。
离线 `humanwam predict` 也支持 `video.*`、`state.*` NPZ 字段及 JSON language metadata，
并像 GR00T feedback76 一样额外返回 dense `action[16,76]`。

服务是高层 policy RPC，不是低层 SONIC `pose` topic 发布器。
身体 motion64 仍由现有 G1 token 发布器送入匹配的 SONIC decoder，Inspire6+6 走现有手部驱动。
官方 stock `UNITREE_G1_SONIC` 的78维输出、H40和每手7维 consumer 与此契约不同，
需要使用现有 feedback76 执行端；不能把6维手值送入7维消费者。
`reset` 仅是无状态 policy RPC，不重置机器人或 SONIC 会话。执行端负责真实时钟、预热、
丢弃过期预测及动作发布；本服务不自动补历史、不发机器人命令。

## 验证范围

CPU 检查覆盖与现有 GR00T feedback76 配方逐字段对照、实际 pinned GR00T PolicyClient
和 serializer 的真实 ZeroMQ loopback、图像顺序、state44拼接、H16动作拆分、
错误请求后恢复、资产/统计不匹配、原生 Hydra composition 及合成 parquet 数据转换。
测试使用替代模型输出，不表示 GPU forward、训练、SONIC encoder/decoder 闭环或真机测试通过。
本次没有产生训练权重，也没有启动执行器。
