# G1 全身 ImageWAM Delta-Flow：训练与数据处理

本文对应 `wholebody.inspire_action76.state44.p25.v1`：从 FLUX.2-klein-base-4B
开始全量微调视觉 DiT 和动作专家，冻结 VAE，使用预计算的 Qwen3 文本特征。
这是 **motion-latent policy**，不是直接回归全身电机目标的76关节控制器。

代码分工：

| 文件 | 作用 |
|---|---|
| `src/imagewam/datasets/wholebody.py` | 原始字段映射、归一化、窗口、图像解码 |
| `src/imagewam/wholebody_delta.py` | state仅进动作专家、初始化及checkpoint格式检查 |
| `configs/wholebody/deltaflow76_p25.yaml` | 模型和训练模板，不直接用于启动 |
| `scripts/wholebody/cache_text.py` | 从数据集任务文本生成独立缓存 |
| `scripts/wholebody/build_config.py` | 按真实加载器生成正式/试跑配置 |
| `scripts/wholebody/train.py` | 五步试跑和正式训练入口 |

## 1. 数据要求与字段映射

输入为单任务、30 FPS 的 LeRobot v2.1 数据，所有 episode 参与训练，没有验证集。
数量、总帧数和 instruction 从 metadata 读取，不要求固定97条或固定任务名称。
episode ID 必须从0连续编号，`info.json` 的 split 应为 `{"train":"0:N"}`。
当前版本要求 `chunks_size=1000`。这里只检查已有数据，不自动删除、裁剪或重排原始录像。

```text
DATASET_DIR/
├── meta/
│   ├── info.json
│   ├── tasks.jsonl
│   ├── episodes.jsonl
│   └── episodes_stats.jsonl
├── data/chunk-000/episode_000000.parquet
└── videos/chunk-000/
    ├── observation.images.ego_view/episode_000000.mp4
    └── observation.images.chest/episode_000000.mp4
```

必需数值字段：

| 原始字段 | 每帧维度 | 用途 |
|---|---:|---|
| `action.motion_token` | 64 | motion-latent 动作目标 |
| `observation.state` | 43 | 选取其中29维非手部关节状态 |
| `observation.inspire_left_hand_state` | 6 | 左手实测反馈 |
| `observation.inspire_right_hand_state` | 6 | 右手实测反馈 |
| `observation.projected_gravity` | 3 | 当前重力方向 |
| `timestamp`, `frame_index`, `episode_index`, `index`, `task_index` | 各1 | 时序/索引检查 |

模型输出顺序严格为：

```python
action76 = concat([
    action.motion_token,                    # [0:64]
    observation.inspire_left_hand_state,     # [64:70]
    observation.inspire_right_hand_state,    # [70:76]
])
```

左右手各6维顺序为 `[pinky, ring, middle, index, thumb_bend, thumb_rotation]`。
metadata 可以记录上述通用名字，或分别带 `left_` / `right_` 前缀；代码检查顺序与侧别，
不自动重排手指。
手部使用与 motion token **同一 Parquet 行的实测反馈**，offset=0。
不使用 `teleop.*` 的二值开关、7D抽象手动作或发给执行器的 command 标签。
这里训练的是连续手状态预测，不能将它描述为已经还原了原始下发指令。

当前状态的顺序为：

```python
body29 = concat([observation.state[:22], observation.state[29:36]])
state44 = concat([body29, left_hand6, right_hand6, projected_gravity3])
# state44: body [0:29], left [29:35], right [35:41], gravity [41:44]
```

`state44` 只通过动作专家的 `Linear(44, hidden)` 加到动作 hidden tokens。
视觉分支不创建 proprio encoder，不把 state 拼入文本，也不读取未来 state。

## 2. 每个训练样本怎么构造

每一帧 `t` 都可以作为 anchor，窗口不跨 episode：

| 内容 | 帧范围 |
|---|---|
| 历史动作 source | `t-40 ... t-1` |
| 动作监督目标 | `t ... t+39` |
| 当前 state | 仅 `t` |
| 当前/未来图像 | `t` 与 `t+40`，间隔约1.333秒 |

episode 开头历史不足时重复本 episode 最早可用值并返回 padding mask。
`t=0` 完全无历史时使用**归一化空间的全零 history**，不拿 `action[0]` 填充 source。
末尾不足时重复本 episode 最后值并返回 mask。mask 的具体使用由原生 loss 决定，
不能笼统理解为所有 pad 都必然不计入每项 loss。

当前 state 在数据张量的40步时间轴上重复，模型只取当前项。
当前及未来手状态（`t..t+39`）是 action 标签，不作为未来 proprio 输入。
未来图像是视觉监督目标，不是推理时可用的观测；`noise_idm` 动作上下文使用
当前图像、文本及噪声未来图像 token 的 detached KV，不向动作提供真实未来图像值。

图像按固定顺序处理：ego 在左、chest 在右。每路 RGB 直接双线性 antialias resize
至224×224，再横拼224×448，像素从 uint8 映射到 `[-1,1]`。
不是保持宽高比的 letterbox，也不是 center crop。原始图像可以是不同正整数 H/W，
但必须是3通道RGB视频。输出单样本形状：

```text
video                  [3, 2, 224, 448]
action / past_action   [40, 76]
proprio                [40, 44]  # 当前 state 重复，不是未来40帧 state
text_hidden_states     [128, 7680]
text_attention_mask    [128]
```

视频按 PTS seek/decode 所需帧，使用小型 LRU，不把整个数据集的 RGB 预解码到内存。
帧编号/PTS 的对齐检查不代表已经验证了全部传感器与执行器的硬件同步。

### 归一化

fresh-base 训练必须重新计算本数据集全部 episode 的统计，不借用其他任务的文件。
motion64、body29 和 gravity3 使用各自维度的 q01/q99：

```text
x_normalized = 2 * (x - q01) / (q99 - q01) - 1
```

输出 clamp 到 `[-5,5]`，近常量维度另做安全处理。
手部固定按物理 `[0,1] → [-1,1]`，不使用手部样本分位数拉伸。
本协议的 Parquet 手部值已经是 `[0,1]`，**不能再除1000**。
只有部署时硬件返回原始 `[0,1000]` 才先除1000一次。
反归一化不替代执行端的物理限位。

### 文本缓存

任务文本来自 `meta/tasks.jsonl`，完整 prompt 为：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

缓存文件名由完整 prompt 的 SHA256 和 context length 决定。
缓存只含 instruction 的 Qwen3 特征，不含 state 或未来动作；换任务需要对应的新缓存。

## 3. Delta-flow 与训练目标

```text
visual source = 0.5 * z_current + sqrt(0.75) * epsilon
action source = normalized_past_action + 0.5 * epsilon
```

“视觉25%”指当前 latent 系数的平方为0.25，不是25%的 state、像素混合或 dropout。
动作 source 的76维全部加噪，包括双手连续实测状态。动作 source 不再做单位方差归一化。

视觉 FM＋IC 的总权重为0.5，视觉 IC 权重1.0；动作 FM 权重1.0，动作 IC 权重0.5。
两种 IC 均为一步更新后的 L1。视觉分支由自己的视觉目标更新，动作 FM/IC 的梯度
通过 knowledge insulation / stop-gradient 阻断，不更新视觉分支，但保留视觉条件。
不使用 AURA、C-Pred、history dropout 或 LoRA。

## 4. 环境与外部模型

命令均从仓库根目录执行。使用独立环境，不要把其他 backbone 的旧版本依赖混入。
已验证的服务器环境是 Linux、Python3.11、PyTorch2.7.1+cu128、Transformers4.56.1、
Accelerate1.12.0、DeepSpeed0.18.5。以下 CUDA12.8 安装方式需要相应驱动支持：

```bash
uv venv .venv-wholebody --python 3.11
source .venv-wholebody/bin/activate
uv pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements/wholebody-cu128.txt
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
```

不需要另装 `torchcodec`，本数据适配器使用 PyAV。
根目录旧 `pyproject.toml` 服务于多个 backbone，不能用其旧 Transformers pin
覆盖这里的4.56.1。上面不执行 editable install，源码由 `PYTHONPATH` 引入。

按仓库 FLUX.2 说明准备外部源码、FLUX.2-klein-base-4B、FLUX.2 AE 和 Qwen3-4B：

```bash
git clone https://github.com/black-forest-labs/flux2 third_party/flux2
git -C third_party/flux2 checkout 50fe5162777813d869182b139e83b10743caef15

export FLUX2_SRC="$(pwd)/third_party/flux2"
export FLUX2_MODEL_PATH=/path/to/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/path/to/ae.safetensors
export QWEN3_MODEL_DIR=/path/to/Qwen3-4B
export DATASET_DIR=/path/to/your/lerobot_dataset
export WORK_DIR=/path/to/new/wholebody_work
```

需要先完成有权限的模型下载。下面缓存阶段使用本地文件和 offline 模式，不会替你
下载受限权重。FLUX.2源码与各模型/AE/数据的许可和访问条件分别适用，仓库代码
许可证不等于给所有权重或训练数据授予同一许可。

## 5. 从数据到正式训练

### 5.1 重新生成本任务统计

`WORK_DIR` 必须在原始数据目录以外。准备程序拒绝覆盖已有统计：

```bash
python -m imagewam.datasets.wholebody \
  --dataset-dir "$DATASET_DIR" --output-dir "$WORK_DIR/inputs"
```

产出 `dataset_stats.json` 与 `dataset_audit.json`，记录字段、全量统计与原文件 SHA。
这一步不写原始数据，也不创建验证集。之后数据变化会触发 hash 检查，应使用新的
workdir 重新准备，不能只替换几条 Parquet 后继续冒用旧统计。

### 5.2 缓存纯任务文本

先选择一张空闲 GPU，以下只是命令示例，不要在已有训练占用的卡上另起任务：

```bash
python scripts/wholebody/cache_text.py \
  --dataset-dir "$DATASET_DIR" --cache-dir "$WORK_DIR/qwen3" \
  --qwen-model "$QWEN3_MODEL_DIR" --flux2-src "$FLUX2_SRC" --device cuda:0
```

### 5.3 从基础模型生成动作专家共享初始化

```bash
python scripts/flux2/preprocess_action_dit_flux2.py \
  --model-config configs/wholebody/deltaflow76_p25.yaml \
  --flux2-src-path "$FLUX2_SRC" --flux2-model-path "$FLUX2_MODEL_PATH" \
  --variant klein-base-4b --action-dim 76 --apply-alpha-scaling true \
  --device cpu --output "$WORK_DIR/action_shared_flux2_4b.pt"
```

仅生成114个 FLUX.2 基础模型转换共享张量，不是从 LIBERO、衣服或其他机器人任务
的已训练权重续训。76D输入/输出和44D state投影等六个张量明确新初始化。
输出文件由配置生成器计算并固定 SHA，加载器检查完整共享键、形状、metadata及
数值有限性，允许记录在案的 FP32→BF16 转换。不要使用旧任务 checkpoint 代替它。

### 5.4 生成实际训练配置

`--output-dir` 必须是尚不存在的新目录：

```bash
python scripts/wholebody/build_config.py \
  --dataset-dir "$DATASET_DIR" --stats-path "$WORK_DIR/inputs/dataset_stats.json" \
  --text-cache-dir "$WORK_DIR/qwen3" --output-dir "$WORK_DIR/run60" \
  --flux2-src "$FLUX2_SRC" --flux2-model "$FLUX2_MODEL_PATH" \
  --ae-model "$FLUX2_AE_MODEL_PATH" --qwen-model "$QWEN3_MODEL_DIR" \
  --action-initializer "$WORK_DIR/action_shared_flux2_4b.pt" \
  --epochs 60 --batch-size 8 --accumulation 2 --num-processes 8
```

产生 `production.yaml`、`smoke.yaml`、`build_receipt.json`。
每轮更新数来自原生 loader 和 Accelerate 的实际分片长度，再按 accumulation 取 ceil。
总步数是该值乘 epochs，5%线性warmup后 cosine 到1e-6。不能只改 `max_steps`，
也不能把其他数据集的728 updates/epoch照抄过来。

默认8卡×每卡8×累积2，名义全局 batch128，BF16、ZeRO-1、activation checkpointing，
学习率1e-4、weight decay0.01、seed42。每轮保存一次，保留最近4份权重和最近1份
training state。没有 validation，日志 loss 不能作为真机成功率。

### 5.5 CPU检查与五步真实试跑

```bash
python tests/test_wholebody_dataset.py
python tests/test_wholebody_delta.py
python tests/test_wholebody_training_cli.py

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 \
  scripts/wholebody/train.py --config "$WORK_DIR/run60/smoke.yaml" --smoke
```

试跑执行5次真实 optimizer update，不保存权重/optimizer。检查当前 state 不改变视觉
输入、detached workspace、source数值、目标对应梯度、FP32参数更新及跨rank一致性，
并在各rank产出 `smoke_rank_N.json`。试跑失败时先看日志，不要直接跳过进入正式训练。
综合 FM 包含视觉 FM，不能单靠它的视觉梯度来证明动作 FM 隔离；模型单元测试
另外分别检查 action FM 和 action IC 对视觉的梯度路径。

### 5.6 正式训练

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 \
  scripts/wholebody/train.py --config "$WORK_DIR/run60/production.yaml"
```

正式入口要求对应的全部 smoke receipts，并重新从 FLUX.2 基座和共享初始化加载，
不接着 smoke 的5步模型训练。新输出目录拒绝覆盖，当前入口不提供隐式 resume。
改变卡数、batch、数据、学习率或轮数后，应重新生成配置并重新做对应试跑。
长时间训练建议通过服务器上的 tmux 或作业系统启动，不依赖笔记本 SSH 会话存活。

## 6. 版本、测试与部署边界

本 portable 版的数据/模型 CPU 测试及真实数据配置检查与具体运行记录分开。
原始服务器实现已有八卡五步训练与真实模型隔离检查；移植后的命令若尚未重新
执行完整 GPU 流程，不能把 CPU PASS 写成新的八卡训练或真机 PASS。

早期衣服40/60轮模型使用78D抽象手动作、视觉50% prior，不是此处的76D实测手目标。
后来的衣服76D/p25版本是权重迁移续训，部分归一化刻意继承 parent，也不同于本文
“从基座新训练、重新fit本任务统计”的流程。

本版 checkpoint format 为 `wholebody_inspire76_p25_flux2_v1`，与历史实验专用格式
分开。历史权重应使用其配套发布代码，不能只改格式字符串后绕过严格检查。
适配器还核验当前原生 `imagewam.py` 与 `action_dit_flux2.py` 的源码 SHA。
修改这两个文件后需要重新审核兼容性，不能直接关闭检查。

76D输出前64维需匹配的 SONIC/motion decoder 和低层控制器。后12维是预测的实测
手状态，发送到执行器前还需单位/顺序映射、限位、执行步数、控制频率和急停验证。
R1Lite 的7D推理入口不适用于此76D全身模型。本训练包不含独立机器人控制桥，
没有提供或宣称闭环真机成功率。

仓库不上传原始视频/Parquet、私有数据、任务归一化统计、prompt cache、权重、
optimizer state 或访问凭据。外部数据和权重仍需各自授权。
