# Humanoid Benchmark 训练可复现规范（权威版）

> **所有在本仓库安排 Humanoid Benchmark 训练的人和 agent，开训前必须逐项对照本文件和 [`protocol.yaml`](protocol.yaml)，训完上传前运行 [`check_protocol.py`](check_protocol.py)。**
> 本文件取代 `hb_training/README.md`「训练设置」一节中 2026-11-10 的旧设置（20 epoch、batch 696 等，均已废止）。
> 数据增强的细节以项目规范 v2 为准：[`docs/augmentation_v2.md`](docs/augmentation_v2.md)；H2 任务另见 [`docs/h2/`](docs/h2/)。
> 偏离本文件的唯一合法情形：用户明确声明的例外。例外必须同时写进运行目录、HF 上的 `protocol.json` 和第 7 节的登记表。

最后核对：2026-11-23。核对依据是集群上实际训练产物：GR00T 的 `checkpoint-*/training_args.bin`、Psi-Zero 的 `train_args.txt`、FastWAM 的 `out/config.yaml`、HF 上已发布的 `protocol.json`，不是凭配置文件推测。

---

## 0. 目的与原则

**目的**：第三方研究者只要拿到以下四样东西，就能重训出统计上等价的模型，并复现评估数字。
- 数据集：HF repo + commit + 条数规则
- 本仓库 commit：子模块 commit 已由 `.gitmodules` 固定
- 环境版本：见 `hb_training/envs/`
- HF checkpoint 目录里的 `protocol.json` 和完整配置

**原则**

1. **接口层所有模型完全一致**：数据版本、条数、相机、state/action 定义、增强规范 v2、全局 batch 512、步数规则、训练中 MSE 协议。
2. **模型内部按各自官方配方**：网络结构、优化器分组、归一化方式、动作块长度、官方训练种子。
3. **同一模型在所有任务上配方一致**：种子、batch、学习率曲线、存档规则都一样。新任务、补训、重训、续训都不得悄悄改动。
4. **一切随机源固定并记录**：训练种子、数据采样种子、增强种子、本体常偏种子、评估种子。
5. **一切输入固定版本**：数据 commit、代码 commit、基座权重、Python/torch/CUDA 版本、节点和 GPU 布局（布局会影响数据分片和随机流）。
6. **一切产物可追溯**：每个 checkpoint 都带上 protocol、完整配置、归一化统计、训练日志、训练中 MSE 曲线和所用代码快照。
7. **禁止事项**
   - 梯度累积（任何模型 `gradient_accumulation_steps = 1`）
   - 续训峰值学习率等于原峰值
   - 改种子刷结果
   - 覆盖 HF 上已发布的目录
   - 用事后补跑的 MSE 代替训练中 MSE
   - 用模型原生图像增强叠加本规范的增强

---

## 1. 全局固定项（接口层，所有模型）

| 项 | 规定 |
|---|---|
| 本体输入 | G1：44 维 = 29 身体关节角 + 实测双手 6+6 + 投影重力 3。H2：78 维 = 75 关节 + 重力 3。**不含关节速度**，也不计算速度统计。 |
| 动作 | G1：76 维 = motion token 64 + 同帧实测双手 12。H2 A17/A18：75 维。H2 A14：82 维 = body31 + hands44 + SONIC planner 7。 |
| 相机 | G1：`ego_view` + `chest`。H2：`head` + `chest`（不用腕部相机，11-14 起 H2 一律两视角）。 |
| 视觉增强 | 规范 v2：15 项增强，每个样本均匀随机挑 1 项放大到 1.5 倍；颜色在两路相机和时间维度上共享；几何每路相机独立；传感器类只作用于观测图。实现在 `hb_groot/hb_aug_v2.py`（sha256 前缀 `9f7713cea4f846ef`）。各模型原生图像增强全部关闭。 |
| 本体增强 | `hb_aug_v2.perturb_joints`：29 个身体关节逐关节常偏 U(±2°)，再加 N(0, 0.25°) 噪声，裁剪在 ±3σ。**不做本体延迟，不做历史丢帧**（2026-11-12 起，所有模型对齐 GR00T A20 配方）。监督动作和实测手部标签保留原值。 |
| 评估 / rollout | 关闭所有随机扰动，保留相同的输入排列和归一化。 |
| 梯度累积 | 永远为 1。显存不够时加节点、降每卡 batch。 |
| 条数 | 按数据集实际条数全用。例外见第 5 节：B6 v5t7 用前 100 条；B6 v6 用前 500 条，但 GR00T/Psi-Zero 的纯 v6 版用 542 条；B7 黑衣服剪裁版用前 100/107 条。 |
| 训练中 MSE | 每个模型的每次训练都必须伴随训练测，每 250 步一次（第 6 节）。开训前必须在 `eval-action-mse/hb_eval_common.py` 的 `TASKS` 登记数据源，并建好 `build_cache.py <task> train10` 缓存。 |
| 上传 | 每次训练写进 HF 的新子目录，绝不覆盖。目录内容见第 8 节。 |

---

## 2. 随机种子（2026-11-23 逐个 checkpoint 实测）

**规则**
- 每个模型使用**其上游官方训练脚本的种子**，在所有任务、续训、补训、重训中一律不变。
- 新接入的模型同样用上游官方种子；上游没有规定时用 **42**。
- 训练中 MSE 的评估种子全模型统一为 **1234**。
- 多种子实验必须经用户批准，并在 protocol 里写明每个种子。

| 模型 | 训练种子 | 来源 / 设置位置 | 其他随机源 |
|---|---|---|---|
| GR00T N1.7 | **42** | 上游 `gr00t/configs/data/data_config.py: seed = 42` → `experiment.py: set_seed(config.data.seed)`；HF `TrainingArguments.seed = 42`、`data_seed = None`。所有 `groot-runs-2x32/*` 和 `groot-runs-ft512/*` 的 `training_args.bin` 都是 42。 | 数据采样：`trainer.py` 每步 `reset_seed(seed + global_step)`，是确定的。增强随机流：见第 3 节，**目前不确定**。 |
| FastWAM | **42** | 上游 `FastWAM/configs/train.yaml: seed: 42`；`fastwam/runs/*/out/config.yaml` 都是 42，续训也是 42。 | 推理适配器 `ad_fastwam.py`：10 步去噪，seed 42。增强随机流：**目前不确定**（第 3 节）。 |
| Psi-Zero | **292285** | 上游 Psi0 官方 finetune 脚本（`scripts/train/psi0/*.sh` 都写 `--seed=292285`）→ `hb_psi0/hb_args.py`。 | 本体常偏：`hb-bias-seed=292285`，用 `default_rng([seed, epoch, episode])`，是确定的。视觉增强随机流：**目前不确定**。上游 `set_seed` 启用了 `use_deterministic_algorithms(True, warn_only=True)` 和 `cudnn.benchmark=False`。 |
| ACT | **292285** | `hb_act/hb_act_args.py`，复用 Psi0 仓库的 LeRobot ACT 和同一套参数。 | 同 Psi-Zero。 |
| π0.5 | **42** | openpi `TrainConfig.seed = 42`；`hb_pi05/hb_g1_data.py` 的 `HB_AUG_SEED` 默认 42。 | 常偏：`default_rng([seed, epoch, episode])`，确定。增强流：`torch.initial_seed()` 加全局 rank，在世界规模和 worker 数相同时是确定的。 |
| DiT4DiT | **42** | `hb_dit4dit/hb_d4d_config.py: "seed": 42`；`HB_AUG_SEED` 默认 42。 | 数据顺序：`default_rng([seed, pass])` 打乱，可以精确续训。常偏确定。 |
| Delta Flow | **42** | `deltaflow-runs/<task>/run/production.yaml: seed: 42`。 | 增强种子：image_policy `20261110`、proprio_policy `20261106`；`split-manifest.json` 种子 42。 |
| 训练中 MSE | **1234** | `hb_groot/hb_inloop_eval.py` 和 `eval-action-mse/hb_inloop_core.py`：`SEED + 1000*rank + batch_idx`。 | 评估完恢复训练的 RNG 状态，不影响训练随机流。 |

> 各模型种子不同，是因为它们各自沿用上游官方值，这和「模型内部按官方配方」的原则一致。种子不改变 benchmark 接口，不影响公平性。
> **是否把所有模型统一成同一个种子由用户决定**。决定之前不得改动任何模型的种子：一改，同一模型已发布的 checkpoint 之间就不一致了。

---

## 3. 已知的不可复现来源（必须知道，部分待修）

1. **GR00T、Psi-Zero、FastWAM 的视觉增强随机流没有由种子决定（待修，修之前需用户批准）**
   - `hb_groot/hb_launch.py`：`default_rng([pid, urandom])`
   - `hb_psi0/hb_psi0_config.py: hb_rng()`：pid + urandom
   - `fastwam/tools/hb_fastwam_aug.py`：pid、样本下标、`time_ns`
   - 后果：同种子重训时，「每个样本选哪项增强、扰动多大」每次都不同。分布完全相同，结果只在统计上等价，逐位不同。GR00T 和 FastWAM 的关节常偏也由这个流产生。
   - 修法：改成从 `(训练种子, global rank, worker id, epoch, 样本下标)` 派生，参照 π0.5、DiT4DiT、Delta Flow 的做法。修好后，修改前训的模型在附录注明「增强随机流不可逐位复现」。
2. **GPU 计算不是逐位确定的**
   - 除 Psi0 的 `warn_only` 外，没有开 `cudnn.deterministic` / `use_deterministic_algorithms`。
   - bf16、NCCL 归约顺序、原子操作都会引入逐位差异。
   - 本 benchmark 的复现标准是**统计复现**：同数据、同配置、同种子、同布局时，训练中 nMSE 曲线和闭环成功率落在随机波动范围内。
3. **布局影响随机流和数据分片**
   - 每卡 batch、节点数、GPU 数、`num_workers` 都会改变采样顺序和增强流。
   - 复现时必须用 protocol 里记录的布局（例如 GR00T 2 节点 × 8 × 32，`num_workers` 8）。不得为了省卡改布局，除非全局 batch 不变，并在 protocol 里写明。
4. **已发布的 `protocol.json` 没有 `seed` 字段**
   - 种子目前只记在 `training_args.bin`、`train_args.txt`、`training_config.yaml` 里。
   - 以后上传必须在 protocol 里显式写 `seed`、`aug_seed`、代码 commit 和环境版本（第 8 节）。
   - 已发布目录是否补写由用户决定：补写会改动已发布文件。

---

## 4. 各模型当前配方（canonical）

`$G` = `…/preflight-20261104T1648Z/policies-20261109`（训练 NAS 根目录）。启动脚本均在本目录 `hb_*` 下有同步副本。

### 4.1 GR00T N1.7（全局 batch 512，4000 步）

- **代码与权重**
  - 代码：`third_party/Isaac-GR00T` @ `7aa0bca` + `hb_groot/hb_launch.py`
  - 基座：`$G/models/groot-n17-base`；VLM 是 `nvidia/Cosmos-Reason2-2B`
  - 环境：`envs/groot-py312`，Python 3.12.12、torch 2.9.0+cu128
- **启动器**
  - G1：`hb_groot/groot_pair2x32.sh`，配置 `g1_feedback76_h16.py`
  - H2：`groot_pair2x32_h2.sh`，配置 `h2_body31_hand44_h16.py`
  - H2 A14（动作 82）：`groot_quad16_h2p.sh`，配置 `h2_body31_hand44_planner7_h16.py`
- **布局**：2 节点 × 8 × 每卡 32 = 512，不累积。A14 是 4 节点 × 8 × 16。
- **全量微调**：`--tune_llm --tune_visual`，投影层、扩散动作头、VLLN 都训练。
- **学习率与步数**：lr 1e-4，`warmup_ratio` 0.05，之后 cosine 到 0，weight decay 1e-5；4000 步，C6 为 8000 步。
- **存档**：`save_steps` 500、`save_total_limit` 5，即第 2000/2500/3000/3500/4000 步。只有最后一个 checkpoint 保留续训状态。
- **数据加载**：`dataloader_num_workers` 8、`shard_size` 1024、`num_shards_per_epoch` 100000、`episode_sampling_rate` 0.1；`embodiment_tag` 为 `NEW_EMBODIMENT`；动作块 16。

### 4.2 Psi-Zero（全局 batch 512，4000 步）

- **代码与权重**
  - 代码：`third_party/Psi0` @ `cda3fdf` + `hb_psi0/`（`hb_args.py` 生成全部参数）
  - 基座：`models/psi0/psi0/postpre.sonic1.1.unifolm.2609181726.40k`；文本用 `clip-vit-large-patch14`
  - 环境：`envs/psi0`，Python 3.11.18、torch 2.7.0+cu126
- **启动器与布局**：`hb_b512/psi0_multi.sh <task> <nnodes> <rank> <master> <port> <per_gpu>`。2 节点 × 8 × 32；A14 为 4 × 8 × 16。
- **种子与精度**：`--seed=292285`，bf16，ddp。
- **学习率**：动作头 1e-4、`lang-backbone` 1e-6、`vision-tower` 1e-5、`mm-projector` 1e-4，`--model.tune-vlm`，开 gradient checkpointing。
- **优化**：cosine 调度，warmup 2.5%（4000 步时为 100 步），weight decay 1e-6，betas (0.95, 0.999)，`max_grad_norm` 1.0。
- **步数与存档**：4000 步，每 2000 步存一次，只保留 2 个。C6 是例外，也是 4000 步，见第 7 节。
- **动作与图像**：动作块 30（Psi0 原生），`observation-horizon` 1，flow 噪声调度，1000 个扩散训练步；图像 resize 并中心裁剪到 270×480。
- **增强与归一化**：原生 `img-aug`、`view-aug` 关闭，开 `--data.transform.model.hb-aug`；`state-noise-std` 0；动作归一化 `bounds`，state 做归一化。
- **本体延迟**：`hb-delay-prob` 必须为 0。启动前要 `unset HB_DELAY_PROB`，H2 的 0.15 例外已于 11-22 取消。
- **数据加载**：`num_workers` 8，`prefetch` 4。

### 4.3 FastWAM（全局 batch 512，4000 步 + 续训 2000 步 = 6000 步）

- **代码与权重**
  - 代码：上游 `yuantianyuan01/FastWAM` @ `7faa7110`（子模块 `third_party/FastWAM`）+ `fastwam_tools/`（`hb_fastwam_train.py`、`hb_fastwam_aug.py`、`hb_fastwam_convert.py`、`hb_fastwam_cont.py`）
  - 基座：Wan2.2-TI2V-5B（DiT、VAE、T5）和 umt5-xxl tokenizer；ActionDiT 由 `preprocess_action_dit_backbone.py` 初始化（`action-init-<g1|h2|h2p>.pt`）
  - 环境：`envs/fastwam`，Python 3.11.18、torch 2.7.1+cu128
- **启动器**：主训练 `hb_fastwam/fastwam_task.sh`；续训 `fastwam_cont.sh`，C6 用 `fastwam_cont4k.sh`。
- **布局**：4 节点 × 8 × 每卡 16 = 512，每卡 16 是 A800 80 GB 放得下的最大值（`chosen.json`）；accelerate + DeepSpeed ZeRO-1，bf16，不累积。
- **主训练**：lr 1e-4 cosine，weight decay 0.01，`max_grad_norm` 1.0，seed 42，`num_workers` 8；第 2000、4000 步存档。
- **续训**：只加载权重（DiT 和本体编码器，经 `hb_fastwam_cont.py`），AdamW 重新开始；峰值 lr 3e-5（原峰值 × 0.3），warmup 5%，cosine 降到 1%；2000 步，在第 5000、6000 步存档。C6 续训 4000 步，到第 12000 步。
- **视频与动作**：17 帧片段，两路相机并排成 224×448；动作块 16；T5 文本嵌入预先算好。

### 4.4 其他模型（当前状态，开训前需按本规范补齐）

| 模型 | 全局 batch | 步数 | 存档 | 关键配置 | 训练中 MSE |
|---|---|---|---|---|---|
| π0.5 | 256 | 4000 | 2000 / 4000 | openpi @ `215abfb`；动作块 16，图像 224 | **未接入，开训前必须接好** |
| DiT4DiT | 64 | 8000 | 6000 / 8000 | DiT4DiT @ `66a6f3a`；lr：backbone_interface 1e-5 / action_model 1e-4 / base 3e-5，`cosine_with_min_lr`（最小 5e-7），AdamW (0.9, 0.95)，warmup 5%；关节角做 sin/cos（G1 73 维 / H2 153 维） | **未接入，开训前必须接好** |
| Delta Flow | 256（每卡 8 × 32） | 8000（C6 为 16000） | 6000 / 8000 | lr 见 `production.yaml`，warmup 5%，cosine；增强种子 20261110 / 20261106 | 已接入 |
| ACT | 256 | 4000 | 2000 / 4000 | Psi0 自带的 LeRobot ACT，复用 Psi-Zero 的数据视图和增强 v2 | 由合作者负责，接入情况开训前核对 |

---

## 5. 数据版本（每个任务当前最新版，开训前逐项核对）

本地视图目录名里带 HF commit 前缀；HF 仓库都在 `humanoid-benchmark` 组织下，均为私有。

| 任务键 | 任务 | HF 数据 @ commit | 条数 | 机器人 |
|---|---|---|---|---|
| a3v4 | A3 开冰箱门 v4 | @`6393d4a` | 全部 | G1 |
| a7v4 | A7 关橱柜门 v4 | @`90cc2b5` | 全部 | G1 |
| a10v3 | A10 推小车 v3 | @`6e410c5` | 全部 | G1 |
| basketv2 | A20 篮子转移，伸手前剪裁 v2（`a20-basket-transfer-pre-reach-trim-expert-v2`） | @`c4df4d4` | 100 | G1 |
| b1 | B1 饭团进微波炉（`b1-riceball-microwave-expert-v1`） | @`36da298` | 115 | G1 |
| b5 | B5 推椅子（`b5-chair-push-expert-v1`） | @`b85b7e9` | 100 | G1 |
| b5ss | B5 standing start，现名 A22（`A22-chair-push-standing-start-expert-v1-formerly-b5`） | @`913f5a1` | 100 | G1 |
| b6v6 | B6 丢垃圾 v6（`b6-pedal-bin` 根目录） | @`ad2f5bf0` | 前 500 条（0–499）；GR00T/Psi-Zero 纯 v6 用 542 条 | G1 |
| b7t | B7 黑衣服，剪裁版（`b7-black-clothes-hamper`） | @`949da46` | 前 100 / 107 | G1 |
| b7gt | B7 绿衣服，剪裁 v2（`b7-green-clothes-hamper` 根目录） | @`22e3b83` | 421 | G1 |
| b8 | B8 擦桌子（`b8-table-wipe-expert-v1`） | @`3b2e618` | 全部 | G1 |
| c6 | C6 浴室开门取物（`c6-bathroom-door-fetch-expert-v1`） | @`3afba69` | 全部 | G1 |
| a14cr3 | A14 拧灯泡，cropped-v3（`a14-screw-bulb` 根目录） | @`79fe08a` | 全部 | H2，动作 82 |
| a17cr2 | A17 拧瓶盖，cropped-v2（`a17-unscrew-bottle` 根目录） | @`3520b70f` | 100 | H2 |
| a18cr2 | A18 倒水，cropped-v2（`a18-pour-water` 根目录） | @`44cdd571` | 100 | H2 |

- 新任务（A9、A29、B4 v2 等）开训前必须逐项核对字段、维度、条数和帧率。全部任务都是 30 Hz，**A29 是 50 Hz**，处理方式要先问用户。
- 核对通过后同时登记三处：各模型的 `prepare_*` 注册表、`hb_eval_common.TASKS`、上传脚本。

---

## 6. 训练中 MSE 协议

- 每 250 步（`HB_INLOOP_EVERY`）用开环动作还原 nMSE 评估一次。
- 评估集：任务有从不参与训练的 heldout 条目时用 `heldout`，否则用 `train10`（均匀抽 10 条训练集 episode）。
- 每次前向 8 个查询；种子 `1234 + 1000*rank + batch_idx`；GR00T 用 4 步去噪。评估完恢复训练的 RNG 状态。
- 曲线写在 `<run>/logs/inloop-mse.jsonl`（Psi-Zero 写在 `<project_dir>/inloop-mse.jsonl`），并随 checkpoint 上传。
- 开训后必须在日志里看到 `HB_INLOOP ... ready` 或 `installed`，并且没有 `disabled`。
- 任务键按数据源路径匹配：新任务键复用同一份数据时会自动匹配。

---

## 7. 例外与偏离登记表（论文附录必须列出）

| 模型 / 任务 | 偏离 | 决定 |
|---|---|---|
| GR00T C6（`c6-bathroom-door-fetch/b512`） | 11-13 旧配置：4 节点 × 每卡 8 × **梯度累积 2**，8000 步 | 11-14 用户：先不重训 |
| GR00T basket v1、B7 黑 / 绿 v1（`groot-runs-ft512`） | 同上，4 × 8 × 累积 2 | 已有按新配置训的新数据版本，旧目录保留 |
| GR00T / Psi-Zero basket v2 | HF 上已有的版本是从 v1 checkpoint 续训 2000 步（`b512-trimv2-from-b512`、`b512-trimv2`），不是在 v2 上从 base 训的 | 11-23 补训纯 v2，任务键 `basketv2p`，上传 `b512-trimv2-pure/` |
| Psi-Zero C6 | 4000 步，没有按长任务翻倍 | 11-13 用户：不重训 |
| Psi-Zero A14 cr3（`cropped-v3-b512`）、旧 A17/A18 | 带本体延迟 0.15（原 H2 指南例外） | 11-22 取消；重训为 `a14cr3a`（`cropped-v3-b512-nodelay`）和 `b512-2cam-cropped-v2` |
| π0.5、Psi-Zero、ACT 在 11-12 前训的 | 15% 本体延迟 + 15% 历史丢帧 | 结果保留，附录说明；受影响任务按对齐配置重训 |
| B6 v6 条数 | FastWAM 和 Delta Flow 用前 500 条；GR00T / Psi-Zero 纯 v6（`b512-v6-pure`）用全部 542 条；早期的 v6 续训版也是 542 条 | 11-18 与 11-23 用户 |
| Delta Flow A17 cr2（`a17cr2lg`） | 沿用 legacy a18 e40 配方（旧增强 `b6-visual-aug-v2`，无本体增强） | 11-22 用户：Delta Flow 先不管 |
| Delta Flow B6 v6 | 从 v5 `step_010200` 只加载权重续训，不是从 base 训 | 11-15 计划 |
| FastWAM B5 standing start | 不训 | 用户 |
| 多个早期运行 | **没有训练中 MSE**：2026-11-15 训练中评估机制上线前训的（如 GR00T C6、Psi-Zero C6），以及任务键当时没登记、评估自动关闭的（GR00T A14 cr3、FastWAM B7 绿衣服 4000 步及续训、FastWAM A14 cr3 4000 步，Psi-Zero A14 cr3 只有 1 个点） | 只有事后补测的开环 MSE；11-22 起不再允许，新训练必须伴随训练测 |
| 种子 | Psi-Zero / ACT 是 292285，其余模型是 42（各用上游官方值） | 待用户决定是否统一 |
| 增强随机流 | GR00T、Psi-Zero、FastWAM 由 pid、urandom、时间产生，不可逐位复现（第 3 节） | 待修 |

新例外追加到本表，同时写进该运行的 `protocol.json` 的 `exceptions` 字段。

---

## 8. 上传到 HF 的内容（模型仓库公开，数据集私有）

- **路径**：`humanoid-benchmark/<model>/<task_dir>/<subdir>/step-XXXXXX/`。每次训练用新的 `<subdir>`，例如 `b512`、`b512-v6-pure`、`b512-cont2k`，**绝不覆盖已有目录**。
- **必须包含**
  - 权重
  - `protocol.json`：模型、任务、数据 repo@commit 与条数、增强描述、步数、存档点、全局 batch、每卡 batch、GPU 数、学习率与调度、续训来源
  - 完整训练配置（GR00T `experiment_cfg/` + `training_args.bin`；Psi-Zero `run_config.json`、`argv.txt`、`train_args.txt`；FastWAM `training_config.yaml`）
  - 归一化统计
  - 训练日志和训练中 MSE
  - 所用适配代码快照 `code/`
  - 推理说明
  - 续训状态（优化器等）**不上传**
- **今后 `protocol.json` 还必须加上**（第 3 节第 4 条）
  - `seed`、`aug_seed`、`bias_seed`
  - 本仓库 commit
  - Python/torch/CUDA 版本
  - 节点 × GPU × 每卡 batch 布局
  - `exceptions`

---

## 9. 开训前检查清单（每次都做）

1. **数据**：HF repo@commit、条数规则、fps、相机、state 和 action 维度都与第 5 节一致。新数据先逐项核对。
2. **实际取一个 batch**：确认原始本体是 44 / 78 维、不含速度；编码后的维度符合模型原生配方；增强日志里出现 `HB_AUG` / `HB_FW_AUG`，并带 `boost=`。
3. **配方**：种子、全局 batch、梯度累积 = 1、步数、学习率曲线、存档点都与第 2、4 节和 `protocol.yaml` 一致。Psi-Zero 要确认 `HB_DELAY_PROB` 未设置。
4. **训练中 MSE**：`hb_eval_common.TASKS` 已登记、`train10` 缓存已建好；开训后日志有 `HB_INLOOP` 且是启用状态。
5. **上传**：新子目录已在自动上传里登记，不与已有目录重名。
6. **训完**：运行 `check_protocol.py`（下面的命令），全部 PASS 或只剩已登记的例外，才算交付。
7. **偏离**：写进第 7 节、运行目录和 `protocol.json`。

```bash
python3 hb_training/check_protocol.py groot  $G/groot-runs-2x32/<task>
python3 hb_training/check_protocol.py psi0   $G/psi0-runs/<task>/b512
python3 hb_training/check_protocol.py fastwam $G/fastwam/runs/<task>[-cont2k]
```

---

## 10. 环境

- 各训练环境的完整包版本快照在 `hb_training/envs/<env>.txt`（2026-11-23 导出）。重建环境时按快照装，不要装最新版。
- 硬件：阿里云 PAI DLC，4 节点 × 8 张 A800 80 GB，NAS 共享 `$G`。
- 上游代码版本以本仓库 `.gitmodules` 和子模块 commit 为准：

| 子模块 | commit |
|---|---|
| Isaac-GR00T | `7aa0bca` |
| Psi0 | `cda3fdf` |
| FastWAM | `7faa7110` |
| openpi | `215abfb` |
| DiT4DiT | `66a6f3a` |
| GR00T-WholeBodyControl | `1e1ec4f` |
