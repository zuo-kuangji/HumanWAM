# Humanoid Benchmark 训练可复现规范（权威版）

> **所有在本仓库安排 Humanoid Benchmark 训练的人和 agent，开训前必须逐项对照本文件和 [`protocol.yaml`](protocol.yaml)，训完上传前运行 [`check_protocol.py`](check_protocol.py)。**
> 本文件取代 `hb_training/README.md`「训练设置」一节中 2026-11-10 的旧设置（20 epoch、batch 696 等，均已废止）。
> 数据增强的细节以项目规范 v2 为准：[`docs/augmentation_v2.md`](docs/augmentation_v2.md)；H2 任务另见 [`docs/h2/`](docs/h2/)。
> 偏离本文件的唯一合法情形：用户明确声明的例外。例外必须同时写进运行目录、HF 上的 `protocol.json` 和第 7 节的登记表。

最后核对：2026-11-23（2026-11-25 增补 ODE 和 hb_baselines 增强种子）。核对依据是集群上实际训练产物：GR00T 的 `checkpoint-*/training_args.bin`、Psi-Zero 的 `train_args.txt`、FastWAM 的 `out/config.yaml`、HF 上已发布的 `protocol.json`，不是凭配置文件推测。

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
| 存档（2026-11-27 用户） | **小模型**（ACT、FLASH、FM-DiT、DP、ODE 等 hb_baselines 模型）：每次训练在最大步数上均匀存 6 个 checkpoint（第 k × 步数 / 6 步，k = 1..6；hb_baselines 取整到训练中 MSE 的 250 步网格，最后一个就是训练结束），例如 30000 步每 5000 步一个、4000 步为 750/1250/2000/2750/3250/4000。**大模型**（GR00T、Psi-Zero、FastWAM、Delta Flow，以及 π0.5、DiT4DiT 这类更大的）：除训练结束的 checkpoint 外最多再存一个中间点，选在总步数约 80% 处（4000 步存 3200 和 4000；C6 8000 步存 6400 和 8000；FastWAM 续训 2000 步存总步数 5600 和 6000）。只对之后开训的运行生效，正在跑和已训完的保留原存档。 |
| 条数 | 按数据集实际条数全用。B6 v6 用全部 542 条（2026-11-24 用户，取代 11-18 的「只取前 500 条」）。例外见第 5 节：B6 v5t7 用前 100 条；B7 黑衣服剪裁版用前 100/107 条；11-24 以前用 B6 v6 前 500 条训完的 FastWAM、Delta Flow、FLASH 见第 7 节。 |
| 训练中 MSE | 每个模型的每次训练都必须伴随训练测，每 250 步一次（第 6 节）。开训前必须在 `eval-action-mse/hb_eval_common.py` 的 `TASKS` 登记数据源，并建好 `build_cache.py <task> train10` 缓存。 |
| 上传 | 每次训练写进 HF 的新子目录，绝不覆盖。目录内容见第 8 节。 |

---

## 2. 随机种子（2026-11-23 逐个 checkpoint 实测）

**规则**（2026-11-27 用户定：「42，并且记入项目记忆，规定好以后全都用统一的42，其他模型的每次训练也都要在模型内部保持统一，如果已经没对齐的那些也不必强求，只要记录好用的都是什么种子就可以了，核心目标是确保整个benchmark上所有的训练和评估都严格的可复现。」）
- **每个模型只有一个训练种子**，在所有任务、续训、补训、重训中一律不变（模型内部统一）。
- **FM-DiT（v1 和 v2）以后所有训练都用 42**（v1 从 2026-11-27 起；v1 已发布的种子 0 目录和之前的运行保留，只记录）。
- 新接入的模型用 **42**；上游官方脚本固定了别的种子时用上游值，之后同样不变。
- 已经用别的种子训完的运行**不强求重训**，但必须记录它用的种子（本表、第 7 节、该运行的 `config.json` / `protocol.json`）。核心目标：整个 benchmark 上所有训练和评估都严格可复现。
- 训练中 MSE 的评估种子全模型统一为 **1234**。
- 多种子实验必须经用户批准，并在 protocol 里写明每个种子。

| 模型 | 训练种子 | 来源 / 设置位置 | 其他随机源 |
|---|---|---|---|
| GR00T N1.7 | **42** | 上游 `gr00t/configs/data/data_config.py: seed = 42` → `experiment.py: set_seed(config.data.seed)`；HF `TrainingArguments.seed = 42`、`data_seed = None`。所有 `groot-runs-2x32/*` 和 `groot-runs-ft512/*` 的 `training_args.bin` 都是 42。 | 数据采样：`trainer.py` 每步 `reset_seed(seed + global_step)`，是确定的。增强随机流：见第 3 节，11-23 21:05 起由 `HB_AUG_SEED`（= 训练种子）决定，之前启动的训练不确定。 |
| FastWAM | **42** | 上游 `FastWAM/configs/train.yaml: seed: 42`；`fastwam/runs/*/out/config.yaml` 都是 42，续训也是 42。 | 推理适配器 `ad_fastwam.py`：10 步去噪，seed 42。增强随机流：11-23 21:05 起由 `HB_AUG_SEED`（= 训练种子）决定，之前启动的训练不确定（第 3 节）。 |
| Psi-Zero | **292285** | 上游 Psi0 官方 finetune 脚本（`scripts/train/psi0/*.sh` 都写 `--seed=292285`）→ `hb_psi0/hb_args.py`。 | 本体常偏：`hb-bias-seed=292285`，用 `default_rng([seed, epoch, episode])`，是确定的。视觉增强随机流：11-23 21:05 起由 `HB_AUG_SEED`（= 训练种子）决定，之前启动的训练不确定。上游 `set_seed` 启用了 `use_deterministic_algorithms(True, warn_only=True)` 和 `cudnn.benchmark=False`。 |
| ACT（Psi0 版，**2026-11-24 已退役**：Psi0 仓库内的 LeRobot ACT，只对应 HF 旧目录 `act/<task>/epoch-NNN`） | **292285** | `hb_act/hb_act_args.py`，复用 Psi0 仓库的 LeRobot ACT 和同一套参数。 | 同 Psi-Zero。 |
| ACT（hb_baselines 版，**现行且唯一**，HF `act/<task>/b512*`）、DP；FM-DiT v1 已发布的目录 | **0**（FM-DiT v1 自 2026-11-27 起为 **42**，与 v2 相同；hb_baselines `train.py` 的默认值） | 合作者的 hb_baselines 代码；种子记在每个 HF 目录 `config.json` 的 `training.seed`。2026-11-24 逐个核对：act 的 12 个、fm-dit 全部 68 个、dp 1 个目录都是 0。 | 增强按规范 v2（`config.json` 的 `training.augmentation`，实现 `evaluation/augmentation/hb_aug_v2.py`）；增强随机流：Humanoid_benchmark bai `b4b94d29`（2026-11-25 用户）起，所有 hb_baselines 模型都按 FLASH 的做法，每个 DataLoader worker 用 `default_rng([HB_AUG_SEED = 训练种子, torch.initial_seed(), worker id])`，日志打印 `HB_AUG_SEED=<seed>`，`config.json` 记 `training.aug_seed`；之前的运行用 pid + urandom，不可逐位复现。代码版本记在 `config.json` 的 `code_commit`（部分目录缺）。 |
| π0.5 | **42** | openpi `TrainConfig.seed = 42`；`hb_pi05/hb_g1_data.py` 的 `HB_AUG_SEED` 默认 42。 | 常偏：`default_rng([seed, epoch, episode])`，确定。增强流：`torch.initial_seed()` 加全局 rank，在世界规模和 worker 数相同时是确定的。 |
| DiT4DiT | **42** | `hb_dit4dit/hb_d4d_config.py: "seed": 42`；`HB_AUG_SEED` 默认 42。 | 数据顺序：`default_rng([seed, pass])` 打乱，可以精确续训。常偏确定。 |
| Delta Flow | **42** | `deltaflow-runs/<task>/run/production.yaml: seed: 42`。 | 增强种子：image_policy `20261110`、proprio_policy `20261106`；`split-manifest.json` 种子 42。 |
| FLASH | **42** | 上游 NTUMARS/FLASH-Policy @ `f47e490` 的 `roboverse_learn/il/configs/train_config/default_train.yaml`：`training_params.seed`；本项目的移植在 hb_baselines（Humanoid_benchmark bai `9cca6255`），见 4.6。 | 增强随机流：每个 DataLoader worker 用 `default_rng([HB_AUG_SEED=42, torch.initial_seed(), worker id])`，日志打印 `HB_AUG_SEED=42`。 |
| ODE-MLP、ODE-DiT | **42** | 上游 NTUMARS/ODE-POLICY 的 `ode_policy/configs/train_config/default_train.yaml`：`training_params.seed`（与 FLASH 同一套训练代码）；本项目的移植在 hb_baselines（Humanoid_benchmark bai `b4b94d29`），见 4.7。 | 增强随机流：每个 DataLoader worker 用 `default_rng([HB_AUG_SEED=42, torch.initial_seed(), worker id])`，日志打印 `HB_AUG_SEED=42`。 |
| 训练中 MSE | **1234** | `hb_groot/hb_inloop_eval.py` 和 `eval-action-mse/hb_inloop_core.py`：`SEED + 1000*rank + batch_idx`。 | 评估完恢复训练的 RNG 状态，不影响训练随机流。 |

> 各模型种子不同，是因为它们各自沿用上游官方值，这和「模型内部按官方配方」的原则一致。种子不改变 benchmark 接口，不影响公平性。
> 2026-11-27 用户已定：**不跨模型统一**，每个模型内部统一一个种子；FM-DiT 用 42；新模型用 42。各模型现行种子见上表，以后不得改动：一改，同一模型的 checkpoint 之间就不一致了。

---

## 3. 已知的不可复现来源（必须知道，部分待修）

1. **GR00T、Psi-Zero、FastWAM 的视觉增强随机流（2026-11-23 21:05 北京时间起已修复）**
   - `hb_groot/hb_launch.py`：`default_rng([pid, urandom])`
   - `hb_psi0/hb_psi0_config.py: hb_rng()`：pid + urandom
   - `fastwam/tools/hb_fastwam_aug.py`：pid、样本下标、`time_ns`
   - 修复前的后果：同种子重训时，「每个样本选哪项增强、扰动多大」每次都不同。分布完全相同，结果只在统计上等价，逐位不同。GR00T 和 FastWAM 的关节常偏也由这个流产生。
   - 已修（用户 11-23 批准）：训练入口 `hb_groot/hb_launch.py`、`hb_psi0/hb_train.py`、`fastwam_tools/hb_fastwam_train.py` 把 `HB_AUG_SEED` 设为训练种子，并在日志打印 `HB_AUG_SEED=<seed>`；增强随机流改由 `(HB_AUG_SEED, 进程或 DataLoader worker 的 torch.initial_seed(), global rank, worker id)` 派生，FastWAM 另加样本下标和进程内计数。CPU 实测三个模型同种子两次运行逐位一致，不同 worker 的流不同。
   - 只对 11-23 21:05（13:05 UTC）之后启动的训练生效，分布不变。之前训的模型（包括已发布的全部 GR00T、Psi-Zero、FastWAM）在 `protocol.json` 的 `reproducibility.augmentation_random_stream` 里标为 not seeded，附录注明。
1b. **hb_baselines（ACT、FM-DiT、DP、ODE）的增强随机流（2026-11-25 已修复）**
   - 修复前：`evaluation/baselines/data.py` 的 `ChunkDataset` 不传随机数生成器，增强用 `evaluation/augmentation` 的 `worker_rng()`（pid + urandom）；只有 FLASH 的 `FlashChunkDataset` 按种子派生。
   - 已修（用户 11-25）：Humanoid_benchmark bai `b4b94d29` 起，FLASH 的派生方式原样移到基类 `ChunkDataset`，`train.py` 对所有模型设 `aug_seed` = 训练种子，日志打印 `HB_AUG_SEED=<seed>`，`config.json` 记 `training.aug_seed`。CPU 测试：同种子、同 worker 数两次运行逐位一致，换种子则不同，动作标签不变（`tests/framework/test_baselines.py::test_augmentation_stream_is_seeded_and_reproducible`）。
   - 只对之后启动的训练生效，分布不变。之前训的 ACT `b512*`、FM-DiT、DP 保留，`protocol.json` 里标为 not seeded（`hb_repro.py` 按 `config.json` 有没有 `training.aug_seed` 自动判断）。
1c. **hb_baselines 的采样顺序、增强流和中断续跑（2026-11-27 已修，Humanoid_benchmark bai `e4809ea9`）**
   - 修复前：增强流按 `(aug_seed, worker 的 torch.initial_seed(), worker id)` 派生，结果依赖 `num_workers`；`--resume` 不存随机状态，中断后续跑时采样顺序、增强和模型噪声都从头重放。
   - 已修（用户 11-27「全部修」）：采样顺序由 `EpochSampler` 决定（第 e 轮是 `default_rng([seed, 1, e])` 的排列，整 batch）；每个样本用 `default_rng([aug_seed, draw])` 增强，draw 是它在整个运行样本流里的序号；DataLoader 有自己的生成器；`train_state.pt` 保存 torch / CUDA / numpy / python 的随机状态。CPU 实测：1 个和 2 个 worker 的 loss 逐步相同；第 10 步后中断再 `--resume`，每步 loss 与不中断的相同，最终 EMA 权重最大差 0.0。
   - 只对之后开训的运行生效。之前的运行（已固定种子的那批）分布相同、逐位不同，保留。
2. **GPU 计算不是逐位确定的**
   - 除 Psi0 的 `warn_only` 外，没有开 `cudnn.deterministic` / `use_deterministic_algorithms`。
   - bf16、NCCL 归约顺序、原子操作都会引入逐位差异。
   - 本 benchmark 的复现标准是**统计复现**：同数据、同配置、同种子、同布局时，训练中 nMSE 曲线和闭环成功率落在随机波动范围内。
3. **布局影响随机流和数据分片**
   - 每卡 batch、节点数、GPU 数、`num_workers` 都会改变采样顺序和增强流。
   - 复现时必须用 protocol 里记录的布局（例如 GR00T 2 节点 × 8 × 32，`num_workers` 8）。不得为了省卡改布局，除非全局 batch 不变，并在 protocol 里写明。
4. **`protocol.json` 的种子与复现字段（2026-11-23 起自动写入，已发布的已补写）**
   - 自动上传调用 `hf-sidecar-tools-20261110/hb_repro.py`（仓库同步在 `hb_training/hf_tools/`），给每个 `protocol.json` 加顶层 `seed` 和 `reproducibility` 块：训练种子及来源、增强随机流是否由种子决定（看运行日志里的 `HB_AUG_SEED=` 行）、训练中 MSE 种子、布局、环境版本、硬件、上游代码、规范链接和 commit。
   - 已发布的目录用 `hb_repro_backfill.py` 补写：只加字段，不改其他内容；种子尽量从该目录的运行文件读出并标明是否核实；`legacy/` 不动。
   - 2026-11-24 更正：act 仓库里 12 个 hb_baselines 目录（`b512*`），第一次补写误按 Psi0 版 ACT 填了种子来源、环境、硬件和上游代码，已用 `hb_repro_fix_hbbaselines.py` 按各目录 `config.json` 改正（种子 0）。之后的自动上传也从运行文件读种子（`training_args.bin`、`argv.txt` / `train_args.txt`、`training_config.yaml`、`config.json`），`training_seed_verified_from_run_files` 为 true。

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
- **存档**：`save_steps` 500、`save_total_limit` 5，即第 2000/2500/3000/3500/4000 步。只有最后一个 checkpoint 保留续训状态。2026-11-27 起新开的训练按第 1 节的大模型规则只存 3200 和 4000（C6 6400 和 8000），启动脚本 `hb_groot/groot_pair2x32_r80.sh`、`groot_pair2x32_h2_r80.sh`、`groot_quad16_h2p_r80.sh`（`save_steps` 3200、`save_total_limit` 2，HF Trainer 在最后一步必存）。
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
- **步数与存档**：4000 步，每 2000 步存一次，只保留 2 个。C6 是例外，也是 4000 步，见第 7 节。2026-11-27 起新开的训练按第 1 节的大模型规则存 3200 和 4000：设环境变量 `HB_CKPT_SAVE=3200`（`hb_psi0/hb_args.py`，`checkpointing_steps` 3200，训练器在最后一步必存）。
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
- **主训练**：lr 1e-4 cosine，weight decay 0.01，`max_grad_norm` 1.0，seed 42，`num_workers` 8；第 2000、4000 步存档（2026-11-27 起新开的训练按第 1 节存 3200 和 4000：`hb_fastwam/fastwam_task_r80.sh`，`save_every` 3200，训练器在最后一步必存）。
- **续训**：只加载权重（DiT 和本体编码器，经 `hb_fastwam_cont.py`），AdamW 重新开始；峰值 lr 3e-5（原峰值 × 0.3），warmup 5%，cosine 降到 1%；2000 步，在第 5000、6000 步存档（2026-11-27 起新开的续训存 5600 和 6000：`hb_fastwam/fastwam_cont_r80.sh`，`save_every` 1600）。C6 续训 4000 步，到第 12000 步。
- **hb_baselines 的续训**（2026-11-27，同一规则）：`train.py --init-from <run>/checkpoints/step-N`（Humanoid_benchmark bai `c926770a` 起）只加载权重（原 run 的 `train_state.pt` 写于该步时取原始权重和 EMA，否则两者都取存档的 EMA），AdamW 和调度重新开始，峰值默认原峰值 × 0.3、warmup 默认续训步数的 5%，按模型自己的 cosine 降到 0；续训 2000 步（长任务 4000）；其他值要 `--continuation-exception` 登记用户的决定，峰值不低于原峰值时拒绝。checkpoint、日志、训练中 MSE、EMA 衰减都按总步数计，结束写 `complete.json`（`continued_from`、`continued_from_step`、`total_steps_after`）。
- **视频与动作**：17 帧片段，两路相机并排成 224×448；动作块 16；T5 文本嵌入预先算好。

### 4.4 其他模型（当前状态，开训前需按本规范补齐）

| 模型 | 全局 batch | 步数 | 存档 | 关键配置 | 训练中 MSE |
|---|---|---|---|---|---|
| π0.5 | 256 | 4000 | 2000 / 4000 | openpi @ `215abfb`；动作块 16，图像 224 | **未接入，开训前必须接好** |
| DiT4DiT | 64 | 8000 | 6000 / 8000 | DiT4DiT @ `66a6f3a`；lr：backbone_interface 1e-5 / action_model 1e-4 / base 3e-5，`cosine_with_min_lr`（最小 5e-7），AdamW (0.9, 0.95)，warmup 5%；关节角做 sin/cos（G1 73 维 / H2 153 维） | **未接入，开训前必须接好** |
| Delta Flow | 256（每卡 8 × 32） | 8000（C6 为 16000） | 6000 / 8000 | lr 见 `production.yaml`，warmup 5%，cosine；增强种子 20261110 / 20261106 | 已接入 |
| ACT（hb_baselines 版，现行且唯一，见 4.5） | 512（A10 v3 为 256） | 4000（A10 v3 为 8000） | 按目录 | 合作者的 hb_baselines：种子 0，lr 1e-4，AdamW (0.95, 0.999)，wd 1e-6，warmup 200 步 + cosine，backbone lr × 0.1，EMA 0.9999，AMP；完整配置在每个 HF 目录的 `config.json` | 无（`config.json` 里有训练后的 `offline_eval`） |
| ACT（Psi0 版，早期 `epoch-010/020`，**2026-11-24 退役**） | 256 | 20 轮 | 第 10 / 20 轮 | Psi0 自带的 LeRobot ACT（`hb_training/hb_act/`），复用 Psi-Zero 的数据视图和增强 v2；种子 292285。不再用于任何训练 | 无 |
| FM-DiT、DP | 512（FM-DiT 续训目录为 64） | 4000 | 按目录 | 合作者的 hb_baselines，种子 0，配置在每个 HF 目录的 `config.json`；HF 仓库 `fm-dit`、`dp` 归合作者，没有补写 `protocol.json` 的复现字段 | 由合作者负责 |
| FM-DiT v2（`--policy fm_dit_v2`，见 4.8） | 512 | 12000 | 6 个均匀存档点 | FLASH-Policy 自带的 FM-DiT，种子 42，lr 1e-4，500 步预热后 diffusers cosine，EMAModel | 必须（train.py 强制） |

**hb_baselines 按本规范强制（2026-11-27，Humanoid_benchmark bai `e4809ea9` 起）**：`train.py` 的 batch、步数（长任务 × 2）、warmup、`num_workers`、种子默认就是 `protocol.yaml` 的配方（`PROTOCOL_RECIPES`：act 512/4000/200/32、flash 512/12000/500/40、fm_dit_v2 512/12000/500/32、ode 512/5000/500/32；FM-DiT v1、DP 只核对 batch 512）；其他值、`--save-every`、非 30 Hz 数据都会被拒绝，除非 `--protocol-exception` 登记用户的决定。所有模型（含 ACT、FM-DiT v1、DP）都必须带训练中 MSE；`evaluation/` 下有未提交改动时拒绝开训（`--allow-dirty-tree` 例外）。`config.json` 和每个 checkpoint 的 `protocol.json` 记录任务键、数据集 repo@commit（`--dataset-id`，或从 `<name>-<commit>/data` 目录名推出）、存档点、环境版本（Python、torch、torchvision、numpy、Pillow、CUDA、cuDNN、GPU、驱动）、`hb_aug_v2.py` 和训练中 MSE 代码的 sha256、偏离项。

### 4.5 ACT 只用 hb_baselines（2026-11-24 用户）

- 本项目以后所有 ACT 训练（新任务、补训、重训、续训）一律用合作者的 hb_baselines 代码：GitHub `zuo-kuangji/Humanoid_benchmark` 的 `evaluation/baselines/train.py --policy act --contract <contract>`，增强 `evaluation/augmentation/hb_aug_v2.py`，环境 `evaluation/baselines/requirements.txt`（Python 3.12、torch 2.9.0+cu128）。已发布的 ACT 用过的 commit 是 `4385573e`（B6）和 `028df1bb`（A10 v3、A20 basket v2）。
- 配置以 HF `humanoid-benchmark/act` 现有 `b512*` 目录的 `config.json` 为准：种子 0，全局 batch 512，4000 步，lr 1e-4，AdamW (0.95, 0.999)，wd 1e-6，warmup 200 步 + cosine，backbone lr × 0.1，EMA 0.9999，AMP，增强 v2，图像 168 × 224。上传沿用 `hb_baselines/1` 格式和仓库里的 `code/hb_baselines_v1/` 加载器。
- **2026-11-27 起按本规范强制**（Humanoid_benchmark bai `e4809ea9`，见 4.4 末段）：`--policy act` 不写参数就是 512 / 4000 步 / warmup 200 / `num_workers` 32 / 种子 0，必须带训练中 MSE，存 6 个均匀存档点（4000 步：750、1250、2000、2750、3250、4000），上传仍只传最后两个。
- 训练中 MSE：已发布的 ACT 只有训练后的 `offline_eval`。Humanoid_benchmark bai `9cca6255` 起 hb_baselines 有训练中 MSE 接口 `evaluation/baselines/inloop.py`（加 `--inloop-task`，FLASH 已在用，FM-DiT、DP、ACT 都能用）。下次训 ACT 用它按第 6 节接上。
- 11-24 晚第一次按本节训 ACT：B6 v6 全部 542 条（任务键 b6v6p，训练中 MSE 用 b6v6 train10），Humanoid_benchmark bai `8c3b88d2`，阿里云 DSW 2 号卡，4000 步，存 2000、4000；训练中 MSE 从 0.2572（250 步）降到 0.0673（4000 步）。上传 `act/b6-pedal-bin/b512-v6-pure/`。
- 11-25 起 ACT 用种子固定的增强流（Humanoid_benchmark bai `b4b94d29` 起，第 3 节 1b），日志里有 `HB_AUG_SEED=0`，`config.json` 里有 `training.aug_seed = 0`。下面两个是按本节训的合规样板。都在 Humanoid_benchmark bai `2c91a514` 的工作树上训，阿里云 DSW 1 × RTX 4090 46 GB，num_workers 32，prefetch 2，nice 19（同机在跑闭环评估），存 2000、4000 步，`protocol.json` 的 protocol_commit 是 `6814d6c`：
  - B1 v2（任务键 b1v2）：`b1-riceball-microwave-expert-v2-30hz` @`b58f0434`，全部 200 条，30 Hz。训练中 MSE（b1v2 train10）2000 步 0.0852，4000 步 0.0731。HF `act/b1-riceball-microwave/b512-v2/`，act @`b0bcea4f`。
  - B4（任务键 b4）：`b4-cushion` 根目录 v2 @`194e7cde`，全部 302 条，30 Hz。训练中 MSE（b4 train10）2000 步 0.0564，4000 步 0.0497。HF `act/b4-cushion/b512/`，act @`cff510cf`。
  - A29（任务键 a29）：`a29-drawer-close-black-marble-30hz-v1` @`d03c19e6`，全部 100 条，30 Hz，模型输入只用 ego_view + chest（数据里的 kitchen、drawer 两路不用）。同一配方和代码，但硬件是阿里云 PAI DLC A800 集群的 1 × A800 80 GB。训练中 MSE（a29 train10）2000 步 0.1396，4000 步 0.1296。HF `act/a29-drawer-close/b512/`，act @`7d7a302b`。
  - A21（任务键 a21）：`a21-floor-transfer-basket-trim-raw-head-expert-v1` @`682f2bfc`，全部 100 条，30 Hz，ego_view + chest 两路。同一配方和代码，硬件是阿里云 PAI DLC A800 集群的 1 × A800 80 GB。训练中 MSE（a21 train10）2000 步 0.0353，4000 步 0.0288。HF `act/a21-floor-basket-transfer/b512/`，act @`415cfc57`。
  - B5 站立起步（任务键 b5ss，现名 A22）：`A22-chair-push-standing-start-expert-v1-formerly-b5` @`913f5a1`，全部 100 条（30,908 帧），30 Hz，ego_view + chest 两路。同一配方，代码 Humanoid_benchmark bai `d2096858`（对齐规范后的 `train.py`）；按 2026-11-27 存档规则存 750/1250/2000/2750/3250/4000 六个，HF 只传最后两个（3250、4000）。硬件是阿里云 PAI DLC A800 集群的 1 × A800 80 GB。训练中 MSE（b5ss train10）2000 步 0.0534，3250 步 0.0452，4000 步 0.0444。HF `act/b5-chair-push-standing-start/b512/`，act @`42e17a53`。
  - 新剪裁数据（用户 2026-11-27：A3 v4、A7 v4、B1 v2 和 H2 的 A14/A17/A18 按 A29 静止帧方式重剪，所有模型重训）：a3v4p、a7v4p（代码 `d2096858`），b1v2p、a14cr5、a18cr4（代码 `36ded31d`）；同一配方，A800 单卡，存 6 个均匀存档点、HF 只传 3250 和 4000，子目录统一 `b512-trim3f`（用户 2026-11-27 定名），act @`3d9388a7`。训练中 MSE 第 4000 步：0.0629、0.0751、0.0774、0.0879、0.0707。a17cr4 在重训（第一次中断，旧目录改名 `.aborted-20261127T0957Z`，不上传）。
  - 带种子重训（用户 2026-11-25）：b6v6p、b7td、b7gtd、a10v3、a3v4、a7v4、basketv2 在原数据上按同一配方（代码 `2c91a514`，`HB_AUG_SEED=0`）重训，HF 放在原目录名加 `-seeded`，act @`7fb26f44`；**`-seeded` 目录取代原目录**（原目录的增强不能逐位复现，保留只作参考）。
- 本仓库的 `hb_training/hb_act/`（Psi0 仓库内的 LeRobot ACT）**已退役**：入口 `hb_act_train.py` 默认直接退出，只有复现旧的 `act/<task>/epoch-NNN` 发布时才可设 `HB_ALLOW_RETIRED_ACT=1` 运行。说明见 `hb_training/hb_act/RETIRED.md`。

### 4.6 FLASH（2026-11-24 用户接入，用户自己的方法，NeurIPS 2026；先做 B6）

以下由 FLASH 会话 11-24 实测后登记。

- **代码**：上游 NTUMARS/FLASH-Policy @ `f47e490`（Apache-2.0）。移植到合作者的 hb_baselines：GitHub `zuo-kuangji/Humanoid_benchmark` bai `9cca6255`，入口 `evaluation/baselines/train.py --policy flash --contract g1_sonic_dfx6`，模型 `evaluation/baselines/flash.py`。与上游对拍（`check_flash_parity.py`）：多项式矩阵、KKT 边界约束、Tikhonov 历史拟合、DiT、编码器、输出点，12 项最大差均为 0。
- **环境**：hb-baselines，Python 3.12.1、torch 2.9.0+cu128、torchvision 0.24.0。
- **种子**：训练 42（上游 `default_train.yaml` 的 `training_params.seed`）；增强随机流由 `HB_AUG_SEED=42` 决定（第 2 节）；训练中 MSE 1234。
- **布局**：1 节点 × 1 卡 × 每卡 512 = 全局 512，不累积；`num_workers` 40，prefetch 2。B6 那次正式训练的硬件是阿里云 PAI DSW 的 1 × RTX 4090 46 GB（5 号卡、64–79 号核），不是 A800；火山 team_10（RTX 4090D）上的同配置训练停在第 4000 步，只作备份。以后 FLASH 所有任务沿用此布局。
- **步数与存档**（用户 11-25 改）：12000 步，每 4000 步存一次（4000、8000、12000），只上传 8000 和 12000 两个。2026-11-27 起新开的训练按第 1 节存 6 个（2000、4000……12000），上传仍只传 8000 和 12000；评测只评 s12000（用户 11-26）。11-24 原定 16000 步、每 2000 步存一次；11-25 以前按 16000 步训完的 B6（`b512-v6-500`）、B7（`b512-leaddrop`）结果保留。
- **优化**（按上游）：AdamW，lr 1e-4，betas (0.95, 0.999)，wd 1e-6；500 步线性预热后 diffusers cosine，到最后一步（12000）降到 0；EMA 用上游 Diffusion Policy 的 EMAModel（power 0.75，max 0.9999），保存 EMA 权重；bf16 autocast，多项式拟合用 fp32。
- **模型**（按上游 `flash.yaml`）：DiT 512 × 4 层 × 8 头，dropout 0，time embed 256；每路相机一个 ResNet18，从头训，GroupNorm，与其余部分同一学习率；Legendre 6 阶，fit_pad 1、poly_prefix 1、poly_suffix 1，C1 边界约束；history_noise_std 0.5、history_reg_lambda 0.1、consistency_weight 1.0；推理 1 步 Euler。
- **接口适配**（用户 11-24 定，属模型内部，不算例外）：
  - 历史 = 接口的 past_actions（每帧发出的 token 和当帧实测手部开合），8 个点、间隔 2 帧（30 Hz 下 67 ms，论文是 60 ms），截止到 t-1；episode 开始前重复第一行；第 0 帧没有历史行，flow 从零系数起步。
  - 只用当前帧图像（ego_view + chest，168 × 224）和 state44，不用原版的 8 帧观测历史。
  - 输出：多项式在 30 Hz 每帧求值，共 16 行；不用速度前馈。
  - 增强 v2（原生随机裁剪本来不生效）；本体 29 关节常偏加噪声；44 维、无速度；预检已通过。
- **数据**：B6 这次用 b6v6，HF `b6-pedal-bin` @`ad2f5bf0` 前 500 条（ep 0–499，305,667 帧），按当时 11-18 的规定，与 FastWAM、Delta Flow 相同。11-24 起 B6 v6 改用全部 542 条，这次训练登记在第 7 节。
- **训练中 MSE**：`evaluation/baselines/inloop.py` 调用 eval-action-mse 的 `hb_inloop_core`（HumanWAM bai `ff5bf95` 的拷贝），协议不变：每 250 步、b6v6 train10、每批 8 个查询、种子 1234、评 EMA 权重；结果写 `<run>/logs/inloop-mse.jsonl`，日志有 `HB_INLOOP installed/ready`。team_10 上的缓存用同一个 `build_cache.py`（torchcodec 0.8.0）重建，基线 train_mean 0.9049、hold_last_action 0.3427 与 NAS 一致，9 个缓存 episode 的 state/action 逐位核对过。
- **运行与上传**：运行目录 NAS `preflight-20261104T1648Z/flash-runs/b6v6-b512-s16000`（team_10 的部分训练备份在同级 `team10-partial-20261124/`）；HF 上传位置待用户定。上传的 `protocol.json` 要带 `seed` 和 `reproducibility`（`hb_repro.fields` 能按 `hb_baselines/1` 格式和 policy `flash` 生成），有例外再加 `exceptions`。

### 4.7 ODE-MLP / ODE-DiT（2026-11-25 用户接入；先做 B6）

- **代码**：hb_baselines，GitHub `zuo-kuangji/Humanoid_benchmark` bai `b4b94d29`，入口 `evaluation/baselines/train.py --policy ode_mlp|ode_dit --contract g1_sonic_dfx6`，模型 `evaluation/baselines/policies.py`（`ODEPolicy`、`ODEDiTPolicy`）和 `ode_nets.py`。
- **来源**：NTUMARS/ODE-POLICY（Apache-2.0）。结构以作者的 G1 移植为准（`real-g1` @ `fd8bec47`，本仓库 `packages/ode_g1` 的 `SonicLatentODE`），因为它的动作就是本项目的 G1 action76；G1 移植的 3 B GR00T 骨干换成上游 `ode_policy`（main @ `cea26d94`，`ode.yaml`）自己的图像编码器；ODE-DiT 的速度场取自 `ode_dit.yaml`。速度场与上游逐行对照过源码。
- **种子**：训练 42（上游 `default_train.yaml`）；增强随机流 `HB_AUG_SEED=42`；训练中 MSE 1234。
- **模型**：
  - ODE 状态 = contract 的第一组动作（G1：SONIC token 64；H2：`body_target` 31）。用 Euler 积分，每行一步、步长 1：b_{k+1} = b_k + f(b_k, s | c)，共 16 行。
  - 速度场：ODE-MLP 是 FiLM 残差 MLP（宽 512、4 层、输入 [body, state44]、输出零初始化、field_scale 1）；ODE-DiT 是 adaLN DiT（512 × 4 块 × 8 头、4 个 token）。
  - 条件 c = Linear(LayerNorm(图像特征)) → 512，不含本体。双手由 c 经 Linear-SiLU-Linear 解码，连续值，不加 sigmoid。
  - 每路相机一个 ResNet18，从头训，GroupNorm，与其余部分同一学习率。
  - 归一化：q01/q99 到 [-1, 1]，DFX6 固定按 [0, 1] 映射，截断 ±5。
  - 损失：带 mask 的 MSE（超出 episode 末尾的行不计），body 和手按维数 64:12 加权。
  - 起点：训练用录制数据里 t-1 行的 body，episode 第一帧用归一化零向量 BOS；部署用 `past_actions[-1]`（实际执行的上一行），没有历史时用 BOS。
  - 不加起点扰动、不加多样性头（G1 移植都没有）。
- **优化**（上游 `default_train.yaml`）：AdamW，lr 1e-4，betas (0.95, 0.999)，wd 1e-6，**学习率不按 batch 缩放**（与本规范所有模型一致：全局 512，用官方学习率原值；上游的 1e-4 对应 batch 32，FLASH 同样如此）；500 步线性预热后 diffusers cosine 降到 0；EMA 用 Diffusion Policy 的 EMAModel（power 0.75，max 0.9999），按上游方式计数，保存 EMA 权重。编码器 bf16 autocast，ODE 头（条件、积分、手）FP32。
- **布局**：1 节点 × 1 卡 × 每卡 512 = 全局 512，不累积；`num_workers` 32，prefetch 2。B6 这次的硬件是工作站 b1ue-jay，1 × RTX 5090 32 GB，AMD Ryzen 9 9950X（32 线程）。以后 ODE 所有任务沿用此布局。其余 18 个任务（2026-11-27 起）在阿里云 PAI DLC 的 A800 集群上训：每个任务 1 张 A800 80 GB，同一节点并行 4–5 个任务，各分同样多的 CPU 核（节点 92 核），代码同为 `b4b94d29`，运行目录 `ode-runs/<task>-b512-s<steps>/`；H2 任务的 contract 是 `h2_sharpa`（A17、A18）和 `h2_sharpa_planner`（A14）。
- **步数与存档**（2026-11-27 起新开的训练按第 1 节存 6 个：5000 步为 750、1750、2500、3250、4250、5000，命令不再写 `--save-every`）：5000 步，C6 为 10000 步（长任务翻倍，与其他模型一致；用户 11-26，依据 B6 普通 50 局：step-005000 28/50、step-006000 28/50、step-008000 23/50；取代 11-24 的 8000 步），每 1000 步存一次。cosine 在最后一步降到 0，所以必须按这个步数从头训，不能从更长的运行里截取中间的 checkpoint。
  - **评测和上传只用最后一个 checkpoint**：普通任务 `step-005000`，C6 `step-010000`。选 checkpoint 只在一个模型第一次评测时做（规则见 `evaluation/README.md`，bai `0f9a5309`），ODE 已在 B6 上选定第 5000 步。中间每 1000 步的存档照存。
  - **例外**：B6 这次是 8000 步的运行，评测和上传用其中的 `step-005000`，见第 7 节。
- **数据**：b6v6，HF `b6-pedal-bin` @`ad2f5bf0` 全部 542 条，331,577 帧。其余任务用 `protocol.yaml` `tasks` 下各任务的最新数据，全部条数。
- **训练中 MSE**：`--inloop-task b6v6 --inloop-code <本仓库 hb_training/eval-action-mse>`，每 250 步、`train10`、每批 8 个查询、种子 1234、评 EMA 权重。ODE 从缓存的 `past_action` 最后一行起步。缺这一项时训练会拒绝启动，除非加 `--allow-no-inloop`，那会记为例外。
  - 在新机器上建 `train10` 缓存：环境 hb-baselines 加 `torchcodec==0.8.0`。按 `hb_eval_common.py` 的布局，`$G/../b6-pedal-bin-v6-ad2f5bf/data` 指向本地数据，再运行 `G=<...> python build_cache.py b6v6 train10`。b1ue-jay 上建出的基线 train_mean 0.9049、hold_last_action 0.3427，与 NAS 一致。
- **命令**（任何机器上完全相同，双方可互相复现）：

```bash
G=<root of eval-action-mse> <hb-baselines env>/bin/python -I evaluation/baselines/train.py \
    --policy ode_mlp --contract g1_sonic_dfx6 --dataset <data>/b6-pedal-bin --out <run> \
    --steps 5000 --batch-size 512 --workers 32 --cache-dir <frame cache> \
    --inloop-task b6v6 --inloop-code <HumanWAM>/hb_training/eval-action-mse
```

  种子、学习率、调度、EMA、horizon、编码器设置都是 `--policy ode_mlp` 的默认值，命令里不再另写。C6 用 `--steps 10000`；`--inloop-task` 填该任务的任务键。
- **已作废的一次**：2026-11-24 的 `b6v6_b512_s8000` 是登记前训的（种子 0、lr 4e-4、增强未固定种子、没有训练中 MSE），不作为正式结果，见第 7 节。
- **上传**：公开仓库 `humanoid-benchmark/ode`，`<task-dir>/b512-<ver>/step-XXXXXX/`，`hb_baselines/1` 格式；B6 在 `b6-pedal-bin/b512-v6/`。每个任务只传最后一个 checkpoint（见「步数与存档」）。

### 4.8 FM-DiT v2（2026-11-25 用户接入，FLASH-Policy 自带的 FM-DiT）

- **代码**：hb_baselines `evaluation/baselines/fm_dit_v2.py`（Humanoid_benchmark bai `3eb27c92` 起），上游 NTUMARS/FLASH-Policy @ `f47e490` 的 `roboverse_learn/il/policies/fm/fm_dit_image_policy.py` + `fm_dit.yaml`；网络复用 flash.py 移植的 FlowTransformer（torchtune RoPE）和 MultiImageObsEncoder。入口 `train.py --policy fm_dit_v2 --contract g1_sonic_dfx6`。
- **配方**：种子 42（上游 `default_train.yaml`），全局 512（1 × 1 × 512），12000 步，AdamW lr 1e-4、betas (0.95, 0.999)、wd 1e-6，500 步线性预热后 diffusers cosine 到 0，EMAModel（power 0.75、max 0.9999）；DiT 512 × 4 × 8 头、dropout 0.1、time embed 256；每路相机一个从头训的 GroupNorm ResNet18；10 步 midpoint 采样；`num_workers` 32；存 6 个均匀存档点。
- **B6 v6**（全部 542 条）：`fmdit-runs/b6v6p-fm_dit_v2-b512-s12000`，训练中 nMSE 12000 步 0.1599（v1 同配置 0.2545），0.5.1 普通 50 局 0/50。
- **B6 续训 30000 步**：用户 11-27 先定续训 30000 步，同日取消（B6、B7、C6 的训练全部暂停，等按 `debug/newton/a29_trim_stationary.py`（Humanoid_benchmark `be2caba1`）的 3 帧剪裁方式重新剪裁数据）。

---

## 5. 数据版本（每个任务当前最新版，开训前逐项核对）

本地视图目录名里带 HF commit 前缀；HF 仓库都在 `humanoid-benchmark` 组织下，均为私有。

| 任务键 | 任务 | HF 数据 @ commit | 条数 | 机器人 |
|---|---|---|---|---|
| a3v4 | A3 开冰箱门 v4 | @`6393d4a` | 全部 | G1 |
| a3v4p | A3 开冰箱门 v4，按 A29 静止帧方式剪裁（Humanoid_benchmark `be2caba1` 的 `debug/newton/a29_trim_stationary.py`，用户 2026-11-27）（`a3-fridge-open-expert-v4`） | @`fe89f5b3` | 全部 125 条 | G1 |
| a7v4 | A7 关橱柜门 v4 | @`90cc2b5` | 全部 | G1 |
| a7v4p | A7 关橱柜门 v4，按 A29 静止帧方式剪裁（Humanoid_benchmark `be2caba1` 的 `debug/newton/a29_trim_stationary.py`，用户 2026-11-27）（`a7-cabinet-close-expert-v4`） | @`d3ec8057` | 全部 100 条 | G1 |
| a10v3 | A10 推小车 v3 | @`6e410c5` | 全部 | G1 |
| basketv2 | A20 篮子转移，伸手前剪裁 v2（`a20-basket-transfer-pre-reach-trim-expert-v2`） | @`c4df4d4` | 100 | G1 |
| b1 | B1 饭团进微波炉（`b1-riceball-microwave-expert-v1`） | @`36da298` | 115 | G1 |
| b1v2 | B1 饭团进微波炉 v2，30 Hz 版（`b1-riceball-microwave-expert-v2-30hz`） | @`b58f0434` | 全部 200 条 | G1 |
| b1v2p | B1 饭团进微波炉 v2 30 Hz，按 A29 静止帧方式剪裁（Humanoid_benchmark `be2caba1` 的 `debug/newton/a29_trim_stationary.py`，用户 2026-11-27） | @`276ceb91` | 全部 200 条 | G1 |
| b4 | B4 靠垫从床到沙发（`b4-cushion` 根目录 v2） | @`194e7cde` | 全部 302 条 | G1 |
| a29 | A29 关抽屉，30 Hz 版（`a29-drawer-close-black-marble-30hz-v1`），只用 ego_view + chest | @`d03c19e6` | 全部 100 条 | G1 |
| a21 | A21 从地上拿篮子放到长桌（`a21-floor-transfer-basket-trim-raw-head-expert-v1`），ego_view + chest | @`682f2bfc` | 全部 100 条 | G1 |
| b5 | B5 推椅子（`b5-chair-push-expert-v1`） | @`b85b7e9` | 100 | G1 |
| b5ss | B5 standing start，现名 A22（`A22-chair-push-standing-start-expert-v1-formerly-b5`） | @`913f5a1` | 100 | G1 |
| b6v6 | B6 丢垃圾 v6（`b6-pedal-bin` 根目录） | @`ad2f5bf0` | 全部 542 条（11-24 用户起）；11-24 以前 FastWAM、Delta Flow、FLASH 训完的版本用前 500 条（0–499） | G1 |
| b7t | B7 黑衣服，剪裁版（`b7-black-clothes-hamper`） | @`949da46` | 前 100 / 107 | G1 |
| b7gt | B7 绿衣服，剪裁 v2（`b7-green-clothes-hamper` 根目录） | @`22e3b83` | 421 | G1 |
| b8 | B8 擦桌子（`b8-table-wipe-expert-v1`） | @`3b2e618` | 全部 | G1 |
| c6 | C6 浴室开门取物（`c6-bathroom-door-fetch-expert-v1`） | @`3afba69` | 全部 | G1 |
| a14cr3 | A14 拧灯泡，cropped-v3（`a14-screw-bulb` 根目录） | @`79fe08a` | 全部 | H2，动作 82 |
| a14cr5 | A14 拧灯泡，cropped-v5，按 A29 静止帧方式剪裁（Humanoid_benchmark `be2caba1` 的 `debug/newton/a29_trim_stationary.py`，用户 2026-11-27）（`a14-screw-bulb` 根目录） | @`33beae7c` | 100 | H2，动作 82 |
| a17cr2 | A17 拧瓶盖，cropped-v2（`a17-unscrew-bottle` 根目录） | @`3520b70f` | 100 | H2 |
| a17cr4 | A17 拧瓶盖，cropped-v4，按 A29 静止帧方式剪裁（Humanoid_benchmark `be2caba1` 的 `debug/newton/a29_trim_stationary.py`，用户 2026-11-27）（`a17-unscrew-bottle` 根目录） | @`f7105314` | 100 | H2 |
| a18cr2 | A18 倒水，cropped-v2（`a18-pour-water` 根目录） | @`44cdd571` | 100 | H2 |
| a18cr4 | A18 倒水，cropped-v4，按 A29 静止帧方式剪裁（Humanoid_benchmark `be2caba1` 的 `debug/newton/a29_trim_stationary.py`，用户 2026-11-27）（`a18-pour-water` 根目录） | @`73fd5225` | 100 | H2 |

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
| GR00T C6（`c6-bathroom-door-fetch/b512`） | 11-13 旧配置：**4 节点** × 8 卡 × 每卡 8 × **梯度累积 2**（全局 512），8000 步；布局和累积都不符合标准 2 节点 × 8 × 32、不累积 | 11-14 用户：先不重训；11-24 用户：布局也登记为例外，不重训 |
| GR00T basket v1、B7 黑 / 绿 v1（`groot-runs-ft512`） | 同上，4 × 8 × 累积 2 | 已有按新配置训的新数据版本，旧目录保留 |
| GR00T / Psi-Zero basket v2 | HF 上已有的版本是从 v1 checkpoint 续训 2000 步（`b512-trimv2-from-b512`、`b512-trimv2`），不是在 v2 上从 base 训的 | 11-23 补训纯 v2，任务键 `basketv2p`，上传 `b512-trimv2-pure/` |
| Psi-Zero C6 | 4000 步，没有按长任务翻倍 | 11-13 用户：不重训 |
| Psi-Zero A10 v3（`a10-trolley-push-v3`）、B1（`b1-riceball-microwave`）、B8（`b8-table-wipe`）、C6（`c6-bathroom-door-fetch`） | 节点布局 **4 节点** × 8 卡 × 每卡 16（全局 512、不累积、4000 步），不符合标准 2 节点 × 8 × 32（A14 的 4 × 8 × 16 是另行规定的布局，不在此列） | 11-24 用户：登记为例外，不重训 |
| Psi-Zero A14 cr3（`cropped-v3-b512`）、旧 A17/A18 | 带本体延迟 0.15（原 H2 指南例外） | 11-22 取消；重训为 `a14cr3a`（`cropped-v3-b512-nodelay`）和 `b512-2cam-cropped-v2` |
| π0.5、Psi-Zero、ACT 在 11-12 前训的 | 15% 本体延迟 + 15% 历史丢帧 | 结果保留，附录说明；受影响任务按对齐配置重训 |
| B6 v6 条数 | 11-24 起规定值是全部 542 条。已训完的 FastWAM、Delta Flow、FLASH（`b6v6-b512-s16000`）用的是前 500 条（当时 11-18 的规定），结果保留，作为例外登记。GR00T / Psi-Zero 纯 v6（`b512-v6-pure`）、ACT 纯 v6（hb_baselines，`act/b6-pedal-bin/b512-v6-pure`）和早期的 v6 续训版都是 542 条，现在符合规范 | 11-18 用户定前 500 条；11-24 用户改为以后都取全部 542 条 |
| ACT 纯 v6（hb_baselines，`b512-v6-pure`） | hb_baselines 的 ACT/FM-DiT/DP 数据集每个样本的增强抽取用 pid、urandom 产生（只有 FLASH 用 `HB_AUG_SEED`），不能逐位复现；和已发布的 `b512*` ACT 一样。分布完全是增强 v2。硬件是阿里云 PAI DSW 1 × RTX 4090 46 GB，`num_workers` 32，进程以 nice 19 运行（同机在跑闭环评估） | 11-24 登记；11-25 用户：hb_baselines 的增强从 Humanoid_benchmark bai `b4b94d29` 起对所有模型按种子固定（第 3 节 1b），之前的运行保留，标为 not seeded；11-27 起已由带种子重训的 `b512-v6-pure-seeded` 等 `-seeded` 目录取代（第 4.5 节） |
| ODE-MLP B6 v6（`b6v6_b512_s8000`，11-24） | 登记前训的：种子 0、lr 4e-4（按 √batch 放大）、增强未固定种子、没有训练中 MSE | 11-25 用户：不作为正式结果，按 4.7 重训 |
| ODE-MLP B6 v6 正式版（`humanoid-benchmark/ode` `b6-pedal-bin/b512-v6`） | 按 11-24 的旧步数训了 8000 步。11-26 规定步数改为 5000 后，正式 checkpoint 取这次运行的 `step-005000`；它在第 5000 步时学习率还没降到 0，和从头训 5000 步的运行不是同一个训练 | 11-26 用户：不重训，登记为例外 |
| Delta Flow A17 cr2（`a17cr2lg`） | 沿用 legacy a18 e40 配方（旧增强 `b6-visual-aug-v2`，无本体增强） | 11-22 用户：Delta Flow 先不管 |
| Delta Flow B6 v6 | 从 v5 `step_010200` 只加载权重续训，不是从 base 训 | 11-15 计划 |
| FastWAM B5 standing start | 不训 | 用户 |
| 多个早期运行 | **没有训练中 MSE**：2026-11-15 训练中评估机制上线前训的（如 GR00T C6、Psi-Zero C6），以及任务键当时没登记、评估自动关闭的（GR00T A14 cr3、FastWAM B7 绿衣服 4000 步及续训、FastWAM A14 cr3 4000 步，Psi-Zero A14 cr3 只有 1 个点） | 只有事后补测的开环 MSE；11-22 起不再允许，新训练必须伴随训练测 |
| 种子 | Psi-Zero 和 Psi0 版 ACT 是 292285；合作者 hb_baselines 的 ACT（b512 目录）、FM-DiT、DP 是 0；其余模型是 42（各用上游或所用代码的默认值） | 2026-11-27 用户：每个模型内部统一一个种子，不跨模型统一；FM-DiT v1/v2 以后都用 42，新模型用 42；已用别的种子训完的不重训，记录种子即可 |
| 增强随机流 | 2026-11-23 21:05 之前启动的 GR00T、Psi-Zero、FastWAM 训练：由 pid、urandom、时间产生，不可逐位复现（第 3 节） | 11-23 已修，之后启动的训练由种子决定；旧结果保留，附录注明 |

| FM-DiT v1 / v2 对比（hb_baselines `--policy fm_dit` 与 `fm_dit_v2`）：B6 v6 全部 542 条（`fmdit-runs/b6v6p-*-b512-s12000`）、A20 basket v2 全部 100 条（`fmdit-runs/basketv2-*-b512-s12000`） | 两版都用种子 42、batch 512、12000 步、每 4000 步存。v1 规定的 hb_baselines 种子是 0；v2 是 FLASH-Policy 自带的 FM-DiT，42 是它的上游默认值，不算例外。增强 v2，随机流已固定（`HB_AUG_SEED=42`），训练中 MSE 每 250 步 | 11-26 用户：v1 用 42，和 B6 对比一致；登记为例外 |
| hb_baselines 训练环境 | 11-24 到 11-26 在阿里云 DSW 上训的 hb_baselines 运行（FM-DiT v1/v2 B6 与 A20、ACT B6 v6 / B1 v2 / B4 / B7、FLASH B6 / B7 等）用 `/root/.venvs/hb-baselines`（Python 3.12.1、Pillow 12.3.0）；A800 上的运行用 NAS 的 `policies-20261109/envs/hb-baselines`（Python 3.12.12、Pillow 13.0.0）。增强 v2 的 JPEG 效果依赖 Pillow，两者逐位不同、分布相同 | 11-27 起钉为 Pillow 13.0.0（`envs/hb-baselines.txt`，hb_baselines `requirements.txt`），每个运行记录环境版本 |
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
- **`seed` 和 `reproducibility` 块**：由 `hb_repro.py` 自动写入（第 3 节第 4 条）。新模型接入自动上传前，先在 `hb_repro.MODELS` 里登记它的种子、环境和上游代码。

---

## 9. 开训前检查清单（每次都做）

1. **数据**：HF repo@commit、条数规则、fps、相机、state 和 action 维度都与第 5 节一致。新数据先逐项核对。
2. **实际取一个 batch**：确认原始本体是 44 / 78 维、不含速度；编码后的维度符合模型原生配方；增强日志里出现 `HB_AUG` / `HB_FW_AUG`，并带 `boost=`；GR00T、Psi-Zero、FastWAM 的日志里还要有 `HB_AUG_SEED=<训练种子>`。
3. **配方**：种子、全局 batch、梯度累积 = 1、步数、学习率曲线、存档点都与第 2、4 节和 `protocol.yaml` 一致。Psi-Zero 要确认 `HB_DELAY_PROB` 未设置。
4. **训练中 MSE**：`hb_eval_common.TASKS` 已登记、`train10` 缓存已建好；开训后日志有 `HB_INLOOP` 且是启用状态。
5. **上传**：新子目录已在自动上传里登记，不与已有目录重名。
6. **训完**：运行 `check_protocol.py`（下面的命令），全部 PASS 或只剩已登记的例外，才算交付。
7. **偏离**：写进第 7 节、运行目录和 `protocol.json`。

```bash
python3 hb_training/check_protocol.py groot  $G/groot-runs-2x32/<task>
python3 hb_training/check_protocol.py psi0   $G/psi0-runs/<task>/b512
python3 hb_training/check_protocol.py fastwam $G/fastwam/runs/<task>[-cont2k]
python3 hb_training/check_protocol.py hb_baselines <run>   # ACT / FLASH / FM-DiT / DP / ODE: run.json + complete.json
```

---

## 10. 环境

- 各训练环境的完整包版本快照在 `hb_training/envs/<env>.txt`（2026-11-23 导出；hb-baselines 2026-11-27 导出，同 Humanoid_benchmark `evaluation/baselines/requirements.lock.txt`）。重建环境时按快照装，不要装最新版。
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
