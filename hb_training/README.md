# hb_training：Humanoid Benchmark 多模型训练适配

本目录存放 Humanoid Benchmark 任务在阿里云 PAI DLC（4 节点 × 8 张 A800）上训练用的适配代码。内容同步自训练节点上的 `$G = …/preflight-20261104T1648Z/policies-20261109`，同步时间 2026-11-10。

## 模型与上游代码

| 目录 | 模型 | 上游代码（HumanWAM 子模块） |
|---|---|---|
| `hb_pi05` | π0.5 | `third_party/openpi` @ `215abfb` |
| `hb_psi0` | Psi-Zero | `third_party/Psi0` @ `cda3fdf` |
| `hb_dit4dit` | DiT4DiT | `third_party/DiT4DiT` @ `66a6f3a` |
| `hb_groot` | GR00T N1.7 | `third_party/Isaac-GR00T` @ `7aa0bca` |
| `hb_deltaflow` | ImageWAM Delta Flow | 分支 `wholebody-deltaflow-training-20261027` @ `9cb38e1`，加上 `hb_deltaflow/code/` 下的改动 |

- 这些适配是在 HumanWAM `c76527b` 上开发并验证的。
- 本分支基于 `main` 的 `c10b67c`，上表四个子模块在这两个提交里固定的版本完全相同。

## 任务与数据

| 任务 | 机器人 | 说明 |
|---|---|---|
| B6 丢垃圾 | G1 | v5t7 数据共 149 条，只取固定的前 100 条（episode 0–99） |
| A3 开冰箱门 | G1 | |
| A7 关橱柜门 | G1 | |
| A18 倒水 | H2 | |

每个任务都是 100 条。

## 训练设置

- **π0.5、Psi-Zero、DiT4DiT、GR00T**：
  - 训 20 个 epoch，第 10、20 个 epoch 各存一次 checkpoint；
  - 自动探测显存，batch 取能放下的最大值；
  - 续训状态（优化器、调度器、RNG）只保留最新一个 checkpoint 的。
- **Delta Flow**：训 40 个 epoch。
- **数据增强**：统一用 Humanoid Benchmark 数据增强规范 v2，各模型原生的图像增强全部关闭。
  - 实现：`hb_groot/hb_aug_v2.py`（`hb_pi05` 里有一份相同的拷贝）；
  - 规范说明：`hb_groot/TRAINING_DEFAULTS_v2.md`。

## 运行方式

1. **部署**：脚本是按 `$G` 的目录结构写的。部署时把 `hb_*`、`hb_relay.sh` 和 `setup/*` 都复制到 `$G` 顶层。
2. **环境**：
   - 每个模型的 Python 环境由各自的 `build_env.sh` 建在 `$G/envs/<name>`；
   - 基础镜像为 PAI `pytorch:2.6.0-gpu-py312-cu124-ubuntu22.04-accl`；
   - GR00T 用的是镜像自带的 `/usr/local/bin/python3.12`；它需要的 ffmpeg 系统库，`groot_node.sh` 发现缺少时会自动用 apt 安装。
3. **每个模型的流程**：
   1. 数据准备：`prep_all.sh` 或 `prepare*.py`；
   2. CPU 验证：`verify_cpu.py`；
   3. 正式训练：`*_chain.sh` / `*_node.sh`。
4. **队列**：
   - 4 个节点各跑一个 `hb_relay.sh`，按顺序执行 `$G/relay/queue.txt` 里的条目；
   - 条目有三种：`pi05 <task>`、`psi0 <task>`、`cmd <script>`；
   - 当时的队列见 `relay_queue.txt`；
   - 不要原地编辑正在运行的 `hb_relay.sh`（bash 是边读边执行的）。新增模型请写 `cmd` 包装脚本，例如 `groot_<task>.sh`、`dit4dit_<task>.sh`、`deltaflow_<task>.sh`。

## setup/

| 文件 | 用途 |
|---|---|
| `download_bases.py` | 下载基座权重 |
| `fetch_public.py` | 下载公开模型 |
| `download_gated.py` | 下载 gated 模型，令牌从标准输入读 |
| `dl_lerobot_pi05.py` | 下载 π0.5 基座 |
| `pdl.py` | 分段并行下载 GitHub release |
| `fa_fetch.sh` | 下载 flash-attn wheel |
| `groot_env.sh`、`groot_prep_run.sh` | GR00T 的环境与数据准备 |
| `deltaflow_prep_and_queue.sh` | Delta Flow 数据准备并加入队列 |

## 令牌

- 需要令牌的是两个 gated 模型：`nvidia/Cosmos-Reason2-2B`（GR00T 用）和 `nvidia/Cosmos-Predict2.5-2B`（DiT4DiT 用）。
- 下载时从 master-0 容器本地盘的 `/root/.hb-secrets/hf_token`（仅 root 可读）或标准输入读取令牌。
- 本仓库和 NAS 上都不存放令牌。

## Delta Flow 代码改动（`hb_deltaflow/code/`）

| 目录 | 对应任务 | 训练时用的完整代码副本 |
|---|---|---|
| `code/g1/` | B6、A3、A7 | `$G/deltaflow-runs/HumanWAM` |
| `code/h2/` | A18 | `$G/deltaflow-runs/HumanWAM-h2` |

- 两个目录里只放相对 `9cb38e1` 新增或修改的文件，保留原来的相对路径。
- 各目录下的 `deltaflow-code.json` 记录了改动说明和文件的 sha256。
- 完整代码副本由 `prepare_deltaflow.py`、`prepare_deltaflow_a18.py` 从此前的 B6、A18 训练目录复制并打补丁生成。
