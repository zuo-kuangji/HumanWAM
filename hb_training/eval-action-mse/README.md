# 开环动作重建 MSE 评估（eval-action-mse）

2026-11-14 写。用来统一比较各模型"有没有学会输出动作"：给模型看数据集里某一帧的观测，让它推理一个 16 步动作块，再和录制的真值动作比。

## 协议（所有模型完全一致，参照 Isaac-GR00T 官方 `gr00t/eval/open_loop_eval.py`）
- **查询点**：每条 episode 从第 0 帧起每隔 16 帧查询一次，覆盖整条 episode，相邻动作块不重叠。
- **模型输入**：与部署时相同，即相机图像、基准原始本体（G1 44 维 / H2 78 维）和指令，所有增强关闭。Delta Flow 额外使用它原生的过去 16 步动作历史，取自录制的真值；这是它模型结构本身的输入。
- **比较对象**：预测动作块的前 16 步（原始单位）对比录制的 t..t+15。超出 episode 末尾的行不计入，不用截断或填充出来的目标。
- **指标**
  - `nmse`：每个动作维度先除以该任务训练帧上的标准差（下限为该向量标准差中位数的 5%），再求 MSE。nMSE = 1 大约相当于一直输出训练均值。
  - 分组 nMSE：G1 分 motion_token 64 维和 hands 12 维；H2 分 body_target 31 维和 hands 44 维。
  - `nmse_by_step`：按预测步 0–15 分别统计。
  - `nmse_per_episode` 和 `nmse_episode_sem`：逐 episode 结果及其标准误。
  - `mse_raw`、`mae_raw`：原始单位下的误差。
- **参照基线**：缓存建好时顺带算出，写在 meta.json 里。
  - `train_mean`：一直输出训练均值。
  - `hold_last_action`：把上一步的真值动作重复 16 次。这条基线用到了真实动作历史，只作参照。
- **评估集**
  - `heldout`：从未参与训练的录制 episode，只有 B6（第 100–148 条）、trolley（第 100–102 条）、B7 黑衣服（第 100–106 条）有。
  - `train10`：每个任务 10 条固定的训练 episode，衡量的是拟合程度，不是泛化能力。

## 文件
| 文件 | 作用 |
|---|---|
| `hb_eval_common.py` | 任务表、缓存格式、指标、主循环（只依赖 numpy） |
| `build_cache.py` | 生成与模型无关的评估缓存 `cache/<task>/<split>/`（CPU，在 GR00T 环境里运行，用 torchcodec 解码） |
| `eval_run.py` | 评估一个检查点在一个任务、一个 split 上的结果 |
| `run_eval.sh` | 进入该模型自己的训练环境，再调用 eval_run.py |
| `adapters/ad_*.py` | 各模型的推理适配器，都走模型自己的部署或推理路径 |
| `make_jobs.py` | 扫描训练目录生成 `jobs.txt`（B6 留出集的作业排最前） |
| `eval_queue.py` | 空档排队（v2，2026-11-15）：读本节点 relay 状态，只在 barrier 等待、relay 队列跑完或 idle-slot 项目（项目脚本里含 eval_queue.py）期间，且 GPU 和进程检查都空闲 2 分钟后才评估；relay 即将开下一项、或出现别的 GPU 进程或训练进程时，15 秒内停掉评估并把作业放回队列 |
| `start_queue.sh` | 在本节点停掉旧队列并启动新队列（每节点只允许一个实例），记录写到 `logs/start-<host>.txt` |
| `results/<id>.json`、`<id>.json.pred.npz` | 评估结果和逐查询点的预测 |
| `smoke/` | CPU 冒烟测试（B6 留出集，1 条 episode，2 个查询点） |

## 用法
```bash
cd $G/eval-action-mse
# 建缓存（已建：所有任务的 train10，以及 b6/trolley/b7 的 heldout）
LD_LIBRARY_PATH=$G/envs/groot-py312/lib/hbstub $G/envs/groot-py312/bin/python build_cache.py <task> <heldout|train10>
# 单次评估
bash run_eval.sh groot b6 heldout $G/groot-runs-ft4k/b6/production/checkpoint-4000 results/x.json --device cuda:0
# 重新生成作业清单（已有结果会保留）
python3 make_jobs.py
# 在某个节点上开启空档排队（只用本节点的卡；worker 需在 PAI 网页终端里执行）
bash start_queue.sh
python3 eval_queue.py --once        # 只查看节点状态和待跑作业，不启动任何评估
touch PAUSE   # 不再启动新的评估；rm PAUSE 恢复
touch STOP    # 停掉本节点所有评估，并退出排队
```
运行中的评估会写 `results/RUNNING-<host>.json`。训练调度可以照常在这些卡上起训练：relay 的 wait_gpu_free 最多多等约 1 分钟。被让出而杀掉的评估不计失败。
注意：DLC 容器里 `nvidia-smi` 报的是宿主机 PID，和容器内 PID 对不上，所以 v2 按 GPU 编号判断占用（v1 因此每 30 秒就把自己的评估当成训练杀掉）。

## 已知限制
- GR00T 的 Qwen3-VL 骨干只支持 FlashAttention2。CPU 冒烟时临时换成 sdpa；GPU 正式评估用原生的 FlashAttention2。
- Psi-Zero 预测 30 步动作块，只评前 16 步；去噪步数用 10（serve_psi0_simple 的默认值，run_config 里没有写）。
- 2026-11-13 之前训练的 π0.5 H2 检查点是双相机（head+chest）。make_jobs 按文件修改时间判断，自动给这些作业加上 `cams=head,chest`。
- Delta Flow 只支持主线 run；velocity、delay、2cam 等变体的输入协议不同，不评。H2 的 Delta Flow 还没有冒烟。
- DiT4DiT 的适配器已写好，但写的时候还没有任何检查点，未冒烟。master-0 上的 basket 训练完成后先跑一次冒烟。


## 训练中评估（2026-11-15 起，GR00T / Psi-Zero / Delta Flow）
训练过程中每 250 步（`HB_INLOOP_EVERY`，设为 0 即关闭）用显存里正在训练的模型测一次。协议、评估集、指标和上面的事后评估完全一致，推理走各模型自己的部署路径；所有卡分摊查询点，最后汇总一次。

| 模型 | 评估模块 | 接入点 | 结果文件 |
|---|---|---|---|
| GR00T | `hb_groot/hb_inloop_eval.py`（Trainer 回调） | `hb_groot/hb_launch.py` | `<run>/logs/inloop-mse.jsonl` |
| Psi-Zero | `hb_psi0/hb_inloop_psi0.py`（包装 FinetuneTrainer.step） | `hb_psi0/hb_train.py` | `<project_dir>/inloop-mse.jsonl` |
| Delta Flow | `hb_deltaflow/hb_inloop_deltaflow.py`（包装 _checkpoint_save_flags） | 各代码树的 `scripts/wholebody/train_deltaflow.py` | `<run>/run/production/inloop-mse.jsonl` |

- Psi-Zero 和 Delta Flow 共用 `hb_inloop_core.py`。
- 评估结束后，各模块的 train/eval 状态和所有随机数状态都恢复原样，不产生梯度；去噪噪声使用固定种子。
- 任何一张卡出错，所有卡都会关掉之后的评估，训练照常继续。
- 新任务必须先建缓存（`build_cache.py <task> train10`）并在 `TASKS` 里登记，否则训练中评估会自动关闭。
- CPU 验证（`test_inloop_cpu.py`、`test_inloop_psi0_df_cpu.py`）：三个模型的输出都和部署或适配器推理逐位相同。
- GPU 实测：GR00T a10v3 在 16 张卡上评 533 个查询点用了 19 秒，约占训练时间的 5%。
