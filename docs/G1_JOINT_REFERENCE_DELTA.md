# G1 joint41 / joint47 ImageWAM Delta-flow

这是一条独立的关节参考训练路线。原来的 `motion64 + Inspire12 = 76D` 模型及发布权重继续使用原入口。
Fast-WAM 的原生 Gaussian FM 接口见 [FASTWAM.md](policies/FASTWAM.md)，本页不将 Fast-WAM 的 LIBERO/RoboTwin 权重转换成 G1 控制器。

## 输出与数据字段

| 配置 | 动作目标 | 躯干姿态来源 | SONIC 接法 |
|---|---|---|---|
| `joint41` | 身体关节29 + 左右实测 Inspire 各6 | 必须另外提供未来世界坐标系 wxyz 四元数轨迹 | v1 joint-reference encoder，再经 decoder |
| `joint47` | joint41 + 世界坐标系躯干 rotation6D | 模型预测，转成 wxyz 四元数 | 同上 |
| 旧 `motion76` | motion token64 + Inspire12 | 原有 motion-token 路线 | 原发布包的 token 接口 |

`joint41` 不是只给 SONIC 29 个数就能闭合的控制链。它还缺未来姿态轨迹。
`joint47` 同时预测姿态，但目前两种关节模型都没有训练权重或真机效果记录。

默认 `body_target=wbc`，身体标签来自 `action.wbc` 的 `[0:22] + [29:36]`，共29维。
这是旧控制器解码后记录的电机位置目标，不能称为原始遥操作运动参考。
新模型预测这种目标再送 SONIC encoder/decoder，会形成一条新的控制链；代码明确记录
`body_target_source=wbc_motor_qtarget`，并将控制等价性标为未验证。
`--body-target measured` 是另一种显式实验，身体标签改用 `observation.state` 的相同29维，不能与 wbc checkpoint 或统计互换。

左右手仍取同一目标行的 `observation.inspire_left_hand_state` 和
`observation.inspire_right_hand_state`。每手按 `[pinky, ring, middle, index, thumb_bend, thumb_rotation]` 排列，物理范围为 `[0,1]`。
这12维是连续实测反馈，不是开关值，也不是 SONIC 原生 Dex3 的左右各7维命令。

47D 的最后6维由 `teleop.body_quat_w` 转换，顺序是旋转矩阵前两列
`[R00,R10,R20,R01,R11,R21]`。需要同时保留 `teleop.stream_mode` 和
`teleop.smpl_frame_index`，有效条件是 mode 属于 `{1,4}` 且 SMPL 帧号大于0。
缺少有效性字段会拒绝准备47D数据。记录器在信号丢失时写入的单位四元数不作为真实姿态标签。
无效时刻只屏蔽最后6维 FM/IC 损失，并将这些目标和历史维度在归一化后置零；其余41维继续训练。
姿态归一化统计只用有效姿态行，不删 episode，不创建验证集。

当前状态始终是 `body29 + Inspire12 + projected_gravity3 = 44D`，只取当前 anchor。
state 通过动作专家的独立投影进入动作 tokens，视觉和文本均不含数值 state。
未来关节、手和姿态只作为动作标签。视觉 FM/IC 单独更新视觉分支，动作损失通过 stop-gradient 阻断对视觉分支的梯度。

## H16 的时间口径

该路线固定30 FPS、history16、horizon16、stride2。每个 anchor 为 `t`：

| 内容 | 原始帧偏移 |
|---|---|
| 历史动作 | `-32,-30,...,-2` |
| 监督动作 | `0,2,...,30` |
| 当前 / 未来图像 | `0 / 32` |
| 当前 state | 仅 `0`，在批数据中重复16份 |

16个动作参考点覆盖1秒。不能把30 FPS连续16帧的0.5秒输出冒充同一方案。
训练窗口不跨 episode，边界有显式 padding mask；历史不足的位置使用归一化零。
训练末尾的 padding 不代表拥有完整部署未来轨迹。SONIC 离线桥接要求完整时间覆盖，拒绝补成虚假的未来窗口。

图像为当前及未来 ego/chest，每路 resize224×224，按 ego左、chest右横拼224×448。
视觉 source 为 `0.5*z_current + sqrt(0.75)*epsilon`，即当前 latent 的平方系数占25%。
动作 source 为 `normalized_past + 0.5*epsilon`，全部41/47维加噪，不做额外方差归一化。
视觉 FM/IC 总权重0.5，视觉 IC 权重1；动作 FM 权重1、动作 IC 权重0.5，两侧 IC 均为一步 L1。

## 准备与训练命令

从仓库根目录运行。环境和基础资产按 [G1_WHOLEBODY_DELTA.md](G1_WHOLEBODY_DELTA.md#4-环境与外部模型)
和 [MODEL_INITIALIZATION.md](MODEL_INITIALIZATION.md) 准备，使用独立 ImageWAM 环境。
冻结 AE/Qwen，缓存纯 instruction，FLUX.2 视觉 DiT 与动作专家全量微调。
共同的114个基础初始化张量可复用，41/47维动作输入/输出与 state 投影重新初始化。
旧76D任务模型或优化器不会被静默部分加载。
新的 joint checkpoint 将本任务的统计 SHA 和 instruction SHA 写入协议。
同为41/47维也不能换成其他任务的统计或文本缓存继续加载。

以下演示47D；41D将 `REPRESENTATION=joint41`、`ORIENTATION=external`，输出目录也要另取名称。
两种配置独立准备自己的统计。变量均需指向用户已准备的本地资产：

```bash
export PYTHONPATH="$(pwd)/src:$(pwd)/packages/humanwam/src${PYTHONPATH:+:$PYTHONPATH}"
export DATASET_DIR=/path/to/lerobot_dataset
export WORK_DIR=/path/to/new_joint47_run
export REPRESENTATION=joint47
export ORIENTATION=predicted_rotation6d
export FLUX2_SRC=/path/to/flux2
export FLUX2_MODEL_PATH=/path/to/flux-2-klein-base-4b.safetensors
export FLUX2_AE_MODEL_PATH=/path/to/ae.safetensors
export QWEN3_MODEL_DIR=/path/to/Qwen3-4B
export ACTION_INITIALIZER=/path/to/flux2_shared_partial_initializer.pt

python -m imagewam.datasets.wholebody_joint_reference \
  --dataset-dir "$DATASET_DIR" --output-dir "$WORK_DIR/inputs" \
  --body-target wbc --orientation-target "$ORIENTATION" \
  --action-horizon 16 --past-action-size 16 --action-stride 2

python scripts/wholebody/cache_text.py \
  --dataset-dir "$DATASET_DIR" --cache-dir "$WORK_DIR/textcache" \
  --qwen-model "$QWEN3_MODEL_DIR" --flux2-src "$FLUX2_SRC"

python scripts/wholebody/build_config.py \
  --representation "$REPRESENTATION" --horizon 16 \
  --dataset-dir "$DATASET_DIR" --stats-path "$WORK_DIR/inputs/dataset_stats.json" \
  --text-cache-dir "$WORK_DIR/textcache" --output-dir "$WORK_DIR/configs" \
  --flux2-src "$FLUX2_SRC" --flux2-model "$FLUX2_MODEL_PATH" \
  --ae-model "$FLUX2_AE_MODEL_PATH" --qwen-model "$QWEN3_MODEL_DIR" \
  --action-initializer "$ACTION_INITIALIZER" \
  --epochs 60 --batch-size 8 --accumulation 2 --num-processes 8
```

统计准备检查原始 metadata、Parquet、时间和字段，保存 SHA；不会修改源数据。
配置生成在 CPU 上实例化真实 dataset/loader 并测量每轮步数，不进行模型训练。
8卡、每卡8、累积2给出名义全局 batch128；总更新数、5% warmup 和保存间隔按实测 loader 推导。
这组新的关节维度配置尚未完成8卡显存验证，不能把配置默认值当成已跑通的结果。
峰值 LR1e-4、cosine至1e-6、WD0.01、BF16、seed42、ZeRO-1，不强制 FlashAttention。

正式训练前先完成同配置五次 optimizer update。试跑不保存模型或优化器；正式训练重新从基础初始化开始：

```bash
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 scripts/wholebody/train.py \
  --config "$WORK_DIR/configs/smoke.yaml" --smoke

accelerate launch --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 8 scripts/wholebody/train.py \
  --config "$WORK_DIR/configs/production.yaml"
```

正式入口核对所有 rank 的五步检查、初始化、统计、维度与 schedule。输出目录必须是新目录，不能覆盖旧实验。
每轮保存一次，保留最近4份权重和最新1份训练状态。需要其他轮数或 batch 时，重新生成完整配置和试跑。

## 推理与 SONIC 离线导出

推理配置见 `configs/humanwam/imagewam-joint41.json` / `imagewam-joint47.json`。
设置 `IMAGEWAM_PYTHON`、`IMAGEWAM_JOINT_MODEL_CONFIG`、`IMAGEWAM_JOINT_CHECKPOINT`、
`IMAGEWAM_JOINT_TEXT_CACHE` 后，使用与原来相同的 `humanwam predict` / `serve` 接口。
`imagewam-joint-train.json` 提供统一 launcher 的训练入口。

输入为当前 uint8 HWC `ego_rgb`、`chest_rgb`，物理量 `current_state[44]`，以及已按stride2采样的
`past_actions[N,41/47]`。必须同时声明 `past_action_frame_offsets=[-2*N,...,-2]`，N最多16。
episode起点可用空历史，但须显式给 `at_episode_start=true`。不接受未来 state、真实未来图像或预归一化历史。
返回的是物理单位关节参考与单独的手反馈预测，不会下发电机。

SONIC官方 v1.1 G1 encoder 使用46帧50Hz参考窗口。离线桥接对1秒稀疏预测做关节插值、姿态 SLERP，
得到51帧50Hz参考，计算关节速度，核对29个关节的名字和顺序，再产生 v1 body packet。
它不把41/47维塞进64D motion-token字段，也不把Inspire12塞进Dex3的14维字段。

完整的 `split_model_prediction`、NPZ 字段与导出命令见
[SONIC_JOINT_REFERENCE.md](policies/SONIC_JOINT_REFERENCE.md)。
41D需要明确提供完整未来姿态，不能用当前躯干姿态或单位四元数静默替代。
47D使用预测6D旋转做 Gram–Schmidt 转换，退化旋转、越界手值、NaN、归一化空间输出或不足时间覆盖都会拒绝导出。

## 验证范围

验证覆盖原生 FM/IC、state 与视觉隔离、动作到视觉 stop-gradient、H16 推理形状、checkpoint schema 隔离，
以及41/47数据映射与实际视频解码。SONIC packet 与固定版本官方 packer 字节一致，并通过官方 unpacker 往返检查。
这些都是离线代码/数据检查。尚未运行新的41/47完整 FLUX.2 GPU训练、SONIC encoder/decoder闭环或真机执行。
