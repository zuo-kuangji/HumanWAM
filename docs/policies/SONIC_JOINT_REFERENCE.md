# G1 joint47 到 SONIC 的参考与时钟边界

这是新增关节目标训练路径的**离线导出**，不启动 ZMQ、不连接机器人，不修改低层控制器。它与现有 `motion64＋Inspire12` 策略是两套动作语义。输入41维不能重命名为64维 motion token，也不能用旧76维模型的归一化统计。

实现位于 `humanwam.sonic_joint_reference`，命令入口为 `scripts/humanwam/export_sonic_joint_reference.py`。只依赖 NumPy 和 Python 标准库，不导入 Isaac Lab、Torch、ROS、Unitree SDK 或 ZMQ。

主训练路线为[新joint47 v2](../G1_JOINT_REFERENCE_DELTA.md)，history16、prediction40、execution16、30FPS stride2。
`humanwam.sonic_joint_runtime` 另提供带单调时钟的参考计划与滚动窗口检查，也不含socket或硬件控制。
两个模块都不替代SONIC encoder、decoder、观测配置和真实执行端。

## 输入与语义

| 字段 | 形状／取值 | 要求 |
|---|---|---|
| `joint_positions` | `[N,29]` | G1绝对关节位置，单位rad，已用本checkpoint统计反归一化 |
| `joint_names` | 29个唯一名称 | 必须显式列出下文canonical关节，禁止凭数组长度猜顺序 |
| `joint_order` | `isaaclab` / `mujoco` / `named` | 前两者必须与完整声明顺序一致，named允许显式重排 |
| `root_quaternions_wxyz` | `[N,4]` | 每个参考时刻的世界系root/pelvis旋转，标量在前，单位四元数 |
| `relative_timestamps` | `[N]` | 非负秒数，严格递增，保留相对观测的真实未来offset |
| `inspire_hand_states` | `[N,12]` | 左6＋右6的实测状态目标，raw反馈范围`[0,1]` |
| `normalized` | `False` | 显式声明，True或缺失均拒绝 |
| `joint_units` | `rad` | 不做degree、PD residual或默认姿态转换 |
| `action_semantics` | `absolute_joint_position_reference` | 显式声明输入拟作为关节参考，不是SONIC64 token |
| `body_target_source` | `wbc_motor_qtarget` / `measured_joint_position` / `direct_joint_reference` | 必须保留训练标签原始来源，不静默互换 |
| `root_orientation_source` | `predicted_future` / `recorded_future` | 禁止用当前IMU或单位旋转静默补齐未来 |
| `frame_index_start` | 非负整数 | 第一条参考的全局控制帧编号，后续逐帧加1 |
| `joint_velocities` | 可选`[N,29]` | 同关节顺序，单位rad/s；缺失时显式差分并记录方法 |

输入数值本身不能证明“已反归一化”或语义正确，调用者必须保留checkpoint的字段映射、统计、source contract及其SHA。已有数据的`action.wbc`是WBC decoder电机qtarget，`observation.state`是实测关节，均不能称为原始运动参考。重新作为SONIC参考时，两者分别构成实验控制链。来源证据见[录制器审计](REFERENCE_LABEL_AUDIT_20261110.md)。

`root_quaternions_wxyz`必须已恢复到声明的world frame。新joint47 v2的最后6维表示`current_heading_relative`，先用当前base/pelvis heading还原world，不能直接当作world quaternion。root不是torso，也不含xyz。当前anchor只用于坐标换算，不是未来state或视觉条件。

本桥接只处理**所提供参考序列的时间轴**。第一条参考对应`frame_index_start`，其真实`relative_timestamps[0]`完整保留，不自动断言它等于当前观测时刻。例如首个预测在未来2/30秒，导出也从2/30秒开始，不向当前时刻反填。真实接入还需明确第一条预测的未来offset、推理延迟与控制器时钟对齐。`recorded_future`可用于离线核验，不能当作机器人运行时可获得的未来状态。

## 模型输出与坐标转换

`split_model_prediction` 支持以下显式schema，调用前仍须使用自己的checkpoint统计完成反归一化。

- `external41` 为 body29＋Inspire12。调用时必须另外提供每个未来采样时刻的root四元数，不会复制当前姿态。
- `predicted_rotation6d47` 为 body29＋Inspire12＋rotation6D。后6维是旋转矩阵前两列依次拼接，顺序为 `[R00,R10,R20,R01,R11,R21]`，不是flatten前两行。Gram-Schmidt转正交矩阵后生成wxyz四元数，零向量与平行列会被拒绝。该输出应由新训练模型预测，不能从41维旧checkpoint凭空取得。

低层`split_model_prediction`不知道训练anchor，下面例子仅适用于已恢复到world的47D物理数据。
v2策略backend先执行`R_world=Rz(yaw_anchor)@R_relative`，再提供world四元数。不能仅凭shape判断frame。

```python
from humanwam.sonic_joint_reference import split_model_prediction

fields = split_model_prediction(
    denormalized_actions47,
    prediction_schema="predicted_rotation6d47",
    normalized=False,
    body_target_source="measured_joint_position",
)
# fields: joint_positions[N,29], inspire_hand_states[N,12],
#         root_quaternions_wxyz[N,4], body_target_source, root_orientation_source
```

## 未来窗口必须覆盖整个执行段

控制参考重采样到50Hz。模式0的10个观测offset为 `0,5,10,...,45`，所以第一窗口必须有至少46个连续控制帧，跨度0.9秒。协议版本v1与控制器发布版本v1.1是不同概念。

| 源样本时间轴 | 覆盖范围 | 说明 |
|---|---:|---|
| `arange(16)/30` | 0.5秒 | 不满足第一窗口 |
| `arange(16)*2/30` | 1.0秒 | 仅满足首个窗口，不足以持续执行16个源间隔并保留0.9秒lookahead |
| `arange(40)*2/30` | 2.6秒 | v2真实预测40点，前16个源间隔用于执行，其余为lookahead |

训练目标必须确实采用stride2和40点预测，不能修改timestamp冒充更远未来。离线导出器检查首个窗口；带时钟的`JointReferenceBuffer`还检查推理延迟、jitter与整个执行段。执行时长为`16*2/30=1.0667秒`，不是16个50Hz tick。低层控制50Hz，encoder的10个lookahead点相距0.1秒，两者都不是模型预测频率。

导出器和runtime均不做末帧padding、未来外推或单位姿态替代。训练source最后24点重复最后历史，仅用于构造source先验；部署时全部40点必须是真实模型预测。

关节位置与Inspire状态用线性插值，root旋转用最短弧SLERP，处理q与−q的等价性。若输入没有速度，在重采样位置上用`np.gradient(..., 1/50, edge_order=1)`求rad/s，并在manifest记录。若提供速度，则按同顺序重排后插值。不会将角度自动wrap、添加默认站姿或进行动作裁剪。

## 完整导出示例

先准备一个包含上述字段的NPZ，字符串和布尔值用普通NumPy scalar保存，不使用object数组。下面只示范已有模型预测的封装，不生成伪造未来姿态。

```python
import numpy as np
from humanwam.sonic_joint_reference import G1_ISAACLAB_JOINT_NAMES

np.savez(
    "joint_reference_input.npz",
    **fields,
    joint_names=np.asarray(G1_ISAACLAB_JOINT_NAMES),
    joint_order="isaaclab",
    relative_timestamps=np.arange(40, dtype=float) * 2 / 30,
    normalized=False,
    joint_units="rad",
    action_semantics="absolute_joint_position_reference",
    frame_index_start=1000,
)
```

在已安装轻量`humanwam`包的环境运行，不会执行任何发布操作。

```bash
python scripts/humanwam/export_sonic_joint_reference.py \
  --input joint_reference_input.npz \
  --output /absolute/path/offline/reference50hz.npz \
  --wire-output /absolute/path/offline/pose_v1.bin
```

输出为50Hz NPZ、同名`.npz.json`语义manifest及可选body-only协议bytes。已有目标文件一律拒绝覆盖。输入或未来覆盖检查失败时返回非零，报告缺失窗口，不写输出。

NPZ中的body字段为 `joint_pos[N,29]`、`joint_vel[N,29]`、`body_quat[N,1,4]`、`frame_index[N]`，另存 `relative_timestamps`、`inspire_hand_states[N,12]`、canonical `joint_names` 和 `metadata_json`。Inspire12完全独立，不能装入官方Dex3左右各7维字段。这里没有实现Inspire硬件控制桥。

## 带时钟的计划检查

主入口`TimedJointPlan.from_policy_prediction`接收backend返回的完整预测dict，检查v2配方、source contract与canonical SHA。world frame尚未验证的来源拒绝映射；measured/WBC实验需显式`allow_experimental_source=True`，并继续记录同步/控制链尚未验证。它不会把这个选择变成真机执行授权。低层`from_prediction`仅是显式物理数组接口，不能代替checkpoint/provenance检查。

观测采集、推理完成和消费时刻必须来自同一声明的monotonic clock，不能混入wall-clock或另一台主机未经对齐的时钟。

```python
from humanwam.sonic_joint_runtime import TimedJointPlan, JointReferenceBuffer

# prediction是当前真实backend返回的完整dict，不是手写数组或过期结果。
plan = TimedJointPlan.from_policy_prediction(
    prediction,
    allow_experimental_source=True,  # 现有measured/WBC实验；不表示控制器已验证
)
buffer = JointReferenceBuffer(
    control_epoch_time_s=controller_epoch_monotonic,
    execution_horizon=16,
    action_interval_s=2/30,
    jitter_budget_s=0.05,
    clock_id=plan.clock_id,
)
buffer.submit(plan, now_s=actual_submit_monotonic)
window = buffer.window(now_s=actual_control_monotonic)
# window.reference / window.body_packet / window.inspire_hand_states
# window.absolute_timestamps_s / execution_deadline_s / diagnostics
```

这里只构建与验证窗口，不发布packet。`body_packet`是按统一50Hz时间网格切出的body-only协议bytes；Inspire12另行返回。未实现Inspire硬件控制桥。在线计划只接受预测未来root，不接受录制未来或当前IMU复制成的未来。

等价的离线命令接受backend完整JSON，输出目录必须是新目录。时间参数必须来自这次真实调用和同一控制时钟，不能为过期JSON随意指定一个新时间。

```bash
python scripts/humanwam/build_sonic_joint_window.py \
  --input /path/to/current_prediction.json --output-dir /path/to/new_window \
  --now "$ACTUAL_MONOTONIC_NOW" --control-epoch "$ACTUAL_CONTROL_EPOCH" \
  --clock-id local_monotonic --jitter-budget 0.05 --allow-experimental-source
```

产出`body_pose_v1.bin`、独立`inspire_reference50hz.npz`和`window_manifest.json`，不启动网络、encoder或机器人。

首次消费才承诺本次执行段，重复读取不会延长deadline。状态区分`empty`、`ready`、`executing`、`stale`与`overrun`。缺少新计划、时间倒退、跨clock ID、结果过旧或剩余lookahead不足会拒绝，不自动hold、blend、跨段拼接或延长执行。真实执行端须处理这些返回并实施自己的停止逻辑。

默认2.6秒预测覆盖必须同时容纳推理耗时、到下一个50Hz tick的取整、1.0667秒执行、0.9秒lookahead及0.05秒jitter。名义延迟余量约0.5833秒，还需减去时钟网格取整；实际以检查器计算为准。关节在同一完整50Hz网格上求速度后再切窗口，重复窗口在相同timestamp上的qvel保持一致。轨迹来源、时间轴和物理速度不能与FM噪声时间上的flow velocity混淆。

## 已核对的原生协议

依据固定官方WBC commit [`1e1ec4f5f27e6c3e03048f693bd7c60cce04afda`](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda)。

- [官方纯NumPy packer](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py#L161) 的真实布局为 `b"pose"`＋1280字节null填充JSON＋各字段的C连续little-endian bytes。部分旧docstring仍写1024，代码常量与C++均为1280。
- 导出header固定`v=1,endian=le,count=1`，只含body的四个必需字段。`count=1`遵循官方packer，不用它代替数组N维帧数。body为f32，frame index为i64。
- [官方协议说明](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/docs/source/tutorials/zmq.md#L236) 要求29个有意义的IsaacLab顺序关节值。permutation已与[G1映射](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/envs/manager_env/robots/g1.py#L93)、原生MJCF和部署参数交叉检查。
- CPU测试将本实现bytes与官方packer逐字节比较，并只用AST提取[官方unpack函数](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/scripts/run_data_exporter.py#L120)进行roundtrip。未实例化原生subscriber，未启动C++控制器，不能称为真机或网络链路验证。

```bash
python -m pytest packages/humanwam/tests/test_sonic_joint_reference.py
python -m pytest packages/humanwam/tests/test_sonic_joint_runtime.py
```

测试覆盖窗口边界、全执行段lookahead、延迟和clock错误、关节重排、差分速度与重叠窗口一致性、SLERP、rotation6D退化拒绝、输入语义、协议字节一致性、损坏packet拒绝和离线CLI无覆盖写入。它们未运行SONIC encoder/decoder或真机。后续接入必须使用匹配的模型与observation config；此导出器不能把旧motion64权重转换成关节预测模型。
