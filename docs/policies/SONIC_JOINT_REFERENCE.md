# G1 joint29＋Inspire12 到 SONIC 的离线参考桥接

这是新增关节目标训练路径的**离线导出**，不启动 ZMQ、不连接机器人，不修改低层控制器。它与现有 `motion64＋Inspire12` 策略是两套动作语义。输入41维不能重命名为64维 motion token，也不能用旧76维模型的归一化统计。

实现位于 `humanwam.sonic_joint_reference`，命令入口为 `scripts/humanwam/export_sonic_joint_reference.py`。只依赖 NumPy 和 Python 标准库，不导入 Isaac Lab、Torch、ROS、Unitree SDK 或 ZMQ。

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

输入数值本身不能证明“已反归一化”或语义正确，调用者必须保留 checkpoint 的动作字段映射、统计和目标来源。数据中的 `action.wbc` 若是 SONIC decoder 下发的电机关节qtarget，与原始运动参考qpos不是同一个信号。将qtarget再次作为SONIC参考会构成不同控制链，不能只因维度相同就宣称等价，必须单独验证。`observation` 的实测关节也不能冒充动作指令。

本桥接只处理**所提供参考序列的时间轴**。第一条参考对应`frame_index_start`，其真实`relative_timestamps[0]`完整保留，不自动断言它等于当前观测时刻。例如首个预测在未来2/30秒，导出也从2/30秒开始，不向当前时刻反填。真实接入还需明确第一条预测的未来offset、推理延迟与控制器时钟对齐。`recorded_future`可用于离线核验，不能当作机器人运行时可获得的未来状态。

## 两种模型输出

`split_model_prediction` 支持以下显式schema，调用前仍须使用自己的checkpoint统计完成反归一化。

- `external41` 为 body29＋Inspire12。调用时必须另外提供每个未来采样时刻的root四元数，不会复制当前姿态。
- `predicted_rotation6d47` 为 body29＋Inspire12＋rotation6D。后6维是旋转矩阵前两列依次拼接，顺序为 `[R00,R10,R20,R01,R11,R21]`，不是flatten前两行。Gram-Schmidt转正交矩阵后生成wxyz四元数，零向量与平行列会被拒绝。该输出应由新训练模型预测，不能从41维旧checkpoint凭空取得。

```python
from humanwam.sonic_joint_reference import split_model_prediction

fields = split_model_prediction(
    denormalized_actions47,
    prediction_schema="predicted_rotation6d47",
    normalized=False,
    body_target_source="wbc_motor_qtarget",
)
# fields: joint_positions[N,29], inspire_hand_states[N,12],
#         root_quaternions_wxyz[N,4], body_target_source, root_orientation_source
```

## SONIC v1.1 的未来窗口

控制参考重采样到50Hz。模式0的10个观测offset为 `0,5,10,...,45`，所以第一窗口必须有至少46个连续控制帧，跨度0.9秒。协议版本v1与控制器发布版本v1.1是不同概念。

| 16个源样本的时间轴 | 覆盖范围 | 导出结果 |
|---|---:|---|
| `arange(16)/30` | 0.5秒 | 拒绝，只能形成26帧 |
| `arange(16)*2/30` | 1.0秒 | 51个50Hz参考帧，满足第一窗口 |

第二种设置要求模型训练目标本来就是stride2采样，不能把连续16帧预测的timestamp乘2后声称有了更远未来。导出器不改变训练horizon，不做末帧padding，不外推缺失未来。额外的未来窗口只能在其offset仍有真实覆盖的范围内使用。

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
    relative_timestamps=np.arange(1, 17, dtype=float) * 2 / 30,
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

## 已核对的原生协议

依据固定官方WBC commit [`1e1ec4f5f27e6c3e03048f693bd7c60cce04afda`](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda)。

- [官方纯NumPy packer](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/utils/teleop/zmq/zmq_planner_sender.py#L161) 的真实布局为 `b"pose"`＋1280字节null填充JSON＋各字段的C连续little-endian bytes。部分旧docstring仍写1024，代码常量与C++均为1280。
- 导出header固定`v=1,endian=le,count=1`，只含body的四个必需字段。`count=1`遵循官方packer，不用它代替数组N维帧数。body为f32，frame index为i64。
- [官方协议说明](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/docs/source/tutorials/zmq.md#L236) 要求29个有意义的IsaacLab顺序关节值。permutation已与[G1映射](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/envs/manager_env/robots/g1.py#L93)、原生MJCF和部署参数交叉检查。
- CPU测试将本实现bytes与官方packer逐字节比较，并只用AST提取[官方unpack函数](https://github.com/NVlabs/GR00T-WholeBodyControl/blob/1e1ec4f5f27e6c3e03048f693bd7c60cce04afda/gear_sonic/scripts/run_data_exporter.py#L120)进行roundtrip。未实例化原生subscriber，未启动C++控制器，不能称为真机或网络链路验证。

```bash
python -m pytest packages/humanwam/tests/test_sonic_joint_reference.py
```

测试覆盖46帧窗口边界、H16采样覆盖不足拒绝、关节重排、差分速度、SLERP、rotation6D退化拒绝、输入单位/归一化/语义检查、协议字节一致性、损坏packet拒绝，以及离线CLI无覆盖写入。后续接入必须沿用匹配的SONIC encoder、decoder和observation config，不能用该导出器把旧motion64模型转换为关节预测模型。
