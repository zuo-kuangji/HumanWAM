# G1 recording label audit, 2026-11-10

This audit inspected the existing G1 recorder and selected dataset schemas over a read-only SSH session. No robot process, training job, dataset, or controller was changed. No video, numerical demonstration, credential, or model weight is included here. Paths below are on the audited G1 host, under `/home/g1/zuo/GR00T-WholeBodyControl`.

## Recorded fields and their actual sources

The clothing collection `outputs/wb_my_task_sonic_v11` has 98 episodes, the trash-bin collection `outputs/wb_trash_bin_sonic_v11` has 81, and the cleaned pillow collection `outputs/wb_chaise_pillow_to_bedhead_sonic_v11_pause_cleaned_all_20261030-054852` has 78. Their `meta/info.json` files declare 30 FPS. A Parquet schema from each collection contains `observation.state[43]`, `action.wbc[43]`, `observation.root_orientation[4]`, `teleop.body_quat_w[4]`, and `teleop.target_body_orientation[6]`. Presence was checked from actual Parquet schemas, not inferred from a training config.

| Field | Recorder source | Interpretation |
|---|---|---|
| `observation.state` | `proprio["body_q"]` plus hand feedback | Measured configuration, converted to the robot-model joint order |
| `action.wbc` | `proprio["last_action"]` plus last hand actions | Previous WBC decoder motor position target; not the original motion reference |
| `observation.root_orientation` | `proprio["base_quat"]` | Raw base/pelvis IMU quaternion in wxyz order |
| `teleop.body_quat_w` | SMPL pose stream `body_quat_w` | Human root/pelvis orientation after the manager's SMPL coordinate transforms |
| `teleop.target_body_orientation` | Separate heading correction and episode yaw normalization | Not an alias for either quaternion field |

`gear_sonic/scripts/run_data_exporter.py:909–934` constructs the measured and WBC arrays separately:

```python
whole_q = self.robot_model.get_configuration_from_actuated_joints(
    body_actuated_joint_values=proprio["body_q"], ...)
whole_action_wbc = self.robot_model.get_configuration_from_actuated_joints(
    body_actuated_joint_values=proprio["last_action"], ...)
frame_data = {"observation.state": whole_q, "action.wbc": whole_action_wbc, ...}
```

Both 43D arrays order the body as `left_leg6, right_leg6, waist3, left_arm7, left_virtual_hand7, right_arm7, right_virtual_hand7`. Selecting `[0:22] + [29:36]` returns the 29 body joints in the dataset's MuJoCo/kinematic traversal order. Names must still be checked from metadata and explicitly reordered before an IsaacLab-order SONIC packet is built.

## Base IMU frame and heading metadata

The C++ control source `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp:2909–2915` reads two distinct IMUs:

```cpp
base_quat = float_to_double<4>(ls->imu_state().quaternion()); // qw,qx,qy,qz
body_torso_quat = float_to_double<4>(imu_torso->quaternion());
```

The base quaternion and measured body joints enter the same state-logger snapshot at lines 2943–2946. `src/state_logger.cpp:109–119` copies them without a heading transform. `include/output_interface/zmq_output_handler.hpp:299–301` packs `state.base_quat` directly. The Python recorder writes this value directly to `observation.root_orientation` at lines 985–989. The separate torso IMU is not used for this dataset field.

Here `world` means the raw base IMU estimator's world/reference frame. The audit verifies the source and lack of an additional software heading transform. It does not establish geographic north, persistence across IMU reinitialization, or a shared world calibration across recording sessions.

`observation.cpp_rotation_offset` stores the initial **motion-reference** root quaternion `init_ref_data_root_rot_array`; `observation.init_base_quat` stores the robot quaternion at heading initialization. C++ `ComputeApplyDeltaHeading()` at lines 590–602 computes

```text
Rz(delta_heading) * heading(init_base_quat) * inverse_heading(initial_reference_root)
```

to align reference motion. Those operations do not modify the raw measured `base_quat` that the recorder stores. A shared state-logger snapshot is evidence of same-stage measured q/root pairing, not proof of hardware sensor synchronization. The sample source contract therefore records `same_stage_pairing=true` and `synchronization_verified=false`.

## Teleoperation root is a different frame

`gear_sonic/scripts/pico_manager_thread_server.py:157` and `:273` identify joint zero as root/pelvis, not torso. `compute_from_body_poses()` at lines 578–603 takes joint-zero global rotation, converts coordinate conventions, and passes it through `process_smpl_joints()` at lines 478–491. The resulting quaternion is copied/interpolated into the pose stream at lines 1400–1431 and 1507–1533. This is the source of `teleop.body_quat_w`.

The exporter writes this raw teleop quaternion at lines 1088–1092. Its separate `_compute_target_body_orientation()` at lines 1207–1228 first computes `Rz(teleop.delta_heading) * body_rotation` and then subtracts the episode's first yaw. Neither operation makes the raw teleop quaternion identical to the measured robot base quaternion. Combining WBC qtargets or measured robot joints with raw teleop orientation requires an explicit experimental contract; this audit does not establish same-stage or common-frame equivalence for that pairing.

## Original 29D reference availability

The inspected Parquet schemas do not contain a full original 29D joint-reference position trajectory. In the current SMPL pose sender, `joint_pos` is initialized as 29 zeros at line 1436 and only six wrist entries are assigned at lines 1487–1493; `joint_vel` is sent as zeros at line 1533. The exporter extracts only left/right wrist triples from that array at lines 695–710. These values must not be labeled as a full original G1 motion reference.

The existing measured q plus measured root pair can support an explicitly labeled `measured` prediction experiment. It does not turn a measured trajectory into an original reference demonstration. `action.wbc` can support a separately labeled `wbc` experiment. A `direct_reference` recipe needs newly verified position/quaternion fields, joint names, coordinate convention, synchronization provenance, and an explicit source contract. No direct-reference source contract is provided for these collections.

## Source fingerprints

SHA256 values were computed on the inspected G1 files. They identify this audit snapshot; future source edits require re-audit.

| Relative source path | SHA256 |
|---|---|
| `gear_sonic/scripts/run_data_exporter.py` | `c2031f73325b7202a3e9fc6ba111f4327faa815053874884467e713c6b71e8a5` |
| `gear_sonic/scripts/pico_manager_thread_server.py` | `b0208304450eb4421b1a6af72a8ac074b09bce839db26a1367febacf82d162a4` |
| `gear_sonic/data/features_sonic_vla.py` | `603895ff94daab7019149d68a30a96218507cf31bd9037c1c6b740621f219759` |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp` | `312b41e376f4302938994e8d47b25004ec4360fa7bdb8e6dcb8574a80fe9c82f` |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/zmq_output_handler.hpp` | `50c39ae3d7f10eb1ce8e9ef328615f0c8baf6b45851c89beb9d8c105d91f1ccf` |
| `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/output_interface/output_interface.hpp` | `6e3d2434d7988c42e5b420dc49832c36c01d3e3c1f606cc9ab033f752ac09d08` |

This audit establishes recorded-field semantics. It does not establish decoder/encoder equivalence, offline task quality, physical rollout success, or that a new joint-reference model has been trained.
