# Psi-Zero（Ψ₀）SONIC backend

官方源为 [physical-superintelligence-lab/Psi0](https://github.com/physical-superintelligence-lab/Psi0)，
固定 commit `cda3fdfafe47cb6fa75400d013de1bd36e3cea05`，Apache-2.0。
上游源码置于版本锁定的`third_party/Psi0`；不要复制`.env`、缓存、数据或权重进入Git。

## 环境与版本

Psi0 原生环境要求 Python3.11、NumPy<2、Torch2.7.0、Transformers4.57.1等，
与当前GR00T N1.7的环境不同。按官方README用独立`.venv-psi`配置，
并在该环境安装本仓库轻量包`pip install -e /path/to/HumanWAM/packages/humanwam`。
模型模块需在原生Psi环境安装并可`import psi`。CPU适配器测试不等于原生环境/GPU验证。

官方Psi0 SONIC使用自己的WBC fork：

```text
https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl.git
d40376994ff094787591ab76a691c50b8e131786
```

这个嵌套gitlink与HumanWAM顶层NVIDIA/NVlabs WBC不是同一个版本，不能互换。
只初始化该依赖时使用HTTPS地址，避免将本机SSH私钥当作项目安装条件：

```bash
git -C third_party/Psi0 config submodule.third_party/GR00T-WholeBodyControl.url \
  https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl.git
git -C third_party/Psi0 submodule update --init third_party/GR00T-WholeBodyControl
```

该命令仅准备代码，不构建WBC、不启用机器人，也不下载控制器权重。
实际客户端/机器人部署按[官方SONIC部署指南](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/real/SONIC/DEPLOYMENT.md)。

## 权重与动作空间

下载的完整run目录至少保留：

```text
RUN/argv.txt
RUN/run_config.json
RUN/checkpoints/ckpt_40000/
```

还需保留配置引用的统计、VLM/action组件、pooled text缓存等资源。
`ckpt_step`必须是明确整数；原生SONIC服务不接受`latest`字符串。
典型SONIC配方为action78/state43/H30，带neck的配方可为action80/state45/H30。
实际维度从`run_config.json`和原生加载结果核对，不把它们改成ImageWAM76D实测手状态。
输入`raw_state_dim`必须按数据实际定义声明，不能把padding后模型维度当作原始state布局。

统一示例对应官方`multi-task.psi-dream.2609092156`：实际action80、state45、H30、
camera `observation.images.head`、checkpoint40000。这里state45包含真实neck状态，
不能把旧43D观测补两个零代替。适配器会根据checkpoint中非零state统计拒绝这种缺失。
该结论已核对官方HF revision`74d7abaa61b51dfe4e95f539a83738c276ce27b2`的
[run_config.json](https://huggingface.co/USC-PSI-Lab/psi-model/blob/74d7abaa61b51dfe4e95f539a83738c276ce27b2/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156/run_config.json)、
`argv.txt`和checkpoint目录；仅下载小型metadata，没有下载模型权重。

## 离线predict

`humanwam.backends.psi0.build_policy(options)`提供持久模型、`predict`和`reset`。
输入为`image_rgb`（当前uint8 HWC单图）、`state_raw`（原始实测state）、`instruction`。
它复用官方SONIC `Server`的checkpoint配置重建、图像变换、state归一化和输出反归一化，
保持推理augmentation关闭。只调用模型完整chunk预测，不调用`Server.run`，
不建立RTC执行线程，不连接机器人，不把该离线路径称作异步RTC等价实现。

输出`actions[H,D]`为checkpoint反归一化的原生动作，附带`action_space=psi0_sonic_native`、
`normalized=false`及维度/来源信息。接口拒绝未来state、目标动作和无关history字段。
当前官方服务器预处理最终只消费`cam0`，因此此适配器拒绝多camera checkpoint。

## 原生RTC服务

```bash
export HUMANWAM_ROOT=/path/to/HumanWAM
export PSI0_PYTHON=/path/to/psi-env/bin/python
export PSI0_RUN_DIR=/path/to/downloaded-complete-run
humanwam serve --config configs/humanwam/psi0-sonic.json --dry-run
humanwam serve --config configs/humanwam/psi0-sonic.json
```

原生命令为`python -m psi.deploy.serve_psi0_sonic --policy psi --rtc --run-dir ... --ckpt-step 40000`，
使用官方RTC WebSocket协议，默认localhost8014。只启动策略服务器，不自动启动机器人或客户端。

## 训练

`psi0-train.json`使用原生`torch.distributed.run`和`scripts/train.py finetune_sonic_psi0_config`，
`PSI0_NATIVE_ARGS`指向JSON字符串数组，装入选定官方训练配方全部原生flag和值。
例如`["--exp","my-task","--train.max_training_steps","40000", ...]`只是格式示例，
还必须提供真实数据、统计和初始化目录，不能把它当成完整可运行的任务配方。
完整参考配置在固定上游的`scripts/train/psi0/finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh`。

训练入口保留原生`.env`要求，项目不会复制含令牌的`.env`。
统一launcher不运行上游训练shell，不执行其export/pkill预处理，也禁止`auto_tag_run`修改Git。
不同SONIC版本的action.body_token与action.body_token_v1_1、手/neck字段及统计必须保持对应。
现有ImageWAM76D数据不因加入registry而自动成为Psi0可用训练数据。

## 本次验证

11项CPU适配器测试和8个subtest通过，覆盖懒加载、实际原生API调用边界、一次归一化/反归一化、
维度拒绝、RGB/有限值、单图限制、checkpoint伴随文件、RTC服务argv及训练命令。
另对固定源码解析原生`Server`、`Psi0Model.predict_action`、`ServerConfig`和训练CLI入口进行静态契约检查。
没有加载真实Psi0权重、没有GPU推理或机器人执行证明。

真实官方metadata检查额外核对了36,340字节的run config、argv和dataset statistics，
并用缩减后的真实state统计做回归测试，拒绝将该state45模型的neck两个通道静默补零。
官方该checkpoint的`model.safetensors`为11,220,488,216字节，本次没有下载。

在NTU现有ImageWAM环境中以`CUDA_VISIBLE_DEVICES=''`执行原生SONIC服务`--help`，
实际在`fastapi`导入处失败；该环境也没有`tyro`、`qwen-vl-utils`，
Transformers4.56.1与Psi0原生4.57.1配方不同。因此这里不能声称原生Psi环境或模型推理已通过，
后续必须在独立Psi环境复验，不能向正在使用的ImageWAM环境直接混装依赖。
