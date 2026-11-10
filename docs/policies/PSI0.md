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

HumanWAM的Psi0训练与部署模板默认使用无颈部profile
`embodiment_profile="g1_neckless43_action78"`，保留原生state43、action78和H30：

| 项目 | 此profile的布局 |
|---|---|
| 当前state43 | 原生身体29D＋手部14D，不含neck或projected gravity |
| 预测action78 | SONIC motion token64D＋原生手部14D，不预测neck |
| 时间窗口 | 原生H30，仍须与实际checkpoint一致 |

表中说明维度组成；state保留原始字段顺序，动作按run的`action_keys`顺序拼接，不按表格重排数据。
这里的手部仍是原生14D，不是ImageWAM的measured Inspire12。
本项目ImageWAM的state44/action76不能直接交给该Psi0入口，44/76的Psi0适配尚未实现。
profile声明的是训练和部署契约，不代表已为你的当前任务训练了模型。
`PSI0_RUN_DIR`必须指向真正按无颈部43/78布局训练的完整run。
不能裁切旧45/80权重、补零观测或只改`run_config.json`冒充无颈部模型。

下载的完整run目录至少保留：

```text
RUN/argv.txt
RUN/run_config.json
RUN/checkpoints/ckpt_40000/
```

还需保留配置引用的统计、VLM/action组件、pooled text缓存等资源。
`ckpt_step`必须是明确整数；原生SONIC服务不接受`latest`字符串。
实际维度、repack字段顺序和统计从`run_config.json`与原生加载结果核对。
无颈部profile必须同时满足action78、raw/model state43，不能只满足其中一个维度。
输入`raw_state_dim`必须按数据实际定义声明，不能把padding后模型维度当作原始state布局。

此前核对的官方`multi-task.psi-dream.2609092156`是带neck的历史示例，
实际action80、state45、H30、camera `observation.images.head`、checkpoint40000。
它不是当前无颈部模板的部署权重。state45的两个neck通道统计非零，
不能把43D观测补两个零后用于该模型，也不能将其80D输出直接裁为78D。
该历史示例已核对官方HF revision`74d7abaa61b51dfe4e95f539a83738c276ce27b2`的
[run_config.json](https://huggingface.co/USC-PSI-Lab/psi-model/blob/74d7abaa61b51dfe4e95f539a83738c276ce27b2/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156/run_config.json)、
`argv.txt`和checkpoint目录；仅下载小型metadata，没有下载模型权重。

## 无颈部官方权重示例

已核对下列官方完整run的metadata，与当前无颈部profile的维度相符。
以下命令下载完整目录，不转换或裁切权重：

```bash
export PSI0_MODEL_ROOT=/path/to/psi0-models
hf download USC-PSI-Lab/psi-model \
  --revision 74d7abaa61b51dfe4e95f539a83738c276ce27b2 \
  --include 'psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223/*' \
  --local-dir "$PSI0_MODEL_ROOT"
export PSI0_RUN_DIR="$PSI0_MODEL_ROOT/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223"
```

该run的checkpoint为40000，model state43、action78、H30、observation horizon1；
单路camera为`observation.images.egocentric`，resize/crop为240×320，
state/action的43/78维统计嵌入`run_config.json`。
该run是官方carry-box任务示例，不是本项目其他任务的新训练权重。
远端权重声明为6,254,227,440字节，safetensors header也确认了state43/action78/H30的投影尺寸；
尚未下载整套权重并验证文件哈希。核对记录见[无颈部模型验证记录](PSI0_NECKLESS_VALIDATION_20261110.md)。
这是原生`action.body_token`的SONIC v1权重，不是`action.body_token_v1_1`的SONIC v1.1权重。
应使用配套版本的token、统计和SONIC接口，不能只因维度都是64D而互换token版本。
其保存配置中`model.rtc=false`；服务启动`--rtc`会使用test-time guidance，
不代表该checkpoint经过RTC训练。
本次核对metadata、权重header与原生加载路径，未加载这套大权重进行GPU推理或机器人测试。

## 离线predict

`humanwam.backends.psi0.build_policy(options)`提供持久模型、`predict`和`reset`。
输入为`image_rgb`（当前uint8 HWC单图）、`state_raw`（原始实测state）、`instruction`。
默认无颈部profile要求`state_raw[43]`，并检查原生预处理后的观测为`[1,1,43]`。
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
# PSI0_RUN_DIR使用上节下载路径，或另一套真正匹配的无颈部43/78完整run。
humanwam serve --config configs/humanwam/psi0-sonic.json --dry-run
humanwam serve --config configs/humanwam/psi0-sonic.json
```

原生命令为`python -m psi.deploy.serve_psi0_sonic --policy psi --rtc --run-dir ... --ckpt-step 40000`，
使用官方RTC WebSocket协议，默认localhost8014。只启动策略服务器，不自动启动机器人或客户端。
启动前按当前profile检查run的维度、repack与归一化统计；默认profile不会把官方带neck的run改造成无颈部run。

## 训练

`psi0-train.json`使用原生`torch.distributed.run`和`scripts/train.py finetune_sonic_psi0_config`，
`PSI0_NATIVE_ARGS`指向JSON字符串数组，装入真实数据、统计、初始化、训练调度等原生flag和值。
例如`["--exp","my-task","--train.max_training_steps","40000", ...]`只是格式示例，
还必须提供真实数据、统计和初始化目录，不能把它当成完整可运行的任务配方。
完整参考配置在固定上游的`scripts/train/psi0/finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh`。
该上游shell含neck字段及45/80设置，不能原样当成当前无颈部任务配方。

当前profile的launcher按`options.horizon`（默认30）、`options.action_keys`和
`options.state_keys`补齐以下受管参数。已有参数必须完全一致，重复或冲突时拒绝启动，
不会静默覆盖用户flag：

| 原生参数 | 默认值 |
|---|---|
| `--model.action-dim`、`--model.odim` | 78、43 |
| `--model.action-chunk-size`、`--data.transform.repack.action-chunk-size` | 同一个horizon，默认30 |
| `--model.action-exec-horizon` | 默认等于horizon，可用`options.action_exec_horizon`设为1至horizon之间的值 |
| repack与field的`pad-action-dim`、`pad-state-dim` | 两处均为78、43 |
| `--data.transform.repack.action-keys`、`--data.transform.field.stat-action-keys` | `action.body_token_v1_1`、`action[:14]` |
| `--data.transform.repack.state-keys`、`--data.transform.field.stat-state-keys` | `observation.state` |

新训练模板默认SONIC v1.1 token。若真实数据使用v1的`action.body_token`，
在配置`options`中显式设`"action_keys": ["action.body_token", "action[:14]"]`，并使用相同版本的统计。
若数据已把token64＋原生手部14按该顺序组装为真实78维`action`列，
可显式设`"action_keys": ["action"]`；这不是将ImageWAM76维动作补零。
state字段也必须真实符合43D，不能把gravity或measured Inspire12重新命名为原生手部14D。

新的43/78训练阶段应重建匹配的数据repack、统计、state/action投影及run配置。
共享VLM与动作Transformer可按原生初始化路径复用；如果action维度或chunk不同，
[固定源码的训练器](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/src/psi/trainers/finetune.py#L333-L356)
只继承动作头的`transformer_blocks`，其余相关投影新初始化。
这属于预训练初始化后的新训练，不是裁切带neck任务权重，也不是完整resume。
训练后的43/78完整checkpoint才用于当前无颈部profile部署。

训练入口保留原生`.env`要求，项目不会复制含令牌的`.env`。
统一launcher不运行上游训练shell，不执行其export/pkill预处理，也禁止`auto_tag_run`修改Git。
不同SONIC版本的action.body_token与action.body_token_v1_1、手/neck字段及统计必须保持对应。
现有ImageWAM76D数据不因加入registry而自动成为Psi0可用训练数据。

仅在复现其他原生完整run时显式使用`embodiment_profile="checkpoint_native"`，
并声明其真实raw state维度。该模式按原checkpoint读取，不注入上述43/78训练参数，
也不会把历史45/80模型改成无颈部模型。

## 已有验证边界

此前11项CPU适配器测试和8个subtest通过，覆盖懒加载、实际原生API调用边界、一次归一化/反归一化、
维度拒绝、RGB/有限值、单图限制、checkpoint伴随文件、RTC服务argv及训练命令。
另对固定源码解析原生`Server`、`Psi0Model.predict_action`、`ServerConfig`和训练CLI入口进行静态契约检查。
没有加载真实Psi0权重、没有GPU推理或机器人执行证明。

上述官方带neck模型的metadata检查额外核对了36,340字节的run config、argv和dataset statistics，
并用缩减后的真实state统计做回归测试，拒绝将该state45模型的neck两个通道静默补零。
官方该checkpoint的`model.safetensors`为11,220,488,216字节，本次没有下载。
这些历史检查不代表新的43/78任务已训练、已加载真实权重或已完成机器人部署。

在NTU现有ImageWAM环境中以`CUDA_VISIBLE_DEVICES=''`执行原生SONIC服务`--help`，
实际在`fastapi`导入处失败；该环境也没有`tyro`、`qwen-vl-utils`，
Transformers4.56.1与Psi0原生4.57.1配方不同。因此这里不能声称原生Psi环境或模型推理已通过，
后续必须在独立Psi环境复验，不能向正在使用的ImageWAM环境直接混装依赖。
