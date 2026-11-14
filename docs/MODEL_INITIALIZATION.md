# 六个 policy 家族的基础权重、初始化与部署

本仓库提供代码、原生入口和配置，不附带模型权重。下面的下载命令供操作者执行，本次整理没有启动大模型下载。每种 policy 使用独立环境，先按 [维护说明](POLICY_MAINTENANCE.md) 准备固定版本源码。示例中 `MODEL_ROOT` 是自行指定的绝对路径。
六个家族共八个注册入口，ImageWAM motion76与joint41/47、FastWAM官方任务与G1 feedback76分别维护。新增Fast-WAM未做GPU推理，joint-reference尚无已训练权重。

| Policy | 从基础模型开始训练 | 已训练模型的部署材料 | 固定来源与限制 |
|---|---|---|---|
| GR00T N1.7 | `nvidia/GR00T-N1.7-3B`，按目标 embodiment 新建训练 recipe | 完整 HF checkpoint、processor、该 checkpoint 的 modality 与归一化统计 | 基础权重 revision `2fc962b973bccdd5d8ce4f67cc63b264d6886495`。基础模型不含 `UNITREE_G1_SONIC`，不能改标签后直接当 SONIC 模型 |
| π₀.₅ | OpenPI `gs://openpi-assets/checkpoints/pi05_base/params` | 任务 checkpoint 的 `params/` 与 `assets/`，匹配原生 config | GCS 路径没有本仓库核实的不可变权重 SHA。固定 OpenPI 源码与 A2A patch，下载后另存文件校验清单 |
| ImageWAM Δ-flow | FLUX.2 klein-base-4B、FLUX.2-dev AE、Qwen3-4B，生成基础模型派生的动作共享初始化 | 任务 MoT 权重、model config、统计、文本缓存，以及 factory 所需基础文件 | motion76默认预测16；joint47 v2历史16、预测40含前瞻尾、执行16。历史motion H40和joint v1必须使用各自协议 |
| Fast-WAM | Wan2.2-TI2V-5B video DiT、原始VAE/T5及Wan tokenizer，生成动作共享backbone初始化 | 官方任务完整MoT`.pt`＋对应统计，另需VAE/T5/tokenizer | 官方HF revision `8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`；LIBERO A7/S8/H32、RoboTwin A14/S14/H32，原生Gaussian FM |
| DiT4DiT | Cosmos-Predict2.5-2B，原生 recipe 构造动作专家 | 任务 `.pt`、上级 run 目录的 `config.yaml` 与 `dataset_statistics.json`，另需 Cosmos pipeline | 官方 LIBERO checkpoint 为 A8 / state16 / H8。Cosmos 是 gated 模型，当前账号访问核验为 HTTP403 |
| Ψ₀ / Psi Zero | 官方 human-video VLM或SONIC VLM，动作Transformer按匹配维度初始化微调 | 与无颈部profile匹配的完整run、指定step的safetensors、统计和CLIP缓存 | 默认`g1_neckless43_action78`为原生state43/action78/H30，hands14且无gravity；历史带neck的45/80 run不能直接部署 |

## 使用前先区分三件事

- **基础初始化**只加载指定的预训练参数，然后为新任务建立投影、统计和调度。它不代表继承了其他机器人任务的训练效果。
- **继续训练**只有在恢复完整 optimizer、scheduler、step、随机数与 sampler 等训练状态时才是精确 resume。只有模型权重时，应记为模型初始化后的新训练阶段，重新记录学习率和步数。
- **部署**通常只需模型参数、配置、归一化统计及原生推理依赖，不需要庞大的 optimizer state。模型文件和统计必须来自同一任务与版本。

共享层可以在维度兼容时继承。输入、输出、state 投影或 embodiment 变更时，必须显式列出新初始化层和未加载键，检查形状与数值，不能静默忽略 mismatch，也不能用零填充假装得到已训练的新动作空间。未经训练的 H16 recipe 也不能替代 H40 checkpoint。

## 1. GR00T N1.7

官方来源为 [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)、[N1.7 基础权重](https://huggingface.co/nvidia/GR00T-N1.7-3B) 和 [Cosmos-Reason2-2B](https://huggingface.co/nvidia/Cosmos-Reason2-2B)。固定原生源码 commit `7aa0bca50f1097f8b17f5c274c98264b72a1419f`。

```bash
hf download nvidia/GR00T-N1.7-3B --revision 2fc962b973bccdd5d8ce4f67cc63b264d6886495 --local-dir "$MODEL_ROOT/groot-n17-base"
hf download nvidia/Cosmos-Reason2-2B --revision 9ce19a195e423419c349abfc86fd07178b230561 --include '*.json' '*.jinja' '*.txt' '*.model' --local-dir "$MODEL_ROOT/cosmos-reason2-assets"
```

第二条仅准备 config、processor 和 tokenizer 资产，不宣称另下载了一套 Cosmos backbone 权重。原生 GR00T checkpoint 自身已包含模型参数，实际加载是否还缺资产要以固定版本的 processor 为准。离线时建立单独的 checkpoint metadata view，将配置中 `model_name` 的 locator 改为本地 Cosmos 资产目录，权重保持指向原始不可变文件，原始配置、统计和 embodiment 不改。

训练用官方 `gr00t/experiment/launch_finetune.py`，其 `--base-model-path` 指向基础 checkpoint，并选择真正匹配数据的 embodiment/modality recipe。部署由 `Gr00tPolicy` 读取训练后完整目录。命令参数见 [GR00T 维护文档](policies/GROOT_N17.md)，不能将基础权重与另一任务的统计拼接。

本次真实基础权重验证使用其已有的 `real_g1_relative_eef_relative_joints`，H40，已验证 GPU forward 与原生 ZMQ；这不等于 SONIC76 或任务闭环成功。SONIC 需要对应训练后的 checkpoint。本仓库提供的新 feedback76/H16 配置只是训练配置。

## 2. π₀.₅

官方来源为 [OpenPI](https://github.com/Physical-Intelligence/openpi)。固定源码 `215abfb217dbac7d5f1273282331b9b1866c0479` 的训练配置使用 `CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params")`。在其独立环境中可用原生下载器准备整个基础目录。

```bash
cd /path/to/third_party/openpi
python -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_base/'))"
python scripts/train.py pi05_libero_a2a_jax_lora_warmup_cosine --exp-name libero_delta
```

先用对应数据和原生统计脚本生成任务 normalization assets，再训练。命名 config 的 weight loader 会加载基础 `params`。部署原生 `create_trained_policy` 时指向任务 step 目录，由其同时读取 `params/` 与 `assets/`。不能把 GCS 基础目录冒充完成任务训练的目录。

维护中的 A2A patch 为 41,245 bytes，SHA256 `d1e5091747129d5c1ec1529b1e4d29b87ee62f729cb04ce3e0e7cc08f70122f8`。它支持 LIBERO action7 / state8 / H10，基座 action capacity32。G1 motion64＋hands12 需要真正扩展和训练，现有 adapter 会拒绝直接套用。已有 R1Lite Torch 发布包也不是 JAX `params` 格式，必须使用其原始 runtime、tokenizer 和统计。详见 [π₀.₅ 文档](policies/PI05.md)。

## 3. ImageWAM Δ-flow

原生基础实现来自 [FLUX.2](https://github.com/black-forest-labs/flux2)，本集成固定 `50fe5162777813d869182b139e83b10743caef15`。三个独立模型为 [klein-base-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B)、[FLUX.2-dev AE](https://huggingface.co/black-forest-labs/FLUX.2-dev) 和 [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B)。以下是本次公开 metadata 核验得到的固定 revision。

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B --revision a3b4f4849157f664bdbc776fd7453c2783562f4d flux-2-klein-base-4b.safetensors --local-dir "$MODEL_ROOT/flux2-klein-base-4b"
hf download black-forest-labs/FLUX.2-dev --revision 26afe3a78bb242c0a8bb181dcc8937bb16e5c66c ae.safetensors --local-dir "$MODEL_ROOT/flux2-dev-ae"
hf download Qwen/Qwen3-4B --revision 1cfa9a7208912126459214e8b04321603b3df60c --local-dir "$MODEL_ROOT/qwen3-4b"
```

新实验可以使用上述固定版本，复现已有发布权重时则先核对发布 manifest 的实际基础文件 SHA，不能推断新 revision 与旧训练时的字节相同。AE 为 gated 模型，需账号本身已获得访问权。

在 HumanWAM 根目录生成动作共享初始化，`FLUX2_SRC`、`FLUX2_MODEL_PATH` 和 `WORK_DIR` 均先设为真实绝对路径。

```bash
python scripts/flux2/preprocess_action_dit_flux2.py \
  --model-config configs/wholebody/deltaflow76_p25.yaml \
  --flux2-src-path "$FLUX2_SRC" --flux2-model-path "$FLUX2_MODEL_PATH" \
  --variant klein-base-4b --action-dim 76 --apply-alpha-scaling true \
  --device cpu --output "$WORK_DIR/action_shared_flux2_4b.pt"
```

它从基础 FLUX.2 转换 114 个共享张量，76D 动作输入/输出及 44D state 投影等六个张量新初始化。历史文件即便叫 `action_dit_flux2_4b_libero_init.pt`，也不能据此称它是训练过的 LIBERO 模型。历史参考文件 SHA256 为 `f938e28ac6fbcbcbb89ecbe441d69b93ab4fb36bb16dc05104dd95edcf0f66a3`，新生成文件应计算自己的 SHA，不能要求序列化结果与旧文件字节相同。

随后运行 `scripts/wholebody/cache_text.py` 缓存本任务 instruction，`scripts/wholebody/build_config.py` 固定 initializer SHA、数据统计及 H16，再以 `scripts/wholebody/train.py` 训练。完整参数和八卡命令见 [全身训练文档](G1_WHOLEBODY_DELTA.md)。state 只进入动作分支，视觉 source 使用当前图像 latent 与噪声；不要把 state 塞入视觉 text token。

部署保留本任务 MoT checkpoint、原配置、统计和文本缓存，按原 factory 加载。新格式 `wholebody_inspire76_p25_flux2_v1` 与历史实验专用格式分别处理，不能只换 YAML 就互换 checkpoint。

新增[关节参考配方](G1_JOINT_REFERENCE_DELTA.md)为独立的`imagewam_joint_reference`入口。
主路线joint47 v2目标是body29＋Inspire12＋相对当前IMU heading的root rotation6D，
历史16、预测40（包含24点前瞻尾）、执行16个源采样间隔。41D外部姿态与世界系47D仅保留legacy v1兼容。
不能把完整76D任务checkpoint直接当41/47D模型。仅复用兼容的FLUX.2基础派生114个共享张量，新建对应动作输入、输出
及state投影，重新拟合该任务统计，并在初始化记录中列明新层与共享层。上面的76D命令
是旧motion配方示例，新路径应使用其独立配置，不能只改输出维度后省略训练。
新增路径尚无训练结果。部署输出经[SONIC离线桥接](policies/SONIC_JOINT_REFERENCE.md)
显式处理未来root姿态、采样时间和`body_target_source`，导出文件不等于已验证控制链。
v2必须提供显式source contract，将示范实测关节、decoder电机目标与真正上游运动参考分别记录。
当前录制数据可用的实测关节和IMU姿态来源见[来源审计](policies/REFERENCE_LABEL_AUDIT_20261110.md)。
选择teleop姿态来源时，录制器的identity fallback仍按其有效性标志屏蔽；
采用实测IMU时以对应契约检查有效性，不冒用teleop的SMPL帧号规则。

## 4. DiT4DiT

官方 [DiT4DiT](https://github.com/Mondo-Robotics/DiT4DiT) 固定 commit `66a6f3a12e2c8740157c1e478795952c040d31dd`。基础 [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B) 的 `diffusers/base/post-trained` 已解析为下列固定 SHA，任务模型来自 [mondo-robotics/dit4dit-model](https://huggingface.co/mondo-robotics/dit4dit-model)。

```bash
hf download nvidia/Cosmos-Predict2.5-2B --revision 0d37c7498f54cee3c599d438d895a0a4a8608064 --local-dir "$MODEL_ROOT/cosmos-predict25-2b"
hf download mondo-robotics/dit4dit-model --revision 46237aebd3df427fcfb6a8ddc5ba5a3ab7b04a44 --include 'dit4dit_libero/*' --local-dir "$MODEL_ROOT/dit4dit"
```

官方任务文件 `dit4dit_libero/final_model/pytorch_model.pt` 为 21,283,256,413 bytes，SHA256 `10cdc70029d84519942157ff741fe570a78f161be3fda6ff1e87286b4c7faa29`。Cosmos 仓库约 21.23 GB，原生冷启动两者合计约 42.51 GB。2026-11-09 实际访问检查的 Cosmos 返回 HTTP403，未自动接受条款，也未下载这些大文件，因此未声称完成 DiT4DiT 真实权重推理。

原生 `baseframework.from_pretrained` 先构造 Cosmos 再载入任务 `.pt`。保留 `dit4dit_libero/config.yaml`、`dataset_statistics.json` 和 `final_model/pytorch_model.pt` 的相对层级，因为配置查找使用 checkpoint 的 `parents[1]`。在独立运行副本中将 `framework.cosmos25.base_model` 改为实际本地 Cosmos 路径，保存原配置和新配置的校验记录，不修改维度与统计。

从基础开始时使用原生 YAML 和 `DiT4DiT/training/train.py --config_yaml ...`，动作专家由 factory 构造；只有显式配置 `trainer.pretrained_checkpoint` 才加载已有任务参数。使用独立环境和 diffusers commit `3996788b602eaae4da41a1d45726b62e662b73cf`，不要升级 GR00T 环境中的 diffusers。具体命令构建和已修复的原生日志问题见 [DiT4DiT 文档](policies/DIT4DIT.md)。

## 5. Ψ₀ / Psi Zero

官方 [Psi0](https://github.com/physical-superintelligence-lab/Psi0) 固定 commit `cda3fdfafe47cb6fa75400d013de1bd36e3cea05`，权重统一来自 [USC-PSI-Lab/psi-model](https://huggingface.co/USC-PSI-Lab/psi-model)，本次 metadata 固定 revision `74d7abaa61b51dfe4e95f539a83738c276ce27b2`。HumanWAM模板默认使用`g1_neckless43_action78`，即原生身体29D＋手部14D state，以及motion token64D＋手部14D action，不含neck或projected gravity。它不是ImageWAM的state44/action76，后者尚未做Psi0适配。

以下两组用于预训练初始化，按需要下载，保持其目录结构。它们不等于已经完成当前无颈部任务训练的完整run。

```bash
hf download USC-PSI-Lab/psi-model --revision 74d7abaa61b51dfe4e95f539a83738c276ce27b2 --include 'psi0/pre.fast.2605160748.ckpt.ego390k/*' --local-dir "$MODEL_ROOT/psi0"
hf download USC-PSI-Lab/psi-model --revision 74d7abaa61b51dfe4e95f539a83738c276ce27b2 --include 'psi0/postpre.sonic1.1.unifolm.2609181726.40k/*' --local-dir "$MODEL_ROOT/psi0"
```

`pre.fast...` 是 human-video VLM，`model.safetensors` 约 4.26 GB，没有完整任务动作专家。`postpre.sonic...40k` 是已导出的 SONIC 初始化目录，VLM 文件约 8.53 GB，另有约 2.70 GB 的 `action_header.safetensors`。微调原生 `finetune_sonic_psi0_config` 时分别传 `--model.model_name_or_path=POST_DIR` 和 `--model.pretrained-action-header-path=POST_DIR`，无需把这个已导出目录再假装转换成 run checkpoint。如果原action维度或chunk不匹配，原生训练器仅加载动作头`transformer_blocks`，相关投影新初始化；必须记录实际加载与新初始化的层，不能称为无缝resume或直接裁切旧权重。

部署当前无颈部profile时，`PSI0_RUN_DIR`必须是实际按state43/action78训练的完整任务run，并保留`argv.txt`、`run_config.json`、指定step的checkpoint、任务统计与文本缓存。只改模板或维度不产生可用的新权重。

已核对的无颈部官方完整run为`sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223`。
可用下列命令准备默认示例，随后将`PSI0_RUN_DIR`设为下载出的完整run目录：

```bash
hf download USC-PSI-Lab/psi-model --revision 74d7abaa61b51dfe4e95f539a83738c276ce27b2 --include 'psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223/*' --local-dir "$MODEL_ROOT/psi0"
export PSI0_RUN_DIR="$MODEL_ROOT/psi0/psi0/simple-checkpoints/sonic-wbcbox.neckle.flow1000.cosine.lr1.0e-04.b256.gpus8.2608260223"
```

它是官方carry-box任务示例，不是本项目其他任务的新训练权重。checkpoint40000为state43/action78/H30、observation horizon1，camera为`observation.images.egocentric`，resize/crop240×320，43/78维统计嵌入run配置；state保留原始字段顺序，动作按保存的repack顺序读取。远端权重声明为6,254,227,440字节，header中的state/action投影与H30位置参数尺寸也匹配，尚未下载整文件验证其哈希。详见[无颈部模型验证记录](policies/PSI0_NECKLESS_VALIDATION_20261110.md)。

这是SONIC v1的`action.body_token`权重，不能当作v1.1的`action.body_token_v1_1`权重；相同64D宽度不代表token语义相同。其保存配置`model.rtc=false`，服务的`--rtc`只是test-time guidance，不是RTC训练记录。本次只核对metadata、header与加载路径，未进行该权重的GPU推理或真机验证。

此前核对的`multi-task.psi-dream.2609092156`是带neck的历史完整run，约11.22GB，真实metadata为A80/raw state45/H30。它仅用于说明不兼容，不是当前模板的默认权重。不能裁切其权重/输出、补零state，或改配置后冒充43/78模型。adapter会交叉检查当前profile、run配置与实际normalization布局。

新训练模板默认`action.body_token_v1_1 + action[:14]`、`observation.state`，并同时绑定model、repack、field的43/78维度和horizon。v1数据须显式改`options.action_keys`为`["action.body_token", "action[:14]"]`；已正确组装的78维`action`列可设`["action"]`。统计字段顺序必须同步，launcher拒绝受管参数冲突或重复。详见[Psi0训练参数](policies/PSI0.md#训练)。

原生加载还用 `Qwen/Qwen3-VL-2B-Instruct` 的 AutoConfig/AutoProcessor，新 instruction 的 CLIP cache miss 也需要其原生文本编码器资产。分别准备原生所需缓存并记录版本，不可用 Qwen3-4B 或其他 policy 的 tokenizer 替代。原生部署入口为 `python -m psi.deploy.serve_psi0_sonic --policy psi --rtc --run-dir RUN --ckpt-step 40000`，详见 [Ψ₀ 文档](policies/PSI0.md) 与 [官方 SONIC 说明](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/examples/psi0_for_sonic.md)。

Ψ₀ 的 SONIC 子模块使用其 [专用 WBC fork](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl) 的 `d40376994ff094787591ab76a691c50b8e131786`，不能替换为本项目顶部 NVLabs WBC 后宣称等价。本次仅验证公开 metadata、adapter 与 native API/CLI 依赖边界，没有加载 Ψ₀ 大权重或执行机器人。

## 6. Fast-WAM

G1 feedback76/H16 入口需要独立训练的完整 MoT/proprio 权重及 action76/state44 统计，
不能复用下述7/14维发布权重。初始化与 manifest 登记步骤见
[FastWAM SONIC](policies/FASTWAM_SONIC.md#新训练)。

官方源码 [FastWAM](https://github.com/yuantianyuan01/FastWAM)固定commit
`7faa71108368fbb3b6885649f112af607427a2d4`。任务权重来自
[yuanty/fastwam](https://huggingface.co/yuanty/fastwam)，固定revision
`8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`，本入口保留官方Gaussian FM，未混入历史A2A、IC或sparse-joint修改。

| 官方任务profile | 配对权重和统计 | 维度 |
|---|---|---|
| LIBERO | `libero_uncond_2cam224.pt` / `libero_uncond_2cam224_dataset_stats.json` | A7 / state8 / H32 |
| RoboTwin | `robotwin_uncond_3cam_384.pt` / `robotwin_uncond_3cam_384_dataset_stats.json` | A14 / state14 / H32 |

两个任务权重各约12.04GB，不包含VAE和T5。保持原生目录层级，以下载命令中的
`FASTWAM_MODEL_CACHE`作为`DIFFSYNTH_MODEL_BASE_PATH`。以下仅准备LIBERO部署材料，
RoboTwin应成对更换任务权重与统计，不能按相同horizon混用。

```bash
export FASTWAM_MODEL_CACHE="$MODEL_ROOT/fastwam-native-assets"
hf download yuanty/fastwam libero_uncond_2cam224.pt libero_uncond_2cam224_dataset_stats.json --revision 8eaceeb24c3cc92ff2a9c9a9d266a4941b836705 --local-dir "$MODEL_ROOT/fastwam-release"
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth models_t5_umt5-xxl-enc-bf16.pth --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e --local-dir "$FASTWAM_MODEL_CACHE/Wan-AI/Wan2.2-TI2V-5B"
hf download Wan-AI/Wan2.1-T2V-1.3B --include 'google/umt5-xxl/*' --revision 37ec512624d61f7aa208f7ea8140a131f93afc9a --local-dir "$FASTWAM_MODEL_CACHE/Wan-AI/Wan2.1-T2V-1.3B"
```

上述[Wan2.2](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)与
[Wan2.1 tokenizer](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B)均已核验metadata。
VAE约2.82GB，T5约11.36GB。此入口使用`redirect_common_files=false`的原始资产，
不依赖当前返回HTTP401的默认DiffSynth转换仓库。adapter禁止自动下载，缺本地资产即报错。

部署完整任务`.pt`时，原生工厂跳过video/action DiT初始化，因此不额外要求基础DiT分片
和ActionDiT initializer。新训练则还需Wan2.2的`diffusion_pytorch_model*.safetensors`
分片，以及原生`scripts/preprocess_action_dit_backbone.py`生成的基础派生动作backbone。
该initializer不是训练过的LIBERO策略，定制动作维度必须显式设置，不能依赖脚本默认7D。
任务文本缓存由`scripts/precompute_text_embeds.py`生成，训练入口为Hydra `scripts/train.py`。

精确文件SHA、原生缓存目录、配置及命令见[Fast-WAM文档](policies/FASTWAM.md)。
任务checkpoint、统计、相机布局与sigma-shift必须配对；这一入口不生成G1/H2动作，
原生state会进入视觉text context，与本项目ImageWAM的state仅进动作分支变体不同。
已完成公开metadata、小统计和CPU契约检查，尚未下载大权重、进行GPU forward或训练。

## 许可证与发布材料

源码许可证不自动覆盖模型、数据或上游 VLM 的使用条款。下载 gated 模型前，由账号持有人查看并接受官方页面条款或申请访问，本仓库不会自动代为接受。HF 登录在本机或服务器的私有凭据存储中完成，不把 token、代理认证、`.env`、私有路径或账号缓存打包进 Git、配置样例和公开模型仓库。

每次发布至少记录源码 commit、权重 revision 或文件 SHA、实际动作/state 顺序、horizon、相机、归一化来源和初始化层清单。部署包包含自己的推理脚本与必要资产，optimizer state 按训练恢复需求另行保管，不能将“省略训练状态的部署包”描述成可精确恢复训练的完整 checkpoint。
